"""
Streamlit application for building a contextual retrieval ETL pipeline.
Processes PDFs and generates contextual embeddings using Claude, Voyage AI, and Qdrant.
"""

import streamlit as st
import anthropic
import voyageai
from llama_parse import LlamaParse
from llama_index.embeddings.voyageai import VoyageEmbedding 
import tempfile
import shutil
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
from urllib.parse import unquote
import json
from pathlib import Path
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from typing import List, Dict, Any, Optional, Union
import numpy as np
from qdrant_client import QdrantClient, models
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from ratelimit import limits, sleep_and_retry
import sentry_sdk
from prometheus_client import Counter, Histogram
import nest_asyncio
import pandas as pd
nest_asyncio.apply()

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Local imports
from init_utils import initialize_qdrant
from qdrant_adapter import QdrantAdapter, VECTOR_DIMENSIONS

# Set up logging
logger = logging.getLogger(__name__)

# Configuration defaults and constants
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_EMBEDDING_MODEL = "voyage-finance-2"
DEFAULT_QDRANT_URL = "https://3efb9175-b8b6-43f3-aef4-d2695ed84dc6.europe-west3-0.gcp.cloud.qdrant.io"
DEFAULT_LLM_MODEL = "claude-3-haiku-20240307"

DEFAULT_CONTEXT_PROMPT = """Give a short succinct context to situate this chunk within the overall enclosed document broader context for the purpose of improving similarity search retrieval of the chunk. 

Provide a very succint high level overview (i.e. not a summary) of the chunk's content in no more than 100 characters with a focus on keywords for better similarity search retrieval

Answer only with the succinct context, and nothing else (no introduction, no conclusion, no headings).

Example:
Chunk is part of a release of Saint Gobain Q3 2024 results emphasizing Saint Gobain's performance in construction chemicals in the US market, price and volumes effects, and operating leverage. 
"""

# Add after other constants (around line 51)
metrics_template = {
    'start_time': None,
    'total_docs': 0,
    'processed_docs': 0,
    'total_chunks': 0,
    'processed_chunks': 0,
    'errors': 0,
    'stages': {
        'context': {'success': 0, 'failed': 0},
        'dense_vectors': {'success': 0, 'failed': 0},
        'sparse_vectors': {'success': 0, 'failed': 0},
        'upserts': {'success': 0, 'failed': 0}
    }
}

# Add after VECTOR_DIMENSIONS (around line 51)
CALLS_PER_MINUTE = {
    'anthropic': 50,
    'voyage': 100,
    'qdrant': 100
}

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE['anthropic'], period=60)
def rate_limited_context(func, *args, **kwargs):
    return func(*args, **kwargs)

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE['voyage'], period=60)
def rate_limited_embedding(func, *args, **kwargs):
    return func(*args, **kwargs)

# Utility Functions
def cleanup_temp_files():
    """Clean up any temporary files"""
    temp_dir = Path('.temp')
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Error cleaning temp files: {str(e)}")

def cleanup_session_state():
    """Clean up session state and resources on app restart"""
    try:
        if 'clients' in st.session_state:
            for client in st.session_state.clients.values():
                if hasattr(client, 'close'):
                    client.close()
        cleanup_temp_files()
        st.session_state.clear()
    except Exception as e:
        logger.error(f"Error in cleanup: {str(e)}")

def initialize_session_state():
    """Initialize all session state variables with default values."""
    if not st.session_state.get('clients'):
        st.session_state.clients = {}
    if not st.session_state.get('collection_name'):
        st.session_state.collection_name = "documents"
    if not st.session_state.get('chunk_size'):
        st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
    if not st.session_state.get('chunk_overlap'):
        st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP
    if not st.session_state.get('processed_urls'):
        st.session_state.processed_urls = set()
    if not st.session_state.get('processing_metrics'):
        st.session_state.processing_metrics = {
            'start_time': None,
            'total_docs': 0,
            'processed_docs': 0,
            'total_chunks': 0,
            'processed_chunks': 0,
            'errors': 0,
            'stages': {
                'context': {'success': 0, 'failed': 0},
                'dense_vectors': {'success': 0, 'failed': 0},
                'sparse_vectors': {'success': 0, 'failed': 0},
                'upserts': {'success': 0, 'failed': 0}
            },
            'documents_processed': 0,
            'chunks_created': 0,
            'embedding_time': 0,
            'total_tokens': 0
        }

def validate_environment():
    """Validate all required environment variables are set"""
    required_vars = [
        "ANTHROPIC_API_KEY",
        "VOYAGE_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "LLAMAPARSE_API_KEY"
    ]
    missing = [var for var in required_vars if not st.secrets.get(var)]
    if missing:
        st.error(f"Missing required environment variables: {', '.join(missing)}")
        st.stop()

def initialize_clients() -> bool:
    """Initialize all required clients in the correct order."""
    try:
        # First validate environment
        if not st.secrets.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found in secrets")
        
        # Initialize Anthropic client first
        st.write("Initializing Anthropic client...")
        try:
            anthropic_client = anthropic.Client(
                api_key=st.secrets["ANTHROPIC_API_KEY"].strip()
            )
            
            if not validate_anthropic_client(anthropic_client):
                raise ValueError("Failed to validate Anthropic client")
            
            st.session_state.clients['anthropic'] = anthropic_client
            st.write("Anthropic client created and validated successfully")
            
        except Exception as e:
            st.error(f"Anthropic client initialization failed: {str(e)}")
            raise
        
        # Then initialize Voyage
        st.write("Initializing VoyageEmbedding...")
        voyage_embed = VoyageEmbedding(
            model_name=DEFAULT_EMBEDDING_MODEL,
            voyage_api_key=st.secrets["VOYAGE_API_KEY"].strip()
        )
        st.session_state.clients['embed_model'] = voyage_embed
        st.write("VoyageEmbedding created successfully")

        # Initialize QdrantAdapter last with the anthropic client
        st.write("Initializing QdrantAdapter...")
        qdrant_adapter = QdrantAdapter(
            url=st.secrets["QDRANT_URL"].strip(),
            api_key=st.secrets["QDRANT_API_KEY"].strip(),
            collection_name=st.session_state.collection_name,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            anthropic_client=st.session_state.clients['anthropic']
        )
        st.session_state.clients['qdrant'] = qdrant_adapter
        st.write("QdrantAdapter created successfully")

        return validate_clients()
        
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return False

def validate_clients():
    """Validate that all required clients are initialized."""
    if not st.session_state.get('clients'):
        return False
    return all(client in st.session_state.clients 
              for client in ['anthropic', 'embed_model', 'qdrant'])

# Processing Functions
def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string"""
    try:
        token_count = st.session_state.clients['anthropic'].count_tokens(text)
        return token_count
    except Exception as e:
        st.error(f"Error counting tokens: {str(e)}")
        return 0

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception(lambda e: not isinstance(e, KeyboardInterrupt))
)
async def generate_document_context(text: str) -> Any:
    """Generate context for the entire document using prompt caching."""
    try:
        response = await st.session_state.clients['anthropic'].beta.prompt_caching.messages.create(
            model=DEFAULT_LLM_MODEL,
            max_tokens=300,
            system=[{
                "type": "text",
                "text": "Analyze this document to understand its overall context.",
                "cache_control": {"type": "ephemeral"}
            }],
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": text[:2000]}]
            }]
        )
        return response
    except Exception as e:
        logger.error(f"Error generating document context: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception(lambda e: not isinstance(e, KeyboardInterrupt))
)
async def generate_chunk_context(chunk_text: str, doc_context: str) -> str:
    """Generate context for a chunk using the cached document context."""
    try:
        # Combine the prompts
        system_prompt = f"{anthropic.AI_PROMPT} {st.session_state.context_prompt}"
        user_prompt = f"\n\n{anthropic.HUMAN_PROMPT} Document context:\n{doc_context}\n\nChunk text:\n{chunk_text}"
        full_prompt = f"{system_prompt}{user_prompt}{anthropic.AI_PROMPT}"

        # Call the completions.create method with the cache parameter
        response = await st.session_state.clients['anthropic'].acompletions.create(
            model=st.session_state.llm_model,
            max_tokens_to_sample=300,
            prompt=full_prompt,
            cache="ephemeral"
        )

        # Extract the assistant's reply
        result = response['completion'].strip()

        st.session_state.processing_metrics['stages']['context']['success'] += 1
        return result

    except Exception as e:
        logger.error(f"Error generating chunk context: {str(e)}")
        st.session_state.processing_metrics['stages']['context']['failed'] += 1
        raise

async def process_urls_async(urls: List[str]):
    try:
        # Initialize metrics at start
        st.session_state.processing_metrics = {
            'start_time': datetime.now(),
            'total_docs': len(urls),
            'processed_docs': 0,
            'total_chunks': 0,
            'processed_chunks': 0,
            'documents_processed': 0,
            'chunks_created': 0,
            'embedding_time': 0,
            'total_tokens': 0
        }
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each URL with proper metrics updates
        for idx, url in enumerate(urls):
            try:
                result = await process_url(url)
                if result:
                    st.session_state.processing_metrics['processed_docs'] += 1
                    st.session_state.processing_metrics['chunks_created'] += len(result['chunks'])
                    st.session_state.processing_metrics['total_tokens'] += sum(len(chunk['text'].split()) for chunk in result['chunks'])
                    st.session_state.processing_metrics['embedding_time'] += result.get('embedding_time', 0)
                    
                # Update progress
                progress = (idx + 1) / len(urls)
                progress_bar.progress(progress)
                status_text.text(f"Processed {idx + 1}/{len(urls)} documents")
                
            except Exception as e:
                st.error(f"Error processing URL {url}: {str(e)}")
                continue
        
        # Final update to indicate completion
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
                
    except Exception as e:
        st.error(f"Batch processing error: {str(e)}")

def save_processed_urls(urls: set) -> None:
    """Save processed URLs to persistent storage"""
    try:
        with open('processed_urls.json', 'w') as f:
            json.dump(list(urls), f)
    except Exception as e:
        logger.error(f"Error saving processed URLs: {str(e)}")

def display_metrics():
    """Display detailed processing metrics."""
    if 'processing_metrics' not in st.session_state:
        st.warning("No processing metrics available")
        return

    metrics = st.session_state.processing_metrics
    
    # Document Processing Progress
    st.write(f"Processed {metrics['processed_docs']} of {metrics['total_docs']} documents")
    
    # Detailed Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Documents Processed", metrics['processed_docs'])
        st.metric("Total Chunks", metrics['total_chunks'])
        st.metric("Errors", metrics.get('errors', 0))
        
    with col2:
        st.metric("Total Tokens", metrics.get('total_tokens', 0))
        st.metric("Chunks Created", metrics['processed_chunks'])
        
    # Stage-wise Success/Failure Metrics
    st.subheader("Processing Stages")
    stages_df = pd.DataFrame({
        'Stage': list(metrics['stages'].keys()),
        'Success': [s['success'] for s in metrics['stages'].values()],
        'Failed': [s['failed'] for s in metrics['stages'].values()]
    })
    st.dataframe(stages_df)
    
    # Processing Time
    if metrics.get('start_time'):
        elapsed = datetime.now() - metrics['start_time']
        st.metric("Processing Time", f"{elapsed.total_seconds():.2f}s")

    # Add any warnings or errors
    if metrics.get('errors', 0) > 0:
        st.warning(f"Encountered {metrics['errors']} errors during processing")

async def process_chunks_async(chunks: List[Dict[str, Any]], metadata: Dict[str, Any], full_document: str) -> List[Dict[str, Any]]:
    """Process chunks asynchronously with proper embedding and context generation."""
    try:
        processed_chunks = []
        total_chunks = len(chunks)
        
        # Create progress indicators
        chunk_progress = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Processing {total_chunks} chunks...")
        
        # Generate document-level context first
        try:
            doc_context = await rate_limited_context(
                st.session_state.clients['anthropic'].messages.create,
                model=DEFAULT_LLM_MODEL,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": f"Document text:\n{full_document[:2000]}"
                }]
            )
            logger.info("Generated document-level context")
        except Exception as e:
            logger.error(f"Error generating document context: {str(e)}")
            raise

        # Process chunks with batching for better performance
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_tasks = []
            
            for chunk in batch:
                try:
                    # Generate context
                    context = await generate_chunk_context(chunk['text'], doc_context)
                    
                    # Generate embedding using Voyage
                    embedding = st.session_state.clients['embed_model'].embed(
                        texts=[chunk['text']],
                        model=DEFAULT_EMBEDDING_MODEL
                    )
                    
                    # Access embedding values correctly
                    dense_vector = embedding[0]  # Get first embedding array
                    
                    # Upsert to Qdrant
                    success = await st.session_state.clients['qdrant'].upsert_chunk(
                        chunk_text=chunk['text'],
                        context_text=context,
                        dense_embedding=dense_vector,
                        metadata=metadata,
                        chunk_id=str(i)
                    )
                    
                    if success:
                        processed_chunks.append({
                            'text': chunk['text'],
                            'context': context,
                            'embedding': dense_vector
                        })
                        st.session_state.processing_metrics['processed_chunks'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
                    st.session_state.processing_metrics['errors'] += 1
                    continue
                
            # Update progress
            progress = min((i + batch_size) / total_chunks, 1.0)
            chunk_progress.progress(progress)
            status_text.text(f"Processed chunks {i + 1}-{min(i + batch_size, total_chunks)} of {total_chunks}")
            
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Error in process_chunks_async: {str(e)}")
        raise

def validate_anthropic_client(client):
    """Validate the Anthropic client by making a test call."""
    try:
        # Test the client with a simple call
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "test"}]
        )
        return True
    except anthropic.APIStatusError as e:
        if e.status_code == 404:
            raise ValueError(f"API endpoint not found. Check base_url configuration")
        elif e.status_code == 401:
            raise ValueError("Invalid API key")
        else:
            raise ValueError(f"API error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Anthropic client validation failed: {str(e)}")

async def generate_context(text: str, anthropic_client) -> Optional[str]:
    """Generate context for a chunk using Claude with a user-modifiable prompt."""
    try:
        # Use the default context prompt from the code
        context_prompt = st.session_state.get('context_prompt', DEFAULT_CONTEXT_PROMPT)
        
        # Combine prompt with text
        full_prompt = f"{context_prompt}\n\nText to process:\n{text}"

        response = await anthropic_client.messages.create(
            model=DEFAULT_LLM_MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": full_prompt}]
        )
        
        return response.content[0].text.strip()
    except Exception as e:
        st.error(f"Error generating context: {str(e)}")
        return None

async def parse_document(url: str) -> Optional[Dict[str, Any]]:
    """Parse a document from a given URL using LlamaParse."""
    try:
        # Ensure API key is available
        api_key = st.secrets.get("LLAMAPARSE_API_KEY")
        if not api_key:
            raise ValueError("LLAMAPARSE_API_KEY not found in secrets")

        # Download the document
        response = requests.get(url)
        response.raise_for_status()

        # Save content to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name

        # Initialize LlamaParse with explicit API key
        parser = LlamaParse(
            api_key=api_key,
            result_type="text",
            num_workers=8,
            verbose=False,
            mode="FAST",
            parsing_instruction=(
                "this is a financial document. In case the detected language is not english, "
                "translate it to english"
            )
        )

        try:
            # Use asynchronous parsing
            document = await parser.aload_data(temp_path)
            
            if not document:
                raise ValueError("No text extracted from document")

            # Extract text from the document
            text = "\n\n".join([doc.text for doc in document])
            
            return {
                'text': text,
                'title': url.split('/')[-1],
                'source': url,
                'date_processed': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"LlamaParse extraction error: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error parsing document from {url}: {str(e)}")
        return None

    finally:
        # Clean up temporary file
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass

# Add this near the top of the file, after the imports and before the UI code
def parse_sitemap(url: str) -> List[str]:
    """Parse XML sitemap and return list of URLs."""
    try:
        logger.info(f"Fetching sitemap from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        urls = []
        namespaces = [
            './/{http://www.sitemaps.org/schemas/sitemap/0.9}loc',
            './/loc'  # Try without namespace
        ]
        
        for namespace in namespaces:
            urls.extend([unquote(url.text) for url in root.findall(namespace)])
        
        if not urls:
            logger.warning("No URLs found in sitemap")
        else:
            logger.info(f"Found {len(urls)} URLs in sitemap")
            
        return urls
            
    except Exception as e:
        logger.error(f"Error parsing sitemap: {str(e)}")
        raise

def create_semantic_chunks(text: str) -> List[Dict[str, Any]]:
    """Create semantic chunks from text using sentence boundaries and overlap."""
    try:
        # Split text into sentences
        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            sentence_length = len(sentence.split())
            
            # Check if adding the sentence exceeds the chunk size
            if current_length + sentence_length > st.session_state.chunk_size:
                if current_chunk:  # Save current chunk if it exists
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'length': current_length
                    })
                
                # Start a new chunk with overlap
                overlap_sentences = current_chunk[-st.session_state.chunk_overlap:] if st.session_state.chunk_overlap > 0 else []
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'length': current_length
            })
        
        # Update metrics
        st.session_state.processing_metrics['total_chunks'] += len(chunks)
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error creating semantic chunks: {str(e)}")
        raise

async def process_url(url: str) -> Optional[Dict[str, Any]]:
    """Process a single URL and return chunks and metadata"""
    try:
        # Download PDF
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        # Parse PDF with specific options
        parser = LlamaParse(
            api_key=st.secrets["LLAMAPARSE_API_KEY"],
            result_type="markdown",
            verbose=True,
            num_workers=4,
            language="en"
        )
        
        # Load and parse document
        docs = await parser.aload_data(temp_path)
        
        # Access text directly as property
        full_text = "\n\n".join(doc.text for doc in docs)
        
        if not full_text:
            raise ValueError("No text content found in document")
        
        # Extract metadata using QdrantAdapter
        metadata = st.session_state.clients['qdrant'].extract_metadata(full_text, url)
        chunks = create_semantic_chunks(full_text)
        
        return {
            "chunks": chunks,
            "metadata": metadata,
            "text": full_text
        }
        
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        raise
    finally:
        if 'temp_path' in locals():
            os.unlink(temp_path)

# Must be the first Streamlit command
st.set_page_config(page_title="Alpine ETL Processing Pipeline", layout="wide")

# Initialize session state and validate environment
try:
    # Initialize Sentry if configured
    if st.secrets.get("SENTRY_DSN"):
        sentry_sdk.init(dsn=st.secrets["SENTRY_DSN"])
    
    # Initialize session state
    try:
        if st.runtime.exists():
            cleanup_session_state()
            initialize_session_state()
            
            # Initialize clients after session state
            if not initialize_clients():
                st.stop()
                
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}")
        st.error(f"Error initializing session state: {str(e)}")
        st.stop()

except Exception as e:
    st.error(f"Error during initialization: {str(e)}")
    st.stop()

# Start UI code
st.title("Document Processing Pipeline")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    if st.button("Reset Metrics"):
        st.session_state.processing_metrics = metrics_template.copy()
        st.success("Metrics reset successfully")
    
    if st.button("Delete Collection"):
        try:
            st.session_state.clients['qdrant'].delete_collection()
            st.success("Collection deleted successfully")
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
    
    if st.button("Create Collection"):
        try:
            st.session_state.clients['qdrant'].create_collection(st.session_state.collection_name)
            st.success(f"Collection '{st.session_state.collection_name}' created successfully")
        except Exception as e:
            st.error(f"Error creating collection: {str(e)}")
    
    # Collection info
    st.header("Collection Info")
    try:
        if 'qdrant' in st.session_state.clients:
            info = st.session_state.clients['qdrant'].get_collection_info()
            st.write(info)
        else:
            st.error("Qdrant client not initialized")
    except Exception as e:
        st.error(f"Error getting collection info: {str(e)}")
    
    # Configuration
    st.header("Configuration")
    
    # Model settings
    st.subheader("Model Settings")
    embedding_model = st.selectbox(
        "Embedding Model",
        options=list(VECTOR_DIMENSIONS.keys()),
        index=list(VECTOR_DIMENSIONS.keys()).index(DEFAULT_EMBEDDING_MODEL),
        help="Select the embedding model to use"
    )
    
    llm_model = st.selectbox(
        "LLM Model",
        options=[
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229"
        ],
        index=0,
        help="Select the LLM model to use for context generation"
    )
    
    # Vector store settings
    st.subheader("Vector Store Settings")
    qdrant_url = st.text_input(
        "Qdrant URL",
        value=DEFAULT_QDRANT_URL,
        help="URL of your Qdrant instance"
    )
    
    # Chunking settings
    st.subheader("Chunking Settings")
    chunk_size = st.number_input(
        "Chunk Size",
        min_value=100,
        max_value=1000,
        value=DEFAULT_CHUNK_SIZE,  # Use default value initially
        key='chunk_size',  # This will automatically handle session state
        help="Number of tokens per chunk (default: 500)"
    )
    
    chunk_overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=DEFAULT_CHUNK_OVERLAP,  # Use default value initially
        key='chunk_overlap',  # This will automatically handle session state
        help="Number of overlapping tokens between chunks (default: 100)"
    )
    
    # Context prompt
    st.subheader("Context Settings")
    context_prompt = st.text_area(
        "Context Prompt",
        value=DEFAULT_CONTEXT_PROMPT,
        height=300,
        help="Prompt template for context generation"
    )

# Main content
tab1, tab2 = st.tabs(["Process Content", "Search"])

with tab1:
    st.header("Process Content")
    
    sitemap_url = st.text_input(
        "Enter XML Sitemap URL",
        value="https://contextrag.s3.eu-central-2.amazonaws.com/sitemap.xml"
    )
    
    if st.button("Process Sitemap"):
        try:
            with st.spinner("Parsing sitemap..."):
                urls = parse_sitemap(sitemap_url)
                if not urls:
                    st.warning("No URLs found to process")
                    st.stop()
                
                st.session_state.processing_metrics['total_docs'] = len(urls)
                st.session_state.processing_metrics['start_time'] = datetime.now()
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run async processing
                asyncio.run(process_urls_async(urls))
                
                save_processed_urls(st.session_state.processed_urls)
                st.success("Processing complete!")
                
        except Exception as e:
            st.error(f"Error processing sitemap: {str(e)}")
    
    # Display metrics
    st.header("Processing Metrics")
    display_metrics()

with tab2:
    st.header("Search Documents")
    
    query = st.text_input("Enter your search query")
    
    if query:
        try:
            # Generate query embedding using correct method
            query_embedding = st.session_state.clients['embed_model'].get_query_embedding(query)
            
            # Search using the embedding
            results = st.session_state.clients['qdrant'].search(
                query_text=query,
                query_vector=query_embedding,
                limit=5
            )
            
            if not results:
                st.info("No matching documents found.")
            else:
                # Display results
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i} (Score: {result['score']:.3f})"):
                        st.write("**Original Text:**")
                        st.write(result['payload']['chunk_text'])
                        st.write("**Context:**")
                        st.write(result['payload']['context'])
                        st.write("**Metadata:**")
                        display_metadata = {
                            k: result['payload'].get(k, "") 
                            for k in ["company", "date", "fiscal_period", "creation_date", "file_name", "url"]
                        }
                        st.json(display_metadata)
            
        except Exception as e:
            st.error(f"Error performing search: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Powered by Alpine")
