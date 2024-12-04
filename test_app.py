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
        "QDRANT_API_KEY"
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
def generate_context(chunk_text: str) -> str:
    """Generate contextual description for a chunk of text using Claude"""
    try:
        response = st.session_state.clients['anthropic'].messages.create(
            model=DEFAULT_LLM_MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"{DEFAULT_CONTEXT_PROMPT}\n\nText to process:\n{chunk_text}"
            }]
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Error generating context: {str(e)}")
        st.session_state.processing_metrics['stages']['context']['failed'] += 1
        raise

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
    """Create semantic chunks from text using sentence boundaries."""
    try:
        # Split into sentences and combine into chunks
        sentences = text.split('.')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > st.session_state.chunk_size:
                if current_chunk:  # Save current chunk if it exists
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'length': current_length
                    })
                current_chunk = [sentence]
                current_length = sentence_length
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

async def process_url(url: str) -> Optional[Dict]:
    """Process a single URL with context and embeddings."""
    try:
        # Parse document
        doc = await parse_document(url)
        if not doc:
            raise ValueError(f"Failed to parse document: {url}")

        # Generate document-level context
        doc_context = await generate_context(doc['text'], st.session_state.clients['anthropic'])
        if not doc_context:
            raise ValueError(f"Failed to generate document context for {url}")

        # Create chunks
        chunks = create_chunks(doc['text'])
        if not chunks:
            raise ValueError(f"No chunks created for {url}")

        # Process chunks with context and embeddings
        metadata = {
            'url': url,
            'title': doc.get('title', ''),
            'doc_context': doc_context,
            'creation_date': datetime.now().isoformat()
        }

        processed_chunks = await process_chunks_async(chunks, metadata, doc['text'])
        
        if processed_chunks:
            return {
                'chunks': processed_chunks,
                'metadata': metadata
            }
        
        return None

    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        raise

async def process_single_url(url: str, index: int, total: int) -> Optional[Dict[str, Any]]:
    """Process a single URL asynchronously"""
    try:
        # Download and process PDF
        result = await process_url(url)
        
        if result:
            # Process chunks
            processed_chunks = await process_chunks_async(
                result['chunks'],
                result['metadata'],
                result['text']
            )
            
            # Update metrics
            st.session_state.processing_metrics['processed_docs'] += 1
            st.session_state.processed_urls.add(url)
            
            # Update progress
            progress = (index + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Processed {index + 1}/{total} documents")
            
            return processed_chunks
            
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        st.session_state.processing_metrics['errors'] += 1
        return None

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
    """Display current processing metrics"""
    metrics = st.session_state.processing_metrics
    
    # Display overall progress
    if metrics.get('total_docs', 0) > 0:
        progress = metrics['processed_docs'] / metrics['total_docs']
        st.progress(progress)
        st.write(f"Processed {metrics['processed_docs']} of {metrics['total_docs']} documents")

    # Display stage-specific metrics
    if metrics.get('stages'):
        st.subheader("Processing Stages")
        cols = st.columns(4)
        
        for idx, (stage, data) in enumerate(metrics['stages'].items()):
            with cols[idx]:
                total = data['success'] + data['failed']
                if total > 0:
                    success_rate = (data['success'] / total) * 100
                    st.metric(
                        f"{stage.title()}",
                        f"{success_rate:.1f}%",
                        help=f"Success: {data['success']}, Failed: {data['failed']}"
                    )

    # Display detailed metrics
    if metrics.get('start_time'):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents Processed", metrics['documents_processed'])
            st.metric("Chunks Created", metrics['chunks_created'])
        with col2:
            st.metric("Total Tokens", metrics['total_tokens'])
            st.metric("Embedding Time (s)", round(metrics.get('embedding_time', 0), 2))

async def process_chunks_async(chunks: List[Dict[str, Any]], metadata: Dict[str, Any], full_document: str) -> List[Dict[str, Any]]:
    """Process chunks asynchronously with context and embeddings."""
    try:
        processed_chunks = []
        
        # Generate document-level context first
        doc_context = await generate_chunk_context(full_document[:2000], "")
        
        # Process chunks in batches
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Generate contexts for batch
            contexts = await asyncio.gather(*[
                generate_chunk_context(chunk['text'], doc_context)
                for chunk in batch
            ])
            
            # Generate embeddings for batch
            try:
                texts = [chunk['text'] for chunk in batch]
                embeddings = st.session_state.clients['embed_model'].embed(
                    texts,
                    model=DEFAULT_EMBEDDING_MODEL
                )
                st.session_state.processing_metrics['stages']['dense_vectors']['success'] += len(batch)
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                st.session_state.processing_metrics['stages']['dense_vectors']['failed'] += len(batch)
                continue
            
            # Process each chunk in the batch
            for chunk, context, embedding in zip(batch, contexts, embeddings):
                try:
                    # Upsert to Qdrant
                    success = st.session_state.clients['qdrant'].upsert_chunk(
                        chunk_text=chunk['text'],
                        context_text=context,
                        dense_embedding=embedding,
                        metadata=metadata,
                        chunk_id=str(len(processed_chunks))
                    )
                    
                    if success:
                        processed_chunks.append({
                            'text': chunk['text'],
                            'context': context,
                            'embedding': embedding
                        })
                        st.session_state.processing_metrics['stages']['upserts']['success'] += 1
                        st.session_state.processing_metrics['processed_chunks'] += 1
                    
                except Exception as e:
                    logger.error(f"Error upserting chunk: {str(e)}")
                    st.session_state.processing_metrics['stages']['upserts']['failed'] += 1
                    
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Error in process_chunks_async: {str(e)}")
        raise

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE['anthropic'], period=60)
async def generate_chunk_context(chunk_text: str, doc_context_response: Any) -> str:
    """Generate context for a chunk using the cached document context."""
    try:
        response = await st.session_state.clients['anthropic'].messages.create(
            model=DEFAULT_LLM_MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"{st.session_state.context_prompt}\n\nDocument context:\n{doc_context_response}\n\nChunk text:\n{chunk_text}"
            }]
        )
        
        st.session_state.processing_metrics['stages']['context']['success'] += 1
        return response.content[0].text
        
    except Exception as e:
        logger.error(f"Error generating chunk context: {str(e)}")
        st.session_state.processing_metrics['stages']['context']['failed'] += 1
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
        # Download the document
        response = requests.get(url)
        response.raise_for_status()

        # Save content to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name

        # Initialize LlamaParse with optimized settings
        parser = LlamaParse(
            api_key=st.secrets.get("LLAMA_CLOUD_API_KEY"),
            result_type="text",      # Use text for better chunking compatibility
            num_workers=8,           # Increased workers for better performance
            verbose=False,           # Disable verbose output
            mode="FAST",            # Use FAST mode for quicker processing
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
