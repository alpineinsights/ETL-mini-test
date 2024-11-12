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
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from ratelimit import limits, sleep_and_retry
import sentry_sdk
from prometheus_client import Counter, Histogram

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Local imports
from init_utils import initialize_qdrant
from qdrant_adapter import QdrantAdapter, VECTOR_DIMENSIONS

# Set up logging
logger = logging.getLogger(__name__)

# Must be the first Streamlit command
st.set_page_config(page_title="Alpine ETL Processing Pipeline", layout="wide")

# Configuration defaults
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_EMBEDDING_MODEL = "voyage-finance-2"
DEFAULT_QDRANT_URL = "https://3efb9175-b8b6-43f3-aef4-d2695ed84dc6.europe-west3-0.gcp.cloud.qdrant.io"  # Update this with your actual Qdrant URL
DEFAULT_LLM_MODEL = "claude-3-haiku-20240307"
DEFAULT_CONTEXT_PROMPT = """Give a short succinct context to situate this chunk within the overall enclosed document broader context for the purpose of improving similarity search retrieval of the chunk. 
Provide a very succint high level overview (i.e. not a summary) of the chunk's content in no more than 100 characters with a focus on keywords for better similarity search retrieval

Answer only with the succinct context, and nothing else (no introduction, no conclusion, no headings).

Example:
Chunk is part of a release of Saint Gobain Q3 2024 results emphasizing Saint Gobain's performance in construction chemicals in the US market, price and volumes effects, and operating leverage. 
"""

VECTOR_DIMENSIONS = {
    "voyage-finance-2": 1024,
    "voyage-3": 1024
}

# Configure rate limits
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

# Initialize Qdrant client in session state
if 'qdrant_client' not in st.session_state:
    try:
        # Initialize session state variables
        if 'clients' not in st.session_state:
            st.session_state.clients = {}

        # Initialize Anthropic client
        if 'anthropic' not in st.session_state.clients:
            try:
                st.session_state.clients['anthropic'] = anthropic.Client(
                    api_key=st.secrets["ANTHROPIC_API_KEY"]
                )
                logger.info("Successfully initialized Anthropic client")
            except Exception as e:
                st.error(f"Error initializing Anthropic client: {str(e)}")

        # Initialize Qdrant client
        if 'qdrant_client' not in st.session_state:
            try:
                qdrant_client = initialize_qdrant()
                if qdrant_client:
                    st.session_state.clients['qdrant'] = QdrantAdapter(
                        url=st.secrets.get("QDRANT_URL", DEFAULT_QDRANT_URL),
                        api_key=st.secrets["QDRANT_API_KEY"],
                        collection_name="documents",
                        embedding_model=DEFAULT_EMBEDDING_MODEL,
                        anthropic_client=st.session_state.clients.get('anthropic')
                    )
                    logger.info("Successfully initialized Qdrant client")
                else:
                    st.error("Failed to initialize Qdrant client")
            except Exception as e:
                st.error(f"Error initializing Qdrant client: {str(e)}")
        if qdrant_client:
            st.session_state.qdrant_client = QdrantAdapter(
                url=st.secrets.get("QDRANT_URL", DEFAULT_QDRANT_URL),
                api_key=st.secrets["QDRANT_API_KEY"],
                collection_name="documents",
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                anthropic_client=st.session_state.clients['anthropic']
            )
            logger.info("Successfully initialized Qdrant client")
        else:
            st.error("Failed to initialize Qdrant client")
    except Exception as e:
        st.error(f"Error initializing Qdrant client: {str(e)}")

# Initialize metrics structure
metrics_template = {
    'total_documents': 0,
    'processed_documents': 0,
    'total_chunks': 0,
    'successful_chunks': 0,
    'failed_chunks': 0,
    'total_tokens': 0,
    'stored_vectors': 0,
    'cache_hits': 0,
    'errors': [],
    'start_time': None,
    'stages': {
        'parsing': {'success': 0, 'failed': 0},
        'chunking': {'success': 0, 'failed': 0},
        'context': {'success': 0, 'failed': 0},
        'metadata': {'success': 0, 'failed': 0},
        'dense_vectors': {'success': 0, 'failed': 0},
        'sparse_vectors': {'success': 0, 'failed': 0},
        'upserts': {'success': 0, 'failed': 0}
    }
}

# Use this template in both places:
if 'processing_metrics' not in st.session_state:
    st.session_state.processing_metrics = metrics_template.copy()

# Initialize session state for clients if not exists
if 'clients' not in st.session_state:
    st.session_state.clients = {}

# Initialize collection name if not exists
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = "documents"

try:
    # Initialize Anthropic client first
    if 'anthropic' not in st.session_state.clients:
        st.session_state.clients['anthropic'] = anthropic.Client(
            api_key=st.secrets["ANTHROPIC_API_KEY"]
        )
        logger.info("Successfully initialized Anthropic client")

    # Initialize Voyage embedding model
    if 'embed_model' not in st.session_state.clients:
        st.session_state.clients['embed_model'] = VoyageEmbedding(
            api_key=st.secrets["VOYAGE_API_KEY"],
            model_name=DEFAULT_EMBEDDING_MODEL
        )
        logger.info("Successfully initialized Voyage embedding model")

    # Initialize Llama parser
    if 'llama_parser' not in st.session_state.clients:
        st.session_state.clients['llama_parser'] = LlamaParse(
            api_key=st.secrets["LLAMA_PARSE_API_KEY"]
        )
        logger.info("Successfully initialized Llama parser")

    # Initialize Qdrant client last (since it depends on other clients)
    if 'qdrant' not in st.session_state.clients:
        qdrant_client = initialize_qdrant()
        if qdrant_client:
            st.session_state.clients['qdrant'] = QdrantAdapter(
                url=st.secrets.get("QDRANT_URL", DEFAULT_QDRANT_URL),
                api_key=st.secrets["QDRANT_API_KEY"],
                collection_name=st.session_state.collection_name,
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                anthropic_client=st.session_state.clients['anthropic']  # Pass the initialized Anthropic client
            )
            logger.info("Successfully initialized Qdrant client")
        else:
            raise Exception("Failed to initialize Qdrant client")

    logger.info("Successfully initialized all clients")
except Exception as e:
    st.error(f"Error initializing clients: {str(e)}")
    st.stop()

if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = set()

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string"""
    try:
        token_count = st.session_state.clients['anthropic'].count_tokens(text)
        return token_count
    except Exception as e:
        st.error(f"Error counting tokens: {str(e)}")
        return 0

def create_semantic_chunks(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_SIZE,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """Create semantically meaningful chunks from text"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    previous_paragraphs = []
    
    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)
        
        if paragraph_tokens > max_tokens:
            sentences = [s.strip() + '.' for s in paragraph.split('. ') if s.strip()]
            for sentence in sentences:
                sentence_tokens = count_tokens(sentence)
                
                if current_tokens + sentence_tokens > max_tokens:
                    if current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        chunk_tokens = count_tokens(chunk_text)
                        chunks.append({
                            'text': chunk_text,
                            'tokens': chunk_tokens
                        })
                        
                        overlap_text = []
                        overlap_token_count = 0
                        for prev in reversed(previous_paragraphs):
                            prev_tokens = count_tokens(prev)
                            if overlap_token_count + prev_tokens <= overlap_tokens:
                                overlap_text.insert(0, prev)
                                overlap_token_count += prev_tokens
                            else:
                                break
                        
                        current_chunk = overlap_text
                        current_tokens = overlap_token_count
                    
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
                
        elif current_tokens + paragraph_tokens > max_tokens:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_tokens = count_tokens(chunk_text)
            chunks.append({
                'text': chunk_text,
                'tokens': chunk_tokens
            })
            
            overlap_text = []
            overlap_token_count = 0
            for prev in reversed(previous_paragraphs):
                prev_tokens = count_tokens(prev)
                if overlap_token_count + prev_tokens <= overlap_tokens:
                    overlap_text.insert(0, prev)
                    overlap_token_count += prev_tokens
                else:
                    break
            
            current_chunk = overlap_text
            current_tokens = overlap_token_count
            current_chunk.append(paragraph)
            current_tokens += paragraph_tokens
            
        else:
            current_chunk.append(paragraph)
            current_tokens += paragraph_tokens
        
        previous_paragraphs.append(paragraph)
    
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        chunk_tokens = count_tokens(chunk_text)
        chunks.append({
            'text': chunk_text,
            'tokens': chunk_tokens
        })
    
    return chunks

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception(lambda e: not isinstance(e, KeyboardInterrupt))
)
def generate_context(chunk_text: str) -> str:
    """Generate contextual description for a chunk of text using Claude"""
    try:
        # Create message using the correct API format
        response = st.session_state.clients['anthropic'].messages.create(
            model=DEFAULT_LLM_MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"{DEFAULT_CONTEXT_PROMPT}\n\nText to process:\n{chunk_text}"
            }]
        )
        return response.content[0].text  # Access the response content correctly
    except Exception as e:
        logger.error(f"Error generating context: {str(e)}")
        st.session_state.processing_metrics['stages']['context']['failed'] += 1
        raise

def extract_document_metadata(text: str) -> Dict[str, str]:
    """Extract key metadata fields from document using Claude."""
    try:
        response = st.session_state.clients['anthropic'].messages.create(
            model=DEFAULT_LLM_MODEL,
            max_tokens=300,
            temperature=0,
            system="You are a precise JSON generator. You only output valid JSON objects, nothing else.",
            messages=[{
                "role": "user", 
                "content": f"""Extract exactly these three fields from the text and return them in a JSON object:
1. company: The main company name
2. date: The document date in YYYY.MM.DD format
3. fiscal_period: The fiscal period mentioned (both abbreviated and verbose form)

Return ONLY a JSON object like this example, with no other text:
{{"company": "Adidas AG", "date": "2024.04.30", "fiscal_period": "Q1 2024, first quarter 2024"}}

Text to analyze:
{text[:2000]}  # Limit text length to avoid token issues
"""
            }]
        )

        # Get response and clean it
        json_str = response.content[0].text.strip()
        
        # Remove any markdown code block markers if present
        json_str = json_str.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON
        try:
            metadata = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from Claude: {json_str}")
            raise ValueError(f"Failed to parse JSON from Claude response: {e}")

        # Validate required fields
        required_fields = ["company", "date", "fiscal_period"]
        missing_fields = [field for field in required_fields if field not in metadata]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        raise

def process_pdf(file_path: str, filename: str) -> Dict[str, Any]:
    """Process a PDF file and return extracted text and metadata"""
    try:
        # Load documents - returns a list
        documents = st.session_state.clients['llama_parser'].load_data(file_path)
        
        if not documents:
            raise ValueError("No documents extracted from PDF")
            
        # Combine text from all documents
        combined_text = "\n\n".join(doc.text for doc in documents)
        
        # Create chunks from combined text
        chunks = create_semantic_chunks(combined_text)
        
        # Get metadata from first document
        metadata = {
            "filename": filename,
            "title": documents[0].metadata.get("title", ""),
            "author": documents[0].metadata.get("author", ""),
            "creation_date": documents[0].metadata.get("creation_date", ""),
            "page_count": len(documents)
        }
        
        return {
            "text": combined_text,
            "chunks": chunks,  # Add chunks to return dictionary
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Error processing PDF {filename}: {str(e)}")
        raise

async def process_chunks_async(chunks: List[Dict[str, Any]], metadata: Dict[str, Any], full_document: str) -> List[Dict[str, Any]]:
    """Process chunks concurrently while maintaining document-level context"""
    processed_chunks = []
    
    # Create tasks for all chunks
    async def process_single_chunk(chunk):
        try:
            # Generate context
            context = await asyncio.to_thread(
                st.session_state.clients['qdrant'].situate_context,
                doc=full_document,
                chunk=chunk['text']
            )
            context = context[0]  # Get just the context
            st.session_state.processing_metrics['stages']['context']['success'] += 1

            # Generate embedding
            dense_vector = await asyncio.to_thread(
                st.session_state.clients['embed_model'].get_text_embedding,
                chunk['text']
            )
            st.session_state.processing_metrics['stages']['dense_vectors']['success'] += 1

            # Store in Qdrant
            chunk_id = f"{metadata.get('file_name', 'unknown')}_{chunks.index(chunk)}"
            success = await asyncio.to_thread(
                st.session_state.clients['qdrant'].upsert_chunk,
                chunk_text=chunk['text'],
                context_text=context,
                dense_embedding=dense_vector,
                metadata=metadata,
                chunk_id=chunk_id
            )
            
            if success:
                st.session_state.processing_metrics['stages']['upserts']['success'] += 1
                st.session_state.processing_metrics['stored_vectors'] += 1
                
                return {
                    'chunk_text': chunk['text'],
                    'context': context,
                    'dense_vector': dense_vector,
                    **metadata
                }
                
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            st.session_state.processing_metrics['failed_chunks'] += 1
            st.session_state.processing_metrics['errors'].append(str(e))
            return None

    # Process chunks concurrently with a limit
    chunk_tasks = [process_single_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
    
    # Filter out None results (failed chunks)
    processed_chunks = [r for r in results if r is not None]
    st.session_state.processing_metrics['successful_chunks'] += len(processed_chunks)
    
    return processed_chunks

async def process_urls_async(urls: List[str]):
    try:
        tasks = []
        chunk_size = 5
        for i in range(0, len(urls), chunk_size):
            url_batch = urls[i:i + chunk_size]
            batch_tasks = [process_single_url(url, i + j, len(urls)) 
                         for j, url in enumerate(url_batch)]
            
            # Handle exceptions for each batch
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {str(result)}")
                    API_ERRORS.labels(api='processing').inc()
                    continue
                tasks.extend([r for r in result if r is not None])
            
            # Update progress
            progress = min((i + chunk_size) / len(urls), 1.0)
            progress_bar.progress(progress)
            
    except Exception as e:
        logger.error(f"Error in async processing: {str(e)}")
        st.error(f"Processing error: {str(e)}")
        raise

def display_metrics():
    """Display current processing metrics"""
    metrics = st.session_state.processing_metrics
    
    if metrics['start_time']:
        elapsed_time = datetime.now() - metrics['start_time']
        st.metric("Elapsed Time", f"{elapsed_time.seconds} seconds")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents Processed", f"{metrics['processed_docs']}/{metrics['total_docs']}")
    with col2:
        st.metric("Chunks Processed", f"{metrics['processed_chunks']}/{metrics['total_chunks']}")
    
    if metrics['errors'] > 0:
        st.error(f"Errors encountered: {metrics['errors']}")

def parse_sitemap(url: str) -> List[str]:
    """Parse XML sitemap and return list of URLs."""
    try:
        logger.info(f"Fetching sitemap from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Log response details
        logger.info(f"Sitemap response status: {response.status_code}")
        logger.info(f"Sitemap content length: {len(response.content)} bytes")
        
        # Parse XML content
        try:
            root = ET.fromstring(response.content)
            logger.info(f"Successfully parsed XML content")
            
            # Extract URLs (try both with and without namespace)
            urls = []
            namespaces = [
                './/{http://www.sitemaps.org/schemas/sitemap/0.9}loc',
                './/loc'  # Try without namespace
            ]
            
            for namespace in namespaces:
                urls.extend([url.text for url in root.findall(namespace)])
            
            if not urls:
                logger.warning("No URLs found in sitemap")
                st.warning("No URLs found in sitemap. Please check the URL and XML format.")
            else:
                logger.info(f"Found {len(urls)} URLs in sitemap")
                st.info(f"Found {len(urls)} URLs in sitemap")
            
            return urls
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
            st.error(f"Failed to parse XML content: {str(e)}")
            # Log the first 500 characters of the response for debugging
            logger.debug(f"Response content preview: {response.text[:500]}")
            raise
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        st.error(f"Failed to fetch sitemap: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in parse_sitemap: {str(e)}")
        st.error(f"Unexpected error while processing sitemap: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception(lambda e: not isinstance(e, (ValueError, KeyboardInterrupt)))
)
def process_url(url: str) -> Dict[str, Any]:
    temp_file_path = None
    try:
        logger.info(f"Starting to process URL: {url}")
        
        # Download PDF content
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(response.content)
            
            # Process PDF and get chunks
            result = process_pdf(temp_file_path, Path(url).name)
            
            # Extract metadata from full document text
            doc_metadata = extract_document_metadata(result['text'])
            
            # Update result metadata
            result['metadata'].update(doc_metadata)
            result['metadata']['url'] = url
            
            return result
            
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        raise
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")

def save_processed_urls(urls: set) -> None:
    """Save processed URLs to persistent storage"""
    try:
        with open('processed_urls.json', 'w') as f:
            json.dump(list(urls), f)
    except Exception as e:
        logger.error(f"Error saving processed URLs: {str(e)}")

# After imports but before any other code
def initialize_clients():
    """Initialize all required clients"""
    try:
        if 'clients' not in st.session_state:
            st.session_state.clients = {}
        
        # Initialize Anthropic client first
        st.session_state.clients['anthropic'] = anthropic.Client(
            api_key=st.secrets["ANTHROPIC_API_KEY"]
        )
        logger.info("Successfully initialized Anthropic client")
        
        # Initialize Voyage embedding model
        st.session_state.clients['embed_model'] = VoyageEmbedding(
            api_key=st.secrets["VOYAGE_API_KEY"],
            model_name=DEFAULT_EMBEDDING_MODEL
        )
        logger.info("Successfully initialized Voyage embedding model")
        
        # Initialize Qdrant last (depends on other clients)
        qdrant_client = initialize_qdrant()
        if not qdrant_client:
            raise Exception("Failed to initialize Qdrant client")
            
        st.session_state.clients['qdrant'] = QdrantAdapter(
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"],
            collection_name="documents",
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            anthropic_client=st.session_state.clients['anthropic']
        )
        logger.info("Successfully initialized Qdrant client")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing clients: {str(e)}")
        return False

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

# Near the top of the file, after imports
def initialize_session_state():
    """Initialize all session state variables"""
    if 'processing_metrics' not in st.session_state:
        st.session_state.processing_metrics = {
            'start_time': None,
            'total_docs': 0,
            'processed_docs': 0,
            'total_chunks': 0,
            'processed_chunks': 0,
            'errors': 0
        }
    if 'processed_urls' not in st.session_state:
        st.session_state.processed_urls = set()
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = "documents"

# In the sitemap processing code
if st.button("Process Sitemap"):
    try:
        with st.spinner("Parsing sitemap..."):
            urls = parse_sitemap(sitemap_url)
            if not urls:
                st.warning("No URLs found to process")
                st.stop()
            
            st.session_state.processing_metrics['total_docs'] = len(urls)  # Changed from total_documents
            st.session_state.processing_metrics['start_time'] = datetime.now()
            
            # Rest of the processing code...

# After validate_environment() call
if st.runtime.exists():
    cleanup_session_state()
    initialize_session_state()

validate_environment()
if not initialize_clients():
    st.stop()

# Then start UI code
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
        value=DEFAULT_CHUNK_SIZE,
        help="Number of tokens per chunk (default: 500)"
    )
    
    chunk_overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=200,
        value=DEFAULT_CHUNK_OVERLAP,
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
            # Generate query embedding
            query_embedding = st.session_state.clients['embed_model'].get_query_embedding(query)
            
            # Search
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
