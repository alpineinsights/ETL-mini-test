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

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Local imports
from init_utils import initialize_qdrant
from qdrant_adapter import QdrantAdapter, VECTOR_DIMENSIONS

# Set up logging
logger = logging.getLogger(__name__)

# Must be the first Streamlit command
st.set_page_config(page_title="Document Processing Pipeline", layout="wide")

# Configuration defaults
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_MODEL = "voyage-finance-2"
DEFAULT_QDRANT_URL = "https://qdrant.your-domain.com"  # Update this with your actual Qdrant URL
DEFAULT_LLM_MODEL = "claude-3-haiku-20240307"
DEFAULT_CONTEXT_PROMPT = """Given the following text, please provide a concise summary that captures the key points and main ideas:

Text: {text}

Summary:"""

VECTOR_DIMENSIONS = {
    "voyage-finance-2": 1024,
    "voyage-large-2": 1536,
    "voyage-code-2": 1024
}

# Initialize Qdrant client in session state
if 'qdrant_client' not in st.session_state:
    try:
        qdrant_client = initialize_qdrant()
        if qdrant_client:
            st.session_state.qdrant_client = QdrantAdapter(
                url=st.secrets.get("QDRANT_URL", DEFAULT_QDRANT_URL),
                api_key=st.secrets["QDRANT_API_KEY"],
                collection_name="documents",
                embedding_model=DEFAULT_EMBEDDING_MODEL
            )
            logger.info("Successfully initialized Qdrant client")
        else:
            st.error("Failed to initialize Qdrant client")
    except Exception as e:
        st.error(f"Error initializing Qdrant client: {str(e)}")

# Initialize other clients and session state
if 'clients' not in st.session_state:
    try:
        st.session_state.clients = {
            'anthropic': anthropic.Client(
                api_key=st.secrets['ANTHROPIC_API_KEY']
            ),
            'llama_parser': LlamaParse(
                api_key=st.secrets['LLAMA_PARSE_API_KEY']
            ),
            'embed_model': VoyageEmbedding(
                model_name=DEFAULT_EMBEDDING_MODEL,
                voyage_api_key=st.secrets['VOYAGE_API_KEY']
            ),
            'qdrant': QdrantAdapter(
                url=st.secrets.get("QDRANT_URL", DEFAULT_QDRANT_URL),
                api_key=st.secrets["QDRANT_API_KEY"],
                collection_name="documents",
                embedding_model=DEFAULT_EMBEDDING_MODEL
            )
        }
        logger.info("Successfully initialized all clients")
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        st.stop()

# Initialize session state for metrics if not exists
if 'processing_metrics' not in st.session_state:
    st.session_state.processing_metrics = {
        'total_documents': 0,
        'processed_documents': 0,
        'total_chunks': 0,
        'successful_chunks': 0,
        'failed_chunks': 0,
        'total_tokens': 0,
        'stored_vectors': 0,
        'cache_hits': 0,
        'errors': [],
        'start_time': None
    }

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
        message = st.session_state.clients['anthropic'].messages.create(
            model=DEFAULT_LLM_MODEL,
            max_tokens=1024,
            temperature=0.0,
            system="You are an expert at analyzing and summarizing text...",
            messages=[{
                "role": "user",
                "content": f"{DEFAULT_CONTEXT_PROMPT}\n\nText to analyze:\n{chunk_text}"
            }]
        )
        return message.content
    except Exception as e:
        logger.error(f"Error generating context: {str(e)}")
        raise

def process_pdf(file_path: str, filename: str) -> Dict[str, Any]:
    """Process a PDF file and return extracted text and metadata"""
    try:
        result = st.session_state.clients['llama_parser'].load_data(file_path)
        
        if not result or not result.text:
            raise ValueError("No text extracted from PDF")
            
        metadata = {
            "filename": filename,
            "title": result.metadata.get("title", ""),
            "author": result.metadata.get("author", ""),
            "creation_date": result.metadata.get("creation_date", ""),
            "page_count": result.metadata.get("page_count", 0)
        }
        
        return {
            "text": result.text,
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Error processing PDF {filename}: {str(e)}")
        raise

def process_chunks(chunks: List[Dict], metadata: Dict) -> List[Dict]:
    """Process text chunks with context generation and embedding"""
    processed_chunks = []
    for chunk in chunks:
        try:
            # Generate context
            context = generate_context(chunk['text'])
            
            # Generate embedding
            embedding = st.session_state.clients['embed_model'].embed_query(context)
            
            # Store in Qdrant
            chunk_id = f"{metadata['url']}_{len(processed_chunks)}"
            st.session_state.clients['qdrant'].upsert_chunk(
                chunk_id=chunk_id,
                chunk_text=chunk['text'],
                context_text=context,
                dense_embedding=embedding,
                metadata={
                    **metadata,
                    'chunk_index': len(processed_chunks)
                }
            )
            
            processed_chunks.append({
                'id': chunk_id,
                'text': chunk['text'],
                'context': context,
                'tokens': chunk['tokens']
            })
            
            # Update metrics
            st.session_state.processing_metrics['successful_chunks'] += 1
            st.session_state.processing_metrics['total_tokens'] += chunk['tokens']
            st.session_state.processing_metrics['stored_vectors'] += 1
            
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            st.session_state.processing_metrics['failed_chunks'] += 1
            st.session_state.processing_metrics['errors'].append(str(e))
            
    return processed_chunks

def display_metrics():
    """Display current processing metrics"""
    metrics = st.session_state.processing_metrics
    
    if metrics['start_time']:
        elapsed_time = datetime.now() - metrics['start_time']
        st.write(f"Processing time: {elapsed_time}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Documents Processed", 
                 f"{metrics['processed_documents']}/{metrics['total_documents']}")
    with col2:
        st.metric("Successful Chunks", metrics['successful_chunks'])
    with col3:
        st.metric("Failed Chunks", metrics['failed_chunks'])
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric("Total Tokens", metrics['total_tokens'])
    with col5:
        st.metric("Stored Vectors", metrics['stored_vectors'])
    with col6:
        st.metric("Cache Hits", metrics['cache_hits'])
    
    if metrics['errors']:
        with st.expander("Processing Errors"):
            for error in metrics['errors']:
                st.error(error)

def parse_sitemap(url: str = "https://contextrag.s3.eu-central-2.amazonaws.com/sitemap.xml") -> List[str]:
    """Parse XML sitemap and return list of URLs."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse XML content
        root = ET.fromstring(response.content)
        
        # Extract URLs (handle both sitemap and urlset root elements)
        urls = []
        
        # Try sitemap namespace first
        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Look for URLs in both sitemap and url elements
        for loc in root.findall('.//ns:loc', namespaces):
            if loc.text:
                urls.append(unquote(loc.text))
        
        logger.info(f"Found {len(urls)} URLs in sitemap")
        return urls
        
    except Exception as e:
        logger.error(f"Error parsing sitemap: {str(e)}")
        raise

def process_url(url: str) -> Dict[str, Any]:
    """Process a URL and extract its content and metadata."""
    try:
        # Download the PDF file
        response = requests.get(url)
        response.raise_for_status()
        
        # Create a temporary file to store the PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        try:
            # Process the PDF using LlamaParse
            result = st.session_state.clients['llama_parser'].load_data(temp_path)
            
            if not result or not result.text:
                raise ValueError("No text extracted from PDF")
            
            # Extract filename from URL
            filename = url.split('/')[-1]
            
            metadata = {
                "url": url,
                "filename": filename,
                "source_type": "pdf",
                "title": result.metadata.get("title", ""),
                "author": result.metadata.get("author", ""),
                "creation_date": result.metadata.get("creation_date", ""),
                "page_count": result.metadata.get("page_count", 0)
            }
            
            return {
                "text": result.text,
                "metadata": metadata
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        raise

def save_processed_urls(urls: set) -> None:
    """Save processed URLs to persistent storage"""
    try:
        with open('processed_urls.json', 'w') as f:
            json.dump(list(urls), f)
    except Exception as e:
        logger.error(f"Error saving processed URLs: {str(e)}")

# Streamlit UI
st.title("Document Processing Pipeline")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    if st.button("Reset Metrics"):
        st.session_state.processing_metrics = {
            'total_documents': 0,
            'processed_documents': 0,
            'total_chunks': 0,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'total_tokens': 0,
            'stored_vectors': 0,
            'cache_hits': 0,
            'errors': [],
            'start_time': None
        }
        st.success("Metrics reset successfully")
    
    if st.button("Delete Collection"):
        try:
            st.session_state.clients['qdrant'].delete_collection()
            st.success("Collection deleted successfully")
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
    
    if st.button("Create Collection"):
        try:
            st.session_state.clients['qdrant'].create_collection()
            st.success("Collection created successfully")
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
        max_value=2000,
        value=DEFAULT_CHUNK_SIZE,
        help="Number of tokens per chunk"
    )
    
    chunk_overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=DEFAULT_CHUNK_OVERLAP,
        help="Number of overlapping tokens between chunks"
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
            # Parse sitemap
            urls = parse_sitemap(sitemap_url)
            st.session_state.processing_metrics['total_documents'] = len(urls)
            st.session_state.processing_metrics['start_time'] = datetime.now()
            
            progress_bar = st.progress(0)
            
            for i, url in enumerate(urls):
                try:
                    # Check if URL was already processed
                    if url in st.session_state.processed_urls:
                        st.session_state.processing_metrics['cache_hits'] += 1
                        continue
                    
                    # Process URL content
                    result = process_url(url)
                    
                    # Create chunks
                    chunks = create_semantic_chunks(result['text'])
                    st.session_state.processing_metrics['total_chunks'] += len(chunks)
                    
                    # Process chunks
                    processed_chunks = process_chunks(chunks, result['metadata'])
                    
                    # Update metrics
                    st.session_state.processing_metrics['processed_documents'] += 1
                    st.session_state.processed_urls.add(url)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(urls))
                    
                except Exception as e:
                    st.error(f"Error processing {url}: {str(e)}")
                    st.session_state.processing_metrics['errors'].append(str(e))
                    continue
            
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
            query_embedding = st.session_state.clients['embed_model'].embed_query(query)
            
            # Search
            results = st.session_state.clients['qdrant'].search(
                query_text=query,
                query_vector=query_embedding,
                limit=5
            )
            
            # Display results
            for i, result in enumerate(results, 1):
                with st.expander(f"Result {i} (Score: {result['score']:.3f})"):
                    st.write("**Original Text:**")
                    st.write(result['payload']['chunk_text'])
                    st.write("**Context:**")
                    st.write(result['payload']['context'])
                    st.write("**Metadata:**")
                    metadata = {k: v for k, v in result['payload'].items() 
                              if k not in ['chunk_text', 'context']}
                    st.json(metadata)
            
        except Exception as e:
            st.error(f"Error performing search: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Claude, Voyage AI, and Qdrant")
