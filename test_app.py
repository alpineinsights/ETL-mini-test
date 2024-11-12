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
st.set_page_config(page_title="Alpine ETL Processing Pipeline", layout="wide")

# Configuration defaults
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_MODEL = "voyage-finance-2"
DEFAULT_QDRANT_URL = "https://3efb9175-b8b6-43f3-aef4-d2695ed84dc6.europe-west3-0.gcp.cloud.qdrant.io"  # Update this with your actual Qdrant URL
DEFAULT_LLM_MODEL = "claude-3-haiku-20240307"
DEFAULT_CONTEXT_PROMPT = """Give a short succinct context to situate this chunk within the overall enclosed document broader context for the purpose of improving similarity search retrieval of the chunk. 

Make sure to list:
1. The name of the main company mentioned AND any other secondary companies mentioned if applicable. ONLY use company names exact spellings from the list below to facilitate similarity search retrieval.
2. The apparent date of the document (YYYY.MM.DD)
3. Any fiscal period mentioned. ALWAYS use BOTH abreviated tags (e.g. Q1 2024, Q2 2024, H1 2024) AND more verbose tags (e.g. first quarter 2024, second quarter 2024, first semester 2024) to improve retrieval.
4. A very succint high level overview (i.e. not a summary) of the chunk's content in no more than 100 characters with a focus on keywords for better similarity search retrieval

Answer only with the succinct context, and nothing else (no introduction, no conclusion, no headings).

Example:
Main company: Saint Gobain
Secondary companies: none
date : 2024.11.21
Q3 2024, third quarter of 2024
Chunk is part of a release of Saint Gobain Q3 2024 results emphasizing Saint Gobain's performance in construction chemicals in the US market, price and volumes effects, and operatng leverage. 
"""

VECTOR_DIMENSIONS = {
    "voyage-finance-2": 1024,
    "voyage-3": 1024
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

# Initialize session state for clients if not exists
if 'clients' not in st.session_state:
    st.session_state.clients = {}

# Initialize clients
try:
    # Initialize Anthropic client
    st.session_state.clients['anthropic'] = anthropic.Client(
        api_key=st.secrets["ANTHROPIC_API_KEY"]
    )
    
    # Initialize Voyage client
    st.session_state.clients['embed_model'] = VoyageEmbedding(
        model_name=DEFAULT_EMBEDDING_MODEL,
        api_key=st.secrets["VOYAGE_API_KEY"],
        embed_batch_size=10  # Optional: adjust based on your needs
    )
    
    # Initialize LlamaParse client - Fix the API key name
    st.session_state.clients['llama_parser'] = LlamaParse(
        api_key=st.secrets["LLAMA_PARSE_API_KEY"]  # Changed from LLAMA_CLOUD_API_KEY
    )
    
    # Initialize Qdrant client
    qdrant_client = initialize_qdrant()
    if qdrant_client:
        st.session_state.clients['qdrant'] = QdrantAdapter(
            url=st.secrets.get("QDRANT_URL", DEFAULT_QDRANT_URL),
            api_key=st.secrets["QDRANT_API_KEY"],
            collection_name="documents",
            embedding_model=DEFAULT_EMBEDDING_MODEL
        )
    
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

def process_chunks(chunks: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process chunks to add context and embeddings."""
    processed_chunks = []
    
    for chunk in chunks:
        try:
            # Generate context using the detailed prompt
            context = st.session_state.clients['anthropic'].messages.create(
                model=DEFAULT_LLM_MODEL,
                max_tokens=300,
                messages=[{
                    "role": "user", 
                    "content": f"""Give a short succinct context to situate this chunk within the overall enclosed document broader context for the purpose of improving similarity search retrieval of the chunk. 

Make sure to list:
1. The name of the main company mentioned AND any other secondary companies mentioned if applicable. ONLY use company names exact spellings from the list below to facilitate similarity search retrieval.
2. The apparent date of the document (YYYY.MM.DD)
3. Any fiscal period mentioned. ALWAYS use BOTH abreviated tags (e.g. Q1 2024, Q2 2024, H1 2024) AND more verbose tags (e.g. first quarter 2024, second quarter 2024, first semester 2024) to improve retrieval.
4. A very succint high level overview (i.e. not a summary) of the chunk's content in no more than 100 characters with a focus on keywords for better similarity search retrieval

Answer only with the succinct context, and nothing else (no introduction, no conclusion, no headings).

Text to process:
{chunk['text']}"""
                }]
            ).content[0].text
            
            # Generate dense embedding for chunk content
            dense_vector = st.session_state.clients['embed_model'].embed_text(chunk['text'])
            
            processed_chunks.append({
                'chunk_text': chunk['text'],
                'context': context,
                'dense_vector': dense_vector,
                **metadata
            })
            
            # Update metrics
            st.session_state.processing_metrics['successful_chunks'] += 1
            
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            st.session_state.processing_metrics['failed_chunks'] += 1
            st.session_state.processing_metrics['errors'].append(str(e))
            continue
    
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
            with st.spinner("Parsing sitemap..."):
                # Parse sitemap
                urls = parse_sitemap(sitemap_url)
                
                if not urls:
                    st.warning("No URLs found to process")
                    st.stop()
                    
                st.session_state.processing_metrics['total_documents'] = len(urls)
                st.session_state.processing_metrics['start_time'] = datetime.now()
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, url in enumerate(urls):
                    try:
                        status_text.text(f"Processing URL {i+1}/{len(urls)}: {url}")
                        
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
            query_embedding = st.session_state.clients['embed_model'].embed_text(query)
            
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
                    # Display only specified metadata fields
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
