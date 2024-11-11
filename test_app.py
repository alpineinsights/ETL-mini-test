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

# Set up logging
logger = logging.getLogger(__name__)

class QdrantAdapter:
    """Handles interaction with Qdrant vector database for hybrid search."""
    
    def __init__(self, url: str, api_key: str, collection_name: str = "documents"):
        """Initialize Qdrant client."""
        try:
            self.client = QdrantClient(url=url, api_key=api_key)
            self.collection_name = collection_name
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                strip_accents='unicode',
                ngram_range=(1, 2),
                max_features=768,
                sublinear_tf=True
            )
            logger.info("Successfully initialized QdrantAdapter")
        except Exception as e:
            logger.error(f"Failed to initialize QdrantAdapter: {str(e)}")
            raise
        
    def create_collection(self, dense_dim: int = 1024, sparse_dim: int = 768) -> bool:
        """Create or recreate collection with specified vector dimensions."""
        try:
            # Define vector configurations
            dense_config = models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
                on_disk=True
            )
            sparse_config = models.VectorParams(
                size=sparse_dim,
                distance=models.Distance.COSINE,
                on_disk=True
            )
            
            # Create collection with both vector types
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": dense_config,
                    "sparse": sparse_config
                }
            )
            
            # Create payload indices for efficient filtering
            indices = [
                ("timestamp", "text"),  # Changed from datetime to text
                ("filename", "keyword"),
                ("chunk_index", "integer"),
                ("url", "keyword")  # Added URL index
            ]
            
            for field_name, field_type in indices:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
            
            logger.info(f"Created collection {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

    def compute_sparse_embedding(self, text: str) -> Dict[str, List]:
        """Compute sparse embedding using TF-IDF vectorization."""
        try:
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty")
                
            sparse_vector = self.vectorizer.fit_transform([text]).toarray()[0]
            non_zero_indices = np.nonzero(sparse_vector)[0]
            non_zero_values = sparse_vector[non_zero_indices]
            
            if len(non_zero_indices) == 0:
                raise ValueError("No features were extracted from the text")
                
            return {
                "indices": non_zero_indices.tolist(),
                "values": non_zero_values.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error computing sparse embedding: {str(e)}")
            raise

    def upsert_chunk(self,
                     chunk_text: str,
                     context_text: str,
                     dense_embedding: List[float],
                     metadata: Dict[str, Any],
                     chunk_id: str) -> bool:
        """Upsert a document chunk with both dense and sparse embeddings."""
        try:
            if not chunk_text or not context_text:
                raise ValueError("Chunk text and context text cannot be empty")
            if not dense_embedding or len(dense_embedding) == 0:
                raise ValueError("Dense embedding cannot be empty")
                
            sparse_embedding = self.compute_sparse_embedding(context_text)
            current_time = datetime.now().isoformat()
            
            point = models.PointStruct(
                id=chunk_id,
                payload={
                    "chunk_text": chunk_text,
                    "context": context_text,
                    "timestamp": current_time,
                    "vector_type": "hybrid",
                    **metadata
                },
                vectors={
                    "dense": dense_embedding,
                    "sparse": sparse_embedding
                }
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True
            )
            
            logger.info(f"Successfully upserted chunk {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting chunk {chunk_id}: {str(e)}")
            raise

    def search(self,
              query_text: str,
              query_vector: List[float],
              limit: int = 5,
              use_sparse: bool = True,
              score_threshold: float = 0.7,
              filter_conditions: Optional[Dict] = None) -> List[Dict]:
        """Perform hybrid search using dense and optionally sparse vectors."""
        try:
            if not query_text or not query_vector:
                raise ValueError("Query text and vector cannot be empty")
                
            sparse_vector = None
            if use_sparse:
                sparse_embedding = self.compute_sparse_embedding(query_text)
                sparse_vector = models.SparseVector(
                    indices=sparse_embedding["indices"],
                    values=sparse_embedding["values"]
                )
            
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": ("dense", query_vector),
                "limit": limit,
                "score_threshold": score_threshold,
                "with_payload": True,
                "with_vectors": False
            }
            
            if sparse_vector:
                search_params["query_vector_2"] = ("sparse", sparse_vector)
                
            if filter_conditions:
                search_params["query_filter"] = models.Filter(**filter_conditions)
            
            results = self.client.search(**search_params)
            
            formatted_results = []
            for hit in results:
                formatted_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                    "vector_distance": getattr(hit, "vector_distance", None)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status,
                "payload_schema": info.payload_schema
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise
    
    def delete_collection(self) -> bool:
        """Delete the current collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

# Constants and initial setup
STATE_FILE = "./.processed_urls.json"
TEMP_DIR = Path("./.temp")
TEMP_DIR.mkdir(exist_ok=True)

def load_processed_urls():
    """Load previously processed URLs from state file"""
    try:
        with open(STATE_FILE) as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()

def save_processed_urls(urls):
    """Save processed URLs to state file"""
    Path(STATE_FILE).parent.mkdir(exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(list(urls), f)

# Initialize session state
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = load_processed_urls()

if 'processing_metrics' not in st.session_state:
    st.session_state.processing_metrics = {
        'total_documents': 0,
        'processed_documents': 0,
        'total_chunks': 0,
        'successful_chunks': 0,
        'failed_chunks': 0,
        'cache_hits': 0,
        'total_tokens': 0,
        'stored_vectors': 0,
        'start_time': None,
        'errors': []
    }

# Initialize clients
try:
    client = anthropic.Client(
        api_key=st.secrets['ANTHROPIC_API_KEY'],
        default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
    )
    llama_parser = LlamaParse(
        api_key=st.secrets['LLAMA_PARSE_API_KEY'],
        result_type="text"
    )
    embed_model = VoyageEmbedding(
        model_name="voyage-finance-2",
        voyage_api_key=st.secrets['VOYAGE_API_KEY']
    )
    qdrant_client = QdrantAdapter(
        url="https://3efb9175-b8b6-43f3-aef4-d2695ed84dc6.europe-west3-0.gcp.cloud.qdrant.io",
        api_key=st.secrets['QDRANT_API_KEY'],
        collection_name="documents"
    )
    
    # Store clients in session state
    if 'clients' not in st.session_state:
        st.session_state.clients = {
            'anthropic': client,
            'llama_parser': llama_parser,
            'embed_model': embed_model,
            'qdrant': qdrant_client
        }
except Exception as e:
    st.error(f"Error initializing clients: {str(e)}")
    st.stop()

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
    max_tokens: int = 1000,
    overlap_tokens: int = 200
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

DEFAULT_PROMPT_TEMPLATE = """Give a short succinct context to situate this chunk within the overall enclosed document boader context for the purpose of improving similarity search retrieval of the chunk. 

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
Chunk is part of a releease of Saint Gobain Q3 2024 results emphasizing Saint Gobain's performance in construction chemicals in the US market, price and volumes effects, and operatng leverage."""

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception(lambda e: not isinstance(e, KeyboardInterrupt))
)
def generate_context(chunk_text: str) -> str:
    """Generate contextual description for a chunk of text using Claude"""
    try:
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            temperature=0.0,
            system="You are an expert at analyzing and summarizing text. Generate a detailed contextual description that captures the key information, relationships, and concepts from the provided text. Focus on creating a rich semantic representation that would be useful for retrieval.",
            messages=[{
                "role": "user",
                "content": f"Generate a detailed contextual description for the following text:\n\n{chunk_text}"
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
    """Process text chunks to generate context and embeddings"""
    processed_chunks = []
    
    for i, chunk in enumerate(chunks):
        try:
            # Generate context
            context = generate_context(chunk['text'])
            
            # Generate embedding
            embedding = st.session_state.clients['embed_model'].embed_query(context)
            
            # Create chunk metadata
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "chunk_tokens": chunk['tokens']
            }
            
            # Store in Qdrant
            chunk_id = f"{metadata['filename']}_{i}"
            st.session_state.clients['qdrant'].upsert_chunk(
                chunk_text=chunk['text'],
                context_text=context,
                dense_embedding=embedding,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            )
            
            processed_chunks.append({
                "id": chunk_id,
                "text": chunk['text'],
                "context": context,
                "metadata": chunk_metadata
            })
            
            # Update metrics
            st.session_state.processing_metrics['successful_chunks'] += 1
            st.session_state.processing_metrics['total_tokens'] += chunk['tokens']
            st.session_state.processing_metrics['stored_vectors'] += 1
            
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {str(e)}")
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

def reset_metrics():
    """Reset processing metrics to initial state"""
    st.session_state.processing_metrics = {
        'total_documents': 0,
        'processed_documents': 0,
        'total_chunks': 0,
        'successful_chunks': 0,
        'failed_chunks': 0,
        'cache_hits': 0,
        'total_tokens': 0,
        'stored_vectors': 0,
        'start_time': None,
        'errors': []
    }

def parse_sitemap(url: str) -> List[str]:
    """Parse XML sitemap and return list of URLs."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
        # Handle different XML namespaces
        namespaces = {
            'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'
        }
        
        urls = []
        for url_elem in root.findall('.//ns:url', namespaces):
            loc = url_elem.find('ns:loc', namespaces)
            if loc is not None and loc.text:
                urls.append(unquote(loc.text))
        
        return urls
        
    except Exception as e:
        logger.error(f"Error parsing sitemap: {str(e)}")
        raise

def fetch_url_content(url: str) -> Dict[str, Any]:
    """Fetch and process content from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Here you would add your HTML parsing logic
        # For now, we'll just use the raw text
        content = response.text
        
        metadata = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "content_type": response.headers.get('content-type', ''),
            "length": len(content)
        }
        
        return {
            "text": content,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        raise

# Streamlit UI
st.title("Document Processing Pipeline")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    if st.button("Reset Metrics"):
        reset_metrics()
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
        info = st.session_state.clients['qdrant'].get_collection_info()
        st.write(info)
    except Exception as e:
        st.error(f"Error getting collection info: {str(e)}")

# Main content
tab1, tab2 = st.tabs(["Process Documents", "Search"])

with tab1:
    st.header("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.session_state.processing_metrics['total_documents'] = len(uploaded_files)
        st.session_state.processing_metrics['start_time'] = datetime.now()
        
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                temp_path = TEMP_DIR / uploaded_file.name
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # Process PDF
                result = process_pdf(str(temp_path), uploaded_file.name)
                
                # Create chunks
                chunks = create_semantic_chunks(result['text'])
                st.session_state.processing_metrics['total_chunks'] += len(chunks)
                
                # Process chunks
                processed_chunks = process_chunks(chunks, result['metadata'])
                
                # Update metrics
                st.session_state.processing_metrics['processed_documents'] += 1
                
                # Clean up
                temp_path.unlink()
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                st.session_state.processing_metrics['errors'].append(str(e))
                continue
        
        st.success("Processing complete!")
    
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
