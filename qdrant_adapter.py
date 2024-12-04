"""
QdrantAdapter class for managing vector storage operations with hybrid search capabilities.
Handles both dense embeddings from Voyage and sparse embeddings from TF-IDF.
"""

from qdrant_client import QdrantClient, models
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import uuid
import streamlit as st
from ratelimit import limits, sleep_and_retry
from pathlib import Path
import time
from llama_index.core.node_parser import SentenceSplitter
from urllib.parse import unquote
import json

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

VECTOR_DIMENSIONS = {
    "voyage-finance-2": 1024,
    "voyage-large-3": 4096,
    "sparse": 768  # TF-IDF sparse vector dimension
}

DOCUMENT_CONTEXT_PROMPT = """Please analyze this document content and extract key metadata in JSON format with the following fields:
- company: Company name
- fiscal_period: Fiscal period (e.g. Q1 2024)
- date: Document date
- doc_type: Document type (report/transcript/presentation)

Document content:
{doc_content}
"""

CHUNK_CONTEXT_PROMPT = """
Here is a specific chunk from the document:
<chunk>
{chunk_content}
</chunk>

Using the document-level information identified above, create a context giving a very succinct high-level overview (max 100 characters) of THIS SPECIFIC CHUNK's content focusing on keywords for better retrieval

A very succint high level overview (i.e. not a summary) of the chunk's content in no more than 100 characters with a focus on keywords for better similarity search retrieval

Answer only with the succinct context, and nothing else (no introduction, no conclusion, no headings).

Example:
Chunk is part of a release of Saint Gobain Q3 2024 results emphasizing Saint Gobain's performance in construction chemicals in the US market, price and volumes effects, and operating leverage.
"""

# Configure rate limits
CALLS_PER_MINUTE = {
    'anthropic': 50,
    'voyage': 100,
    'qdrant': 100
}

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE['anthropic'], period=60)
def rate_limited_context(func, *args, **kwargs):
    """Rate-limited wrapper for Anthropic context generation."""
    return func(*args, **kwargs)

def is_overloaded_error(exception):
    """Check if the exception is an overloaded error."""
    return isinstance(exception, Exception) and 'overloaded_error' in str(exception)

class QdrantAdapter:
    """Handles interaction with Qdrant vector database for hybrid search."""
    
    # Add the prompt as a class constant
    DOCUMENT_CONTEXT_PROMPT = """Please analyze this document content and extract key metadata in JSON format with the following fields:
    - company: Company name
    - fiscal_period: Fiscal period (e.g. Q1 2024)
    - date: Document date
    - doc_type: Document type (report/transcript/presentation)
    
    Document content:
    {doc_content}
    """

    def __init__(self, url: str, api_key: str, collection_name: str = "documents", embedding_model: str = "voyage-finance-2", anthropic_client = None):
        """
        Initialize Qdrant client with logging.

        Args:
            url: Qdrant server URL
            api_key: API key for authentication
            collection_name: Name of the collection to use
            embedding_model: Name of the embedding model to use
            anthropic_client: Initialized Anthropic client for context generation
        """
        logger.info(f"Initializing QdrantAdapter with model: {embedding_model}")
        try:
            start_time = time.time()
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=60,
                prefer_grpc=False  # Force HTTP protocol if gRPC isn't supported
            )
            logger.info(f"Connected to Qdrant at {url} in {time.time() - start_time:.2f}s")
            
            self.collection_name = collection_name
            self.embedding_model = embedding_model
            self.dense_dim = VECTOR_DIMENSIONS[embedding_model]
            self.sparse_dim = VECTOR_DIMENSIONS['sparse']
            self.anthropic_client = anthropic_client  # Store the Anthropic client

            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=self.sparse_dim,
                stop_words='english'
            )

            # Fit vectorizer on a sample text to initialize vocabulary
            default_text = [
                "company financial report earnings revenue profit loss quarter year fiscal",
                "business market growth strategy development product service customer sales"
            ]

            self.vectorizer.fit(default_text)
            logger.info(f"Initialized TF-IDF vectorizer with vocabulary size: {len(self.vectorizer.vocabulary_)}")

            # Ensure the collection exists
            self._ensure_collection_exists()
            logger.info(f"Successfully initialized QdrantAdapter with {embedding_model}")

        except Exception as e:
            logger.error(f"Failed to initialize QdrantAdapter: {str(e)}", exc_info=True)
            raise

    def _ensure_collection_exists(self):
        """Ensure that the collection exists, and create it if it doesn't."""
        try:
            info = self.client.get_collection(self.collection_name)
            if info.status != "green":
                logger.warning(f"Collection {self.collection_name} exists but status is {info.status}")
                if not self.recover_collection():
                    logger.warning("Recovery failed, recreating collection")
                    self.create_collection()
            else:
                logger.info(f"Found existing collection: {self.collection_name}")
        except Exception as e:
            if "not found" in str(e).lower():
                logger.info(f"Collection {self.collection_name} not found, creating...")
                self.create_collection()
            else:
                logger.error(f"Error ensuring collection exists: {str(e)}")
                raise

    def create_collection(self, collection_name: Optional[str] = None) -> bool:
        """Create a new collection with the specified configuration."""
        try:
            collection_name = collection_name or self.collection_name

            vectors_config = {
                "dense": models.VectorParams(
                    size=self.dense_dim,
                    distance=models.Distance.COSINE
                ),
                "sparse": models.VectorParams(
                    size=self.sparse_dim,
                    distance=models.Distance.COSINE
                )
            }

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,  # Immediate indexing
                    memmap_threshold=20000,  # Better memory management
                    max_optimization_threads=4
                )
            )

            logger.info(f"Created new collection {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(lambda e: not isinstance(e, ValueError))
    )
    def compute_sparse_embedding(self, text: str) -> Dict[str, List[int]]:
        """Compute sparse embedding with logging."""
        try:
            start_time = time.time()
            logger.info("Computing sparse embedding...")
            
            if not text:
                raise ValueError("Input text cannot be empty")
            
            # Transform the text
            sparse_vector = self.vectorizer.transform([text])
            
            # Convert to indices and values
            indices = sparse_vector.indices.tolist()
            values = sparse_vector.data.tolist()
            
            logger.info(f"Generated sparse embedding with {len(indices)} non-zero elements in {time.time() - start_time:.2f}s")
            
            return {
                "indices": indices,
                "values": values
            }
        except Exception as e:
            logger.error(f"Failed to compute sparse embedding: {str(e)}", exc_info=True)
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3)
    )
    def upsert_chunk(self,
                     chunk_text: str,
                     context_text: str,
                     dense_embedding: List[float],
                     metadata: Dict[str, Any],
                     chunk_id: str) -> bool:
        """Upsert a document chunk with both dense and sparse embeddings."""
        try:
            # Input validation
            if not chunk_text or not context_text:
                raise ValueError("Chunk text and context text cannot be empty")
            if not dense_embedding or len(dense_embedding) != self.dense_dim:
                raise ValueError(f"Dense embedding dimension mismatch. Expected {self.dense_dim}, got {len(dense_embedding)}")

            # Generate sparse embedding for the context
            sparse_embedding = self.compute_sparse_embedding(context_text)

            # Create point with both dense and sparse vectors
            point = models.PointStruct(
                id=chunk_id,
                vector={
                    "dense": dense_embedding,
                    "sparse": sparse_embedding
                },
                payload={
                    "chunk_text": chunk_text,
                    "context": context_text,
                    "timestamp": datetime.now().isoformat(),
                    **metadata
                }
            )

            # Upsert point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True
            )

            logger.info(f"Successfully upserted point {chunk_id}")
            return True

        except Exception as e:
            logger.error(f"Error upserting chunk {chunk_id}: {str(e)}")
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=6),
        stop=stop_after_attempt(3)
    )
    def search(self,
               query_text: str,
               query_vector: List[float],
               limit: int = 5,
               use_sparse: bool = True,
               score_threshold: float = 0.7,
               filter_conditions: Optional[Dict] = None) -> List[Dict]:
        """Perform hybrid search using dense and sparse vectors."""
        try:
            # Validate inputs
            if len(query_vector) != self.dense_dim:
                raise ValueError(f"Query vector dimension ({len(query_vector)}) does not match collection ({self.dense_dim})")
            if limit < 1:
                raise ValueError("Limit must be positive")
            if not (0 <= score_threshold <= 1):
                raise ValueError("Score threshold must be between 0 and 1")

            # Prepare search parameters
            search_params = models.SearchParams(
                hnsw_ef=128  # Adjust for better performance
            )

            # Prepare query
            query_vectors = {"dense": query_vector}
            if use_sparse:
                sparse_embedding = self.compute_sparse_embedding(query_text)
                query_vectors["sparse"] = sparse_embedding

            # Build query
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vectors,
                query_filter=models.Filter(**filter_conditions) if filter_conditions else None,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold,
                params=search_params
            )

            return [{
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            } for hit in search_result]

        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "status": getattr(info, "status", "unknown"),
                "vectors_count": getattr(info, "vectors_count", 0),
                "points_count": getattr(info, "points_count", 0),
                "vector_size": self.dense_dim
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {
                "name": self.collection_name,
                "status": "error",
                "error": str(e)
            }

    def delete_collection(self) -> bool:
        """Delete the current collection."""
        try:
            self.client.delete_collection(self.collection_name)
            # Reinitialize vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=self.sparse_dim,
                stop_words='english'
            )
            self.vectorizer.fit([""])  # Initialize with empty vocabulary
            logger.info(f"Deleted collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

    def update_embedding_model(self, new_model: str) -> bool:
        """Update the embedding model and recreate collection if necessary."""
        try:
            new_dim = VECTOR_DIMENSIONS.get(new_model)
            if not new_dim:
                raise ValueError(f"Unknown embedding model: {new_model}")

            if new_dim != self.dense_dim:
                self.embedding_model = new_model
                self.dense_dim = new_dim
                self.create_collection()
                logger.info(f"Updated embedding model to {new_model} and recreated collection")
            return True
        except Exception as e:
            logger.error(f"Error updating embedding model: {str(e)}")
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(is_overloaded_error)
    )
    def extract_metadata(self, doc_text: str, url: str) -> Dict[str, str]:
        """Extract metadata using LLM first, fallback to filename parsing."""
        try:
            # Try LLM-based extraction first
            try:
                prompt = self.DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc_text[:2000])
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )
                metadata = json.loads(response.content)
                metadata.update({
                    'url': url,
                    'file_name': url.split('/')[-1],
                    'creation_date': datetime.now().isoformat()
                })
                return metadata
            except Exception as e:
                logger.warning(f"Failed to extract metadata using Anthropic: {str(e)}")
                
            # Fallback to filename-based extraction
            return self._extract_metadata_from_filename(url)
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            raise

    def recover_collection(self) -> bool:
        """Attempt to recover collection if in a bad state."""
        try:
            info = self.client.get_collection(self.collection_name)
            if info.status != "green":
                logger.warning(f"Collection {self.collection_name} in {info.status} state, attempting recovery")
                self.client.recover_collection(self.collection_name)
                return True
        except Exception as e:
            logger.error(f"Error recovering collection: {str(e)}")
        return False

    def situate_context(self, doc: str, chunk: str) -> str:
        """Generate context for a chunk using both document-level and chunk-specific information."""
        try:
            return rate_limited_context(self._situate_context, doc, chunk)
        except Exception as e:
            logger.error(f"Error generating context: {str(e)}")
            raise

    def _situate_context(self, doc: str, chunk: str) -> str:
        """Internal method to generate context."""
        try:
            # Format the prompt according to Anthropic's requirements
            formatted_prompt = (
                f"\n\nHuman: {CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)}"
                f"\n\nAssistant:"
            )

            response = self.anthropic_client.completions.create(
                model="claude-2",
                max_tokens_to_sample=300,
                prompt=formatted_prompt
            )
            return response.completion.strip()
        except Exception as e:
            logger.error(f"Error generating context: {str(e)}")
            raise

    def process_document(self, doc_text: str, url: str, chunk_size: int = 500, chunk_overlap: int = 50) -> bool:
        """Process a document with detailed logging and proper error handling."""
        try:
            logger.info(f"Starting document processing for {url}")
            
            # Extract metadata
            metadata = self.extract_metadata(doc_text, url)
            logger.info(f"Metadata extracted successfully for {url}")
            
            # Split into chunks - pass both chunk_size and chunk_overlap
            chunks = self._split_text(
                text=doc_text,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            logger.info(f"Document split into {len(chunks)} chunks")
            
            # Process each chunk
            for chunk in chunks:
                try:
                    # Generate dense embedding
                    dense_vector = self.embed_model.embed(texts=[chunk['text']])[0]
                    
                    # Generate sparse embedding
                    sparse_vector = self._generate_sparse_vector(chunk['text'])
                    
                    # Upsert to Qdrant
                    self.upsert_chunk(
                        chunk_text=chunk['text'],
                        dense_embedding=dense_vector,
                        sparse_embedding=sparse_vector,
                        metadata=metadata,
                        chunk_id=chunk['chunk_id']
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
                    continue
                
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {url}: {str(e)}")
            raise

    def _create_point(self, id: str, dense_vector: List[float], sparse_vector: Dict[str, List[int]], payload: Dict) -> models.PointStruct:
        """Create a point for Qdrant with logging."""
        try:
            logger.info(f"Creating point {id}")
            point = models.PointStruct(
                id=id,
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector
                },
                payload=payload
            )
            logger.info(f"Successfully created point {id}")
            return point
        except Exception as e:
            logger.error(f"Failed to create point {id}: {str(e)}")
            raise

    def _split_text(self, text: str, metadata: Dict[str, Any], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        try:
            logger.info(f"Splitting text into chunks with size {chunk_size}")
            splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_text(text)
            
            # Add metadata to each chunk
            processed_chunks = []
            for i, chunk_text in enumerate(chunks):
                processed_chunks.append({
                    'text': chunk_text,
                    'metadata': metadata,
                    'chunk_id': f"{metadata.get('file_name', 'doc')}_{i}"
                })
                
            logger.info(f"Text split into {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            raise

    def _create_point_from_chunk(self, chunk: str, context: str, metadata: Dict) -> models.PointStruct:
        """Create a point for Qdrant from a chunk with logging."""
        try:
            logger.info(f"Creating point from chunk {chunk}")
            dense_vector = self.embed_model.get_text_embedding(chunk)
            sparse_vector = self.compute_sparse_embedding(chunk)
            point = models.PointStruct(
                id=f"{metadata['url']}_{i}",
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector
                },
                payload={
                    "chunk_text": chunk,
                    "context": context,
                    **metadata
                }
            )
            logger.info(f"Successfully created point from chunk {chunk}")
            return point
        except Exception as e:
            logger.error(f"Failed to create point from chunk {chunk}: {str(e)}")
            raise

    def _extract_metadata_from_filename(self, url: str) -> Dict[str, str]:
        """Extract metadata from filename when LLM extraction fails."""
        try:
            # Get filename from URL
            filename = url.split('/')[-1]
            filename = unquote(filename)  # URL decode
            
            # Split filename into components
            parts = filename.replace('.pdf', '').split('_')
            
            # Extract components
            company = parts[0]
            fiscal_period = parts[1]
            year = parts[2]
            doc_type = parts[3] if len(parts) > 3 else 'unknown'
            
            metadata = {
                'company': company,
                'fiscal_period': fiscal_period,
                'date': year,
                'doc_type': doc_type,
                'url': url,
                'file_name': filename,
                'creation_date': datetime.now().isoformat()
            }
            
            logger.info(f"Generated fallback metadata: {metadata}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from filename: {str(e)}")
            # Return basic metadata if extraction fails
            return {
                'url': url,
                'file_name': url.split('/')[-1],
                'creation_date': datetime.now().isoformat()
            }
