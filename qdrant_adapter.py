"""
QdrantAdapter class for managing vector storage operations with hybrid search capabilities.
Handles both dense embeddings from Voyage and sparse embeddings from TF-IDF.
"""

from qdrant_client import QdrantClient, models
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import uuid
import streamlit as st
from ratelimit import limits, sleep_and_retry
from pathlib import Path

logger = logging.getLogger(__name__)

VECTOR_DIMENSIONS = {
    "voyage-finance-2": 1024,
    "voyage-large-3": 4096,
    "sparse": 768  # TF-IDF sparse vector dimension
}

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>

Based on the ENTIRE document above, identify:
1. The main company name and any secondary companies mentioned (use EXACT spellings)
2. The document date (YYYY.MM.DD format)
3. Any fiscal periods mentioned (use BOTH abbreviated tags like Q1 2024 AND verbose tags like first quarter 2024)

This information will be used for ALL chunks from this document.
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
    return func(*args, **kwargs)

class QdrantAdapter:
    """Handles interaction with Qdrant vector database for hybrid search."""
    
    def __init__(self, url: str, api_key: str, collection_name: str = "documents", embedding_model: str = "voyage-finance-2", anthropic_client = None):
        """
        Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL
            api_key: API key for authentication
            collection_name: Name of the collection to use
            embedding_model: Name of the embedding model to use
            anthropic_client: Initialized Anthropic client for context generation
        """
        try:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=60,
                prefer_grpc=False  # Force HTTP protocol
            )
            self.collection_name = collection_name
            self.embedding_model = embedding_model
            self.dense_dim = VECTOR_DIMENSIONS[embedding_model]
            self.sparse_dim = 100  # Reduced dimension for TF-IDF
            self.anthropic_client = anthropic_client  # Store the Anthropic client
            
            # Initialize TF-IDF vectorizer with reduced vocabulary size
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
            
            # More careful collection initialization
            try:
                info = self.client.get_collection(self.collection_name)
                # Only check if collection exists, don't recreate
                logger.info(f"Found existing collection: {self.collection_name}")
            except Exception as e:
                if "not found" in str(e).lower():
                    logger.info(f"Collection {self.collection_name} not found, creating...")
                    self.create_collection()
                else:
                    raise
            
            logger.info(f"Successfully initialized QdrantAdapter with {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize QdrantAdapter: {str(e)}")
            raise
        
    def create_collection(self, collection_name: Optional[str] = None) -> bool:
        """Create collection if it doesn't exist."""
        try:
            collection_name = collection_name or self.collection_name
            
            # Check if collection exists
            try:
                info = self.client.get_collection(collection_name)
                if info.status != "green":
                    logger.warning(f"Collection {collection_name} exists but status is {info.status}")
                    if self.recover_collection():
                        return True
                    logger.warning("Recovery failed, recreating collection")
                else:
                    return True  # Collection exists and is healthy
            except Exception as e:
                if "not found" not in str(e).lower():
                    raise

            # Create collection with optimized settings
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
    def compute_sparse_embedding(self, text: str) -> List[float]:
        """Compute sparse embedding using TF-IDF."""
        try:
            # Transform text to sparse vector
            sparse_matrix = self.vectorizer.transform([text])
            
            # Convert sparse matrix to dense array and pad/truncate to match required dimension
            dense_array = sparse_matrix.toarray().flatten()
            
            # Pad with zeros if necessary
            if len(dense_array) < self.sparse_dim:
                dense_array = np.pad(dense_array, (0, self.sparse_dim - len(dense_array)))
            # Truncate if necessary
            elif len(dense_array) > self.sparse_dim:
                dense_array = dense_array[:self.sparse_dim]
            
            return dense_array.tolist()
            
        except Exception as e:
            logger.error(f"Error computing sparse embedding: {str(e)}")
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(lambda e: not isinstance(e, ValueError))
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
            if not dense_embedding or len(dense_embedding) == 0:
                raise ValueError("Dense embedding cannot be empty")
            
            # Generate UUID for point ID
            point_id = str(uuid.uuid4())
            
            # Generate sparse embedding for the context
            sparse_vector = self.compute_sparse_embedding(context_text)
            
            # Create point with both dense and sparse vectors
            point = models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_embedding,
                    "sparse": sparse_vector
                },
                payload={
                    "chunk_text": chunk_text,
                    "context": context_text,
                    "timestamp": datetime.now().isoformat(),
                    **metadata
                }
            )
            
            # Validate vector dimensions
            if len(dense_embedding) != self.dense_dim:
                raise ValueError(f"Dense vector dimension mismatch. Expected {self.dense_dim}, got {len(dense_embedding)}")
            
            # Upsert point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True
            )
            
            logger.info(f"Successfully upserted point {point_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting chunk: {str(e)}")
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(lambda e: not isinstance(e, ValueError))
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
            if score_threshold < 0 or score_threshold > 1:
                raise ValueError("Score threshold must be between 0 and 1")
            
            # Prepare search parameters
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": ("dense", query_vector),  # Specify vector name
                "limit": limit,
                "with_payload": True,
                "with_vectors": False,
                "score_threshold": score_threshold
            }
            
            # Add sparse search if enabled
            if use_sparse:
                sparse_vector = self.compute_sparse_embedding(query_text)
                search_params["query_vector_2"] = ("sparse", models.SparseVector(
                    indices=[],
                    values=sparse_vector
                ))
            
            # Add filter if provided
            if filter_conditions:
                search_params["query_filter"] = models.Filter(**filter_conditions)
            
            # Execute search
            results = self.client.search(**search_params)
            
            return [{
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            } for hit in results]
            
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
            # Reset vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=self.sparse_dim,
                stop_words='english'
            )
            # Fit vectorizer on initial empty text to initialize vocabulary
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

    def situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:
        """Generate context using both document-level and chunk-specific information with caching."""
        return rate_limited_context(self._situate_context, doc, chunk)

    def recover_collection(self) -> bool:
        """Attempt to recover collection if in a bad state"""
        try:
            info = self.client.get_collection(self.collection_name)
            if info.status != "green":
                logger.warning(f"Collection {self.collection_name} in {info.status} state, attempting recovery")
                self.client.recover_collection(self.collection_name)
                return True
        except Exception as e:
            logger.error(f"Error recovering collection: {str(e)}")
        return False

    def extract_metadata(self, doc_text: str, url: str) -> Dict[str, Any]:
        """Extract metadata from document text using Claude with prompt caching."""
        try:
            # Use prompt caching for document-level analysis
            response = self.anthropic_client.beta.prompt_caching.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                system=[{
                    "type": "text",
                    "text": DOCUMENT_CONTEXT_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }],
                messages=[{
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"Document text:\n{doc_text[:2000]}"
                    }
                }]
            )
            
            # Parse response and extract metadata
            try:
                # Extract metadata from Claude's response
                metadata = {
                    "company": None,
                    "date": None,
                    "fiscal_period": None,
                    "url": url,
                    "file_name": Path(url).name,
                    "creation_date": datetime.now().isoformat()
                }
                
                # Parse Claude's response to extract the fields
                response_text = response.content[0].text
                for line in response_text.split('\n'):
                    if line.startswith('1.') and 'company' in line.lower():
                        metadata['company'] = line.split(':')[-1].strip()
                    elif line.startswith('2.') and 'date' in line.lower():
                        metadata['date'] = line.split(':')[-1].strip()
                    elif line.startswith('3.') and 'fiscal' in line.lower():
                        metadata['fiscal_period'] = line.split(':')[-1].strip()
                
                return metadata
                
            except Exception as e:
                logger.warning(f"Failed to parse metadata from Claude response: {str(e)}")
                # Fall back to filename-based metadata
                filename = Path(url).name
                parts = filename.replace('.pdf', '').split('_')
                return {
                    "company": parts[0] if len(parts) > 0 else "Unknown",
                    "fiscal_period": parts[1] if len(parts) > 1 else None,
                    "date": parts[3] if len(parts) > 3 else None,
                    "url": url,
                    "file_name": filename,
                    "creation_date": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            raise

    def _situate_context(self, doc: str, chunk: str) -> tuple[str, Any]:
        """Generate context using both document-level and chunk-specific information with caching."""
        try:
            response = self.anthropic_client.beta.prompt_caching.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                system=[{
                    "type": "text",
                    "text": CHUNK_CONTEXT_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }],
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"Document text:\n{doc[:2000]}\n\nChunk text:\n{chunk}"
                    }]
                }]
            )
            return response.content[0].text, response
        except Exception as e:
            logger.error(f"Error generating context: {str(e)}")
            raise
