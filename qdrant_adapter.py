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

logger = logging.getLogger(__name__)

VECTOR_DIMENSIONS = {
    "voyage-finance-2": 1024,
    "voyage-large-3": 4096,
    "sparse": 768  # TF-IDF sparse vector dimension
}

class QdrantAdapter:
    """Handles interaction with Qdrant vector database for hybrid search."""
    
    def __init__(self, url: str, api_key: str, collection_name: str = "documents", embedding_model: str = "voyage-finance-2"):
        """
        Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL
            api_key: API key for authentication
            collection_name: Name of the collection to use
            embedding_model: Name of the embedding model to use
        """
        if embedding_model not in ["voyage-finance-2", "voyage-large-3"]:
            raise ValueError("embedding_model must be one of: voyage-finance-2, voyage-large-3")
        
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
            self.sparse_dim = VECTOR_DIMENSIONS["sparse"]
            
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                strip_accents='unicode',
                ngram_range=(1, 2),
                max_features=768,
                sublinear_tf=True
            )
            
            # Try to get collection info or create if doesn't exist
            try:
                info = self.client.get_collection(self.collection_name)
                # Verify vector dimensions match
                if info.config.params["dense"].size != self.dense_dim:
                    logger.warning("Vector dimensions mismatch. Recreating collection...")
                    self.create_collection()
            except Exception:
                self.create_collection()
                
            logger.info(f"Successfully initialized QdrantAdapter with {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize QdrantAdapter: {str(e)}")
            raise
        
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(lambda e: not isinstance(e, ValueError))
    )
    def create_collection(self, sparse_dim: int = 768) -> bool:
        """Create or recreate collection with specified vector dimensions."""
        try:
            # Define vector configurations for both dense and sparse vectors
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
            
            # Create collection
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                    memmap_threshold=20000
                )
            )
            
            logger.info(f"Created collection {self.collection_name}")
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
        """Compute sparse embedding using TF-IDF."""
        try:
            if not text:
                raise ValueError("Input text cannot be empty")
                
            # Fit and transform if vocabulary is empty
            try:
                if not self.vectorizer.vocabulary_:
                    sparse_vector = self.vectorizer.fit_transform([text])
                else:
                    sparse_vector = self.vectorizer.transform([text])
            except Exception as e:
                logger.error("Error in vectorizer, resetting and retrying")
                self.vectorizer = TfidfVectorizer(
                    lowercase=True,
                    strip_accents='unicode',
                    ngram_range=(1, 2),
                    max_features=768,
                    sublinear_tf=True
                )
                sparse_vector = self.vectorizer.fit_transform([text])
            
            # Convert to sparse representation
            indices = sparse_vector.indices.tolist()
            values = sparse_vector.data.tolist()
            
            return {
                "indices": indices,
                "values": values
            }
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
                
            # Compute sparse embedding from context
            sparse_embedding = self.compute_sparse_embedding(context_text)
            
            # Create point
            point = models.PointStruct(
                id=chunk_id,
                vectors={
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
            
            logger.info(f"Successfully upserted chunk {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting chunk {chunk_id}: {str(e)}")
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
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"]
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
            try:
                info = self.client.get_collection(self.collection_name)
                if getattr(info, "status", "unknown") != "green":
                    logger.warning(f"Collection status is {getattr(info, 'status', 'unknown')}")
            except Exception:
                # Collection doesn't exist, create it
                self.create_collection()
                info = self.client.get_collection(self.collection_name)
            
            # Extract only the essential information
            collection_info = {
                "name": getattr(info, "name", self.collection_name),
                "status": getattr(info, "status", "unknown")
            }
            
            # Safely add optional fields
            if hasattr(info, "vectors_count"):
                collection_info["vectors_count"] = info.vectors_count
            if hasattr(info, "points_count"):
                collection_info["points_count"] = info.points_count
                
            return collection_info
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise
    
    def delete_collection(self) -> bool:
        """Delete the current collection."""
        try:
            self.client.delete_collection(self.collection_name)
            # Reset vectorizer
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                strip_accents='unicode',
                ngram_range=(1, 2),
                max_features=768,
                sublinear_tf=True
            )
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
