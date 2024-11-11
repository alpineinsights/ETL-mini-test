"""
QdrantAdapter class for managing vector storage operations with hybrid search capabilities.
Handles both dense embeddings from Voyage and sparse embeddings from TF-IDF.
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SparseIndexQuery
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class QdrantAdapter:
    """Handles interaction with Qdrant vector database for hybrid search."""
    
    def __init__(self, url: str, api_key: str, collection_name: str = "documents"):
        """
        Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL
            api_key: API key for authentication
            collection_name: Name of the collection to use
        """
        try:
            self.client = QdrantClient(url=url, api_key=api_key)
            self.collection_name = collection_name
            # Initialize TF-IDF vectorizer with parameters optimized for sparse embeddings
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                strip_accents='unicode',
                ngram_range=(1, 2),
                max_features=768,  # Match sparse vector dimension
                sublinear_tf=True  # Apply sublinear scaling to term frequencies
            )
            logger.info("Successfully initialized QdrantAdapter")
        except Exception as e:
            logger.error(f"Failed to initialize QdrantAdapter: {str(e)}")
            raise
        
    def create_collection(self, dense_dim: int = 1024, sparse_dim: int = 768) -> bool:
        """Create or recreate collection with specified vector dimensions."""
        try:
            # Define vector configurations
            vectors_config = {
                "dense": models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE
                ),
                "sparse": models.VectorParams(
                    size=sparse_dim,
                    distance=models.Distance.COSINE
                )
            }
            
            # Create collection with minimal configuration
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config
            )
            
            # Create payload indices
            indices = [
                ("timestamp", "text"),
                ("filename", "keyword"),
                ("chunk_index", "integer"),
                ("url", "keyword"),
                ("source_type", "keyword")
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
        """
        Compute sparse embedding using TF-IDF vectorization.
        
        Args:
            text: Input text to vectorize
            
        Returns:
            Dict containing sparse vector indices and values
        """
        try:
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty")
                
            # Fit and transform the text
            sparse_vector = self.vectorizer.fit_transform([text]).toarray()[0]
            
            # Get non-zero elements
            non_zero_indices = np.nonzero(sparse_vector)[0]
            non_zero_values = sparse_vector[non_zero_indices]
            
            # Validate output
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
                payload={
                    "chunk_text": chunk_text,
                    "context": context_text,
                    "timestamp": datetime.now().isoformat(),  # Store as ISO format string
                    **metadata
                },
                vectors={
                    "dense": dense_embedding,
                    "sparse": models.SparseVector(
                        indices=sparse_embedding["indices"],
                        values=sparse_embedding["values"]
                    )
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

    def search(self,
              query_text: str,
              query_vector: List[float],
              limit: int = 5,
              use_sparse: bool = True,
              score_threshold: float = 0.7,
              filter_conditions: Optional[Dict] = None) -> List[Dict]:
        """Perform hybrid search using dense and sparse vectors."""
        try:
            # Prepare search parameters
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_vector,
                "limit": limit,
                "with_payload": True,
                "with_vectors": False
            }
            
            # Add sparse search if enabled
            if use_sparse:
                sparse_vector = self.compute_sparse_embedding(query_text)
                search_params["query_vector_2"] = models.SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"]
                )
            
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
            info = None
            try:
                info = self.client.get_collection(self.collection_name)
            except Exception:
                # Collection doesn't exist, create it
                self.create_collection()
                info = self.client.get_collection(self.collection_name)
            
            if not info:
                return {
                    "name": self.collection_name,
                    "status": "unknown",
                    "vectors_count": 0,
                    "points_count": 0
                }
                
            return {
                "name": info.name,
                "status": info.status,
                "vectors_count": getattr(info, "vectors_count", 0),
                "points_count": getattr(info, "points_count", 0)
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
