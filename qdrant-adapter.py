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
        """
        Create or recreate collection with specified vector dimensions.
        
        Args:
            dense_dim: Dimension of dense vectors (Voyage embeddings)
            sparse_dim: Dimension of sparse vectors (TF-IDF)
            
        Returns:
            bool: True if successful
        """
        try:
            # Define vector configurations
            dense_config = VectorParams(
                size=dense_dim,
                distance=Distance.COSINE,
                on_disk=True  # Store vectors on disk for better memory usage
            )
            sparse_config = VectorParams(
                size=sparse_dim,
                distance=Distance.COSINE,
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
                ("timestamp", "datetime"),
                ("filename", "keyword"),
                ("chunk_index", "integer")
            ]
            
            for field_name, field_type in indices:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
            
            logger.info(f"Created collection {self.collection_name} with dense dim={dense_dim}, sparse dim={sparse_dim}")
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
        """
        Upsert a document chunk with both dense and sparse embeddings.
        
        Args:
            chunk_text: Original chunk text
            context_text: Generated context for the chunk
            dense_embedding: Dense vector from Voyage
            metadata: Additional metadata for the chunk
            chunk_id: Unique identifier for the chunk
            
        Returns:
            bool: True if successful
        """
        try:
            # Input validation
            if not chunk_text or not context_text:
                raise ValueError("Chunk text and context text cannot be empty")
            if not dense_embedding or len(dense_embedding) == 0:
                raise ValueError("Dense embedding cannot be empty")
                
            # Compute sparse embedding from context
            sparse_embedding = self.compute_sparse_embedding(context_text)
            
            # Prepare timestamp
            current_time = datetime.now().isoformat()
            
            # Create point structure
            point = PointStruct(
                id=chunk_id,
                payload={
                    "chunk_text": chunk_text,
                    "context": context_text,
                    "timestamp": current_time,
                    "vector_type": "hybrid",  # Mark as using both dense and sparse
                    **metadata
                },
                vectors={
                    "dense": dense_embedding,
                    "sparse": sparse_embedding
                }
            )
            
            # Upsert point to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True  # Ensure point is fully indexed
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
        """
        Perform hybrid search using dense and optionally sparse vectors.
        
        Args:
            query_text: Search query text
            query_vector: Dense query vector
            limit: Maximum number of results
            use_sparse: Whether to include sparse vector search
            score_threshold: Minimum similarity score
            filter_conditions: Additional filtering conditions
            
        Returns:
            List of search results with scores and payloads
        """
        try:
            # Input validation
            if not query_text or not query_vector:
                raise ValueError("Query text and vector cannot be empty")
                
            # Prepare sparse query if enabled
            sparse_query = None
            if use_sparse:
                sparse_embedding = self.compute_sparse_embedding(query_text)
                sparse_query = SparseIndexQuery(**sparse_embedding)
            
            # Build search conditions
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": ("dense", query_vector),
                "limit": limit,
                "score_threshold": score_threshold,
                "with_payload": True,  # Include full payload in results
                "with_vectors": False  # Don't return vectors for efficiency
            }
            
            # Add sparse query if available
            if sparse_query:
                search_params["sparse_query"] = ("sparse", sparse_query)
                
            # Add filter if provided
            if filter_conditions:
                search_params["query_filter"] = models.Filter(**filter_conditions)
            
            # Execute search
            results = self.client.search(**search_params)
            
            # Format results
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