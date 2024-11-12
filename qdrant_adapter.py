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
            
            # Initialize TF-IDF vectorizer with fixed vocabulary size and additional settings
            self.vectorizer = TfidfVectorizer(
                max_features=self.sparse_dim,
                min_df=1,  # Include terms that appear in at least 1 document
                max_df=0.95,  # Exclude terms that appear in >95% of documents
                stop_words='english',
                ngram_range=(1, 2),  # Include both unigrams and bigrams
                binary=False,  # Use term frequency instead of binary occurrence
                norm='l2',  # L2 normalization of vectors
                use_idf=True,  # Enable inverse document frequency weighting
                smooth_idf=True,  # Add 1 to document frequencies to prevent division by zero
                sublinear_tf=True  # Apply sublinear scaling to term frequencies
            )
            
            # Initialize with a comprehensive financial vocabulary
            default_text = [
                "company financial report earnings revenue profit loss quarter year fiscal",
                "business market growth strategy development product service customer sales",
                "investment risk management operation technology innovation digital data",
                "balance sheet income statement cash flow assets liabilities equity capital",
                "quarterly annual report guidance forecast outlook performance metric",
                "merger acquisition partnership joint venture subsidiary corporation",
                "dividend stock share price market value trading volume investor shareholder",
                "regulatory compliance audit governance risk management internal control",
                "operational efficiency cost reduction margin improvement productivity",
                "research development innovation product launch market expansion global"
            ]
            
            # Fit vectorizer on default text to initialize vocabulary
            self.vectorizer.fit(default_text)
            logger.info(f"Initialized TF-IDF vectorizer with vocabulary size: {len(self.vectorizer.vocabulary_)}")
            
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
        
    def create_collection(self) -> bool:
        """Create or recreate collection with proper vector configuration."""
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
            
            # Create collection with optimized settings
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                    memmap_threshold=20000,
                    max_optimization_threads=4
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

    def generate_chunk_context(self, chunk_text: str, metadata: Dict[str, Any]) -> str:
        """Generate chunk-specific context using document-level metadata."""
        try:
            # Use metadata for document-level information
            company = metadata.get('company', 'Unknown Company')
            date = metadata.get('date', 'Unknown Date')
            fiscal_period = metadata.get('fiscal_period', 'Unknown Period')
            
            # Create a prompt for chunk-specific context
            chunk_context_prompt = f"""
            Here is a specific chunk from the document:
            <chunk>
            {chunk_text}
            </chunk>

            Using the document-level information identified above, create a context giving a very succinct high-level overview (max 100 characters) of THIS SPECIFIC CHUNK's content focusing on keywords for better retrieval.

            Main company: {company}
            Date: {date}
            Fiscal period: {fiscal_period}

            Answer only with the succinct context, and nothing else (no introduction, no conclusion, no headings).
            """
            
            # Generate the context using the prompt
            response = st.session_state.clients['claude'].messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=100,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": chunk_context_prompt
                }]
            )
            return response.content
        except Exception as e:
            logger.error(f"Chunk context generation failed: {str(e)}")
            raise
