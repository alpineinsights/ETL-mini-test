import streamlit as st
from qdrant_client import QdrantClient, models
from typing import Optional
# Optimized gRPC configuration
GRPC_OPTIONS = {
    # Connection Management
    'grpc.keepalive_time_ms': 10000,
    'grpc.keepalive_timeout_ms': 5000,
    'grpc.keepalive_permit_without_calls': 1,
    'grpc.http2.max_pings_without_data': 0,
    'grpc.min_time_between_pings_ms': 10000,
    
    # Message Size Limits
    'grpc.max_receive_message_length': 100 * 1024 * 1024,
    'grpc.max_send_message_length': 100 * 1024 * 1024,
    
    # Connection Lifecycle
    'grpc.max_connection_idle_ms': 300000,
    'grpc.max_connection_age_ms': 600000,
    'grpc.max_connection_age_grace_ms': 5000
}
def get_qdrant_client(url: str, api_key: str, timeout: int = 60) -> Optional[QdrantClient]:
    """Create a Qdrant client with optimized settings."""
    try:
        client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=True,
            timeout=timeout,
            grpc_options=GRPC_OPTIONS
        )
        return client
    except Exception as e:
        st.error(f"Failed to create Qdrant client: {str(e)}")
        return None
from qdrant_client import QdrantClient

def initialize_qdrant():
    """Initialize Qdrant client with optimized settings."""
    """Initialize Qdrant client with proper timeout settings."""
    try:
        client = get_qdrant_client(
        client = QdrantClient(
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"]
            api_key=st.secrets["QDRANT_API_KEY"],
            timeout=60,  # Add timeout for cloud deployments
            prefer_grpc=False  # Force HTTP protocol
        )

        if client is None:
            return None
            
        # Test connection
        # Test connection by getting collections list instead of collection info
        collections = client.get_collections()
        if collections is None:
            st.error("Failed to get collections from Qdrant")
