import streamlit as st
from qdrant_client import QdrantClient

def initialize_qdrant():
    """Initialize Qdrant client with optimized gRPC settings."""
    try:
        # Optimized gRPC configuration
        grpc_options = {
            'grpc.keepalive_time_ms': 10000,          # Send keepalive ping every 10 seconds
            'grpc.keepalive_timeout_ms': 5000,        # Wait 5 seconds for keepalive ping response
            'grpc.http2.max_pings_without_data': 0,   # Allow pings without data
            'grpc.keepalive_permit_without_calls': 1,  # Allow keepalive pings when no calls are in flight
            'grpc.max_receive_message_length': 100 * 1024 * 1024  # 100MB max message size
        }

        client = QdrantClient(
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"],
            prefer_grpc=True,  # Enable gRPC protocol
            timeout=60,        # Connection timeout
            grpc_options=grpc_options
        )
        
        # Test connection by getting collections list
        collections = client.get_collections()
        if collections is None:
            st.error("Failed to get collections from Qdrant")
            return None
            
        return client
        
    except Exception as e:
        st.error(f"Failed to initialize Qdrant client: {str(e)}")
        return None 
