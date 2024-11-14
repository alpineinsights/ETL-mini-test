import streamlit as st
from qdrant_client import QdrantClient

# Add this at module level
GRPC_OPTIONS = {
    # Keepalive settings
    'grpc.keepalive_time_ms': 10000,          # Send keepalive ping every 10 seconds
    'grpc.keepalive_timeout_ms': 5000,        # Wait 5 seconds for keepalive ping response
    'grpc.http2.max_pings_without_data': 0,   # Allow pings without data
    'grpc.keepalive_permit_without_calls': 1,  # Allow keepalive pings when no calls are in flight
    'grpc.min_time_between_pings_ms': 10000,  # Prevent ping storms
    
    # Message size limits
    'grpc.max_receive_message_length': 100 * 1024 * 1024,  # 100MB max receive size
    'grpc.max_send_message_length': 100 * 1024 * 1024,     # 100MB max send size
    
    # Connection lifecycle
    'grpc.max_connection_idle_ms': 300000,    # 5 minutes idle timeout
    'grpc.max_connection_age_ms': 600000,     # 10 minutes max connection age
    'grpc.max_connection_age_grace_ms': 5000  # 5 seconds grace period
}

def initialize_qdrant():
    """Initialize Qdrant client with optimized gRPC settings."""
    try:
        client = QdrantClient(
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"],
            prefer_grpc=True,      # Enable gRPC protocol
            timeout=60,            # Connection timeout
            grpc_options=GRPC_OPTIONS
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
