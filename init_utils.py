import streamlit as st
from qdrant_client import QdrantClient

def initialize_qdrant():
    """Initialize Qdrant client with proper timeout settings."""
    try:
        client = QdrantClient(
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"],
            timeout=60  # Add timeout for cloud deployments
        )
        return client
    except Exception as e:
        st.error(f"Failed to initialize Qdrant client: {str(e)}")
        return None 
