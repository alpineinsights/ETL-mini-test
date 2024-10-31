
import streamlit as st
import anthropic
import voyageai
from llama_parse import LlamaParse
from qdrant_client import QdrantClient
from llama_index.embeddings.voyageai import VoyageEmbedding
import tempfile
import shutil
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
from urllib.parse import unquote
import json
from pathlib import Path
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from prompt_config import get_default_prompt, get_custom_prompt

# Page configuration
st.set_page_config(page_title="PDF Processing Pipeline", page_icon="📚", layout="wide")

# Constants
STATE_FILE = "./.processed_urls.json"
TEMP_DIR = Path("./.temp")
TEMP_DIR.mkdir(exist_ok=True)

# Initialize session state
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = set()
    try:
        with open(STATE_FILE) as f:
            st.session_state.processed_urls = set(json.load(f))
    except FileNotFoundError:
        pass

if 'processing_metrics' not in st.session_state:
    st.session_state.processing_metrics = {
        'total_documents': 0,
        'processed_documents': 0,
        'total_chunks': 0,
        'successful_chunks': 0,
        'failed_chunks': 0,
        'cache_hits': 0,
        'start_time': None,
        'errors': []
    }

# Client initialization
with st.expander("Client Initialization", expanded=True):
    try:
        client = anthropic.Client(
            api_key=st.secrets['ANTHROPIC_API_KEY'],
            default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        llama_parser = LlamaParse(
            api_key=st.secrets['LLAMA_PARSE_API_KEY'],
            result_type="text"
        )
        embed_model = VoyageEmbedding(
            model_name="voyage-finance-2",
            voyage_api_key=st.secrets['VOYAGE_API_KEY']
        )
        st.success("✅ All clients initialized successfully")
    except Exception as e:
        st.error(f"❌ Error initializing clients: {str(e)}")
        st.stop()

# Configuration section
with st.expander("Processing Configuration", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size", value=1000, min_value=100, max_value=4000)
        chunk_overlap = st.number_input("Chunk Overlap", value=200, min_value=0, max_value=1000)
        model = st.selectbox(
            "Claude Model",
            options=[
                "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229"
            ],
            index=0,
            help="Select the Claude model to use for processing"
        )
    with col2:
        context_prompt = st.text_area(
            "Context Prompt",
            value=get_default_prompt(),
            height=200,
            help="Customize the prompt for context generation"
        )
        force_reprocess = st.checkbox("Force Reprocess All")
        if st.button("Reset Processing State"):
            st.session_state.processed_urls = set()
            with open(STATE_FILE, 'w') as f:
                json.dump(list(st.session_state.processed_urls), f)
            st.success("Processing state reset")
            st.rerun()

# [Rest of your existing code for document processing functions and main UI...]
# Include all your existing functions and UI code here, unchanged

# Cleanup temp directory on exit
for file in TEMP_DIR.glob("*"):
    try:
        os.remove(file)
    except:
        pass

