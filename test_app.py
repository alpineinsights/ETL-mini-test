import streamlit as st
import anthropic
import voyageai
from llama_parse import LlamaParse
from qdrant_client import QdrantClient
from llama_index.embeddings.voyageai import VoyageEmbedding
import tempfile
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
from urllib.parse import unquote
import json
from pathlib import Path
import hashlib

# Configure page
st.set_page_config(
    page_title="PDF Processing Pipeline",
    page_icon="📚",
    layout="wide"
)

# Initialize state management
STATE_FILE = "./.processed_urls.json"

def load_processed_urls():
    """Load previously processed URLs from state file"""
    try:
        with open(STATE_FILE) as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()

def save_processed_urls(urls):
    """Save processed URLs to state file"""
    Path(STATE_FILE).parent.mkdir(exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(list(urls), f)

if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = load_processed_urls()

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

# Initialize clients section
with st.expander("Client Initialization", expanded=True):
    try:
        client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])
        llama_parser = LlamaParse(api_key=st.secrets['LLAMA_PARSE_API_KEY'])
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
    with col2:
        force_reprocess = st.checkbox("Force Reprocess All", help="Process all documents even if previously processed")
        reset_state = st.button("Reset Processing State", help="Clear the list of processed documents")
        
        if reset_state:
            st.session_state.processed_urls = set()
            save_processed_urls(st.session_state.processed_urls)
            st.success("Processing state reset successfully")
            st.rerun()

async def process_document(url: str, metrics: dict) -> bool:
    """Process a single document with metrics tracking"""
    try:
        # Download PDF
        pdf_response = requests.get(url, timeout=30)
        pdf_response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_response.content)
            tmp_path = tmp_file.name
            
        try:
            # Parse document
            parsed_docs = llama_parser.load_data(tmp_path)
            
            for doc in parsed_docs:
                # Chunk document
                chunks = []
                current_chunk = []
                current_length = 0
                
                for line in doc.text.split('\n'):
                    line_length = len(line)
                    if current_length + line_length > chunk_size and current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        overlap_text = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                        current_chunk = overlap_text
                        current_length = sum(len(line) for line in current_chunk)
                    
                    current_chunk.append(line)
                    current_length += line_length
                
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                
                metrics['total_chunks'] += len(chunks)
                
                # Process chunks
                for chunk in chunks:
                    try:
                        # Generate context
                        context = client.beta.prompt_caching.messages.create(
                            model="claude-3-haiku-20240307",
                            max_tokens=200,
                            temperature=0,
                            messages=[
                                {
                                    "role": "user", 
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "Please analyze this document chunk and provide a brief contextual summary.",
                                            "cache_control": {"type": "ephemeral"}
                                        },
                                        {
                                            "type": "text",
                                            "text": "\n\nDocument Content:\n" + chunk
                                        }
                                    ]
                                }
                            ],
                            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
                        )
                        
                        # Cache hit tracking
                        if hasattr(context, 'usage') and hasattr(context.usage, 'cache_read_input_tokens'):
                            if context.usage.cache_read_input_tokens > 0:
                                metrics['cache_hits'] += 1
                        
                        # Generate embedding
                        embedding = embed_model.get_text_embedding(chunk)
                        
                        metrics['successful_chunks'] += 1
                        
                    except Exception as e:
                        metrics['failed_chunks'] += 1
                        metrics['errors'].append(f"Chunk processing error in {url}: {str(e)}")
                        continue
            
            return True
            
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        metrics['errors'].append(f"Document processing error for {url}: {str(e)}")
        return False

# Main processing section
st.subheader("Process PDFs from Sitemap")

sitemap_url = st.text_input(
    "Enter Sitemap URL",
    value="https://alpinedatalake7.s3.eu-west-3.amazonaws.com/sitemap.xml"
)

if st.button("Start Processing"):
    try:
        # Reset metrics
        st.session_state.processing_metrics = {
            'total_documents': 0,
            'processed_documents': 0,
            'total_chunks': 0,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'cache_hits': 0,
            'start_time': datetime.now(),
            'errors': []
        }
        
        # Fetch and parse sitemap
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
        # Try different XML namespaces
        namespaces = {
            None: "",
            "ns": "http://www.sitemaps.org/schemas/sitemap/0.9"
        }
        
        pdf_urls = []
        for ns in namespaces.values():
            if ns:
                urls = root.findall(f".//{{{ns}}}loc")
            else:
                urls = root.findall(".//loc")
            
            pdf_urls.extend([url.text for url in urls if url.text.lower().endswith('.pdf')])
            
            if pdf_urls:
                break
        
        if not pdf_urls:
            st.error("No PDF URLs found in sitemap")
            st.stop()
        
        # Filter out already processed URLs unless force_reprocess is True
        if not force_reprocess:
            new_urls = [url for url in pdf_urls if url not in st.session_state.processed_urls]
            skipped = len(pdf_urls) - len(new_urls)
            if skipped > 0:
                st.info(f"Skipping {skipped} previously processed documents")
            pdf_urls = new_urls
        
        if not pdf_urls:
            st.success("No new documents to process!")
            st.stop()
        
        st.session_state.processing_metrics['total_documents'] = len(pdf_urls)
        
        # Process documents with progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_cols = st.columns(4)
        
        for i, url in enumerate(pdf_urls):
            status_text.text(f"Processing document {i+1}/{len(pdf_urls)}: {unquote(url.split('/')[-1])}")
            
            success = process_document(url, st.session_state.processing_metrics)
            if success:
                st.session_state.processing_metrics['processed_documents'] += 1
                st.session_state.processed_urls.add(url)
            
            # Update progress
            progress = (i + 1) / len(pdf_urls)
            progress_bar.progress(progress)
            
            # Update metrics display
            with metrics_cols[0]:
                st.metric("Documents Processed", f"{st.session_state.processing_metrics['processed_documents']}/{st.session_state.processing_metrics['total_documents']}")
            with metrics_cols[1]:
                st.metric("Chunks Processed", st.session_state.processing_metrics['successful_chunks'])
            with metrics_cols[2]:
                st.metric("Cache Hits", st.session_state.processing_metrics['cache_hits'])
            with metrics_cols[3]:
                elapsed = datetime.now() - st.session_state.processing_metrics['start_time']
                st.metric("Processing Time", f"{elapsed.total_seconds():.1f}s")
        
        # Save processed URLs
        save_processed_urls(st.session_state.processed_urls)
        
        # Final summary
        st.success(f"""
            Processing complete!
            - Documents processed: {st.session_state.processing_metrics['processed_documents']}/{st.session_state.processing_metrics['total_documents']}
            - Successful chunks: {st.session_state.processing_metrics['successful_chunks']}
            - Failed chunks: {st.session_state.processing_metrics['failed_chunks']}
            - Cache hits: {st.session_state.processing_metrics['cache_hits']}
            - Total time: {(datetime.now() - st.session_state.processing_metrics['start_time']).total_seconds():.1f}s
        """)
        
        if st.session_state.processing_metrics['errors']:
            with st.expander("Show Errors", expanded=False):
                for error in st.session_state.processing_metrics['errors']:
                    st.error(error)
                    
    except Exception as e:
        st.error(f"Error processing sitemap: {str(e)}")

# Always show current state
with st.expander("Current Processing State", expanded=False):
    st.write(f"Previously processed URLs: {len(st.session_state.processed_urls)}")
    if st.session_state.processed_urls:
        for url in st.session_state.processed_urls:
            st.write(f"- {unquote(url.split('/')[-1])}")
