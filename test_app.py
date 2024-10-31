"""
Streamlit application for building a contextual retrieval ETL pipeline.
Processes PDFs and generates contextual embeddings using various AI models.
"""

import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from prompt_config import get_default_prompt, get_custom_prompt

# Page configuration
st.set_page_config(page_title="PDF Processing Pipeline", page_icon="📚", layout="wide")

# Constants and initial setup
STATE_FILE = "./.processed_urls.json"
TEMP_DIR = Path("./.temp")
TEMP_DIR.mkdir(exist_ok=True)

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

# Initialize session state
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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=60),
    retry=retry_if_exception(lambda e: "overloaded_error" in str(e))
)
def get_chunk_context(client, chunk: str, full_doc: str, system_prompt: str, model: str):
    """Get context with retry logic for overload errors"""
    try:
        context = client.messages.create(
            model=model,
            max_tokens=200,
            system=system_prompt,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": "<document>\n" + full_doc + "\n</document>",
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": "\nHere is the chunk we want to situate within the whole document:\n<chunk>\n" + chunk + "\n</chunk>"
                        }
                    ]
                }
            ]
        )
        return context
    except Exception as e:
        if "overloaded_error" in str(e):
            st.warning(f"Claude is overloaded, retrying in a few seconds...")
            raise e
        raise e

def process_document(url: str, metrics: dict, model: str, context_prompt: str) -> bool:
    """Process a single document URL"""
    try:
        filename = unquote(url.split('/')[-1])
        st.write(f"Downloading {filename}...")
        
        pdf_response = requests.get(url, timeout=30)
        pdf_response.raise_for_status()
        
        # Save to persistent temp directory with unique name
        temp_path = TEMP_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        
        with open(temp_path, 'wb') as f:
            f.write(pdf_response.content)
        
        try:
            st.write("Parsing document...")
            st.write(f"PDF file size: {os.path.getsize(temp_path)} bytes")
            
            parsed_docs = llama_parser.load_data(str(temp_path))
            
            if not parsed_docs:
                st.warning(f"No sections found in document: {filename}")
                return False
                
            st.write(f"Found {len(parsed_docs)} sections")
            
            for doc in parsed_docs:
                full_doc_text = doc.text
                st.write(f"Full document length: {len(full_doc_text)} characters")
                
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
                st.write(f"Created {len(chunks)} chunks")
                
                chunk_progress = st.progress(0)
                for i, chunk in enumerate(chunks):
                    try:
                        st.write(f"Processing chunk {i+1}/{len(chunks)}...")
                        context = get_chunk_context(
                            client=client,
                            chunk=chunk,
                            full_doc=full_doc_text,
                            system_prompt=context_prompt,
                            model=model
                        )
                        
                        embedding = embed_model.get_text_embedding(chunk)
                        
                        metrics['successful_chunks'] += 1
                        chunk_progress.progress((i + 1) / len(chunks))
                        
                        with st.expander(f"Chunk {i+1} Results", expanded=False):
                            st.write("Context:", context.content[0].text)
                            st.write("Embedding size:", len(embedding))
                            
                        if hasattr(context, 'usage') and hasattr(context.usage, 'cache_read_input_tokens'):
                            if context.usage.cache_read_input_tokens > 0:
                                metrics['cache_hits'] += 1
                        
                    except Exception as e:
                        metrics['failed_chunks'] += 1
                        metrics['errors'].append(f"Chunk processing error in {url}: {str(e)}")
                        st.error(f"Error processing chunk: {str(e)}")
                        continue
            
            return True
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        metrics['errors'].append(f"Document processing error for {url}: {str(e)}")
        st.error(f"Error processing document: {str(e)}")
        return False

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
            save_processed_urls(st.session_state.processed_urls)
            st.success("Processing state reset")
            st.rerun()

# Main UI section
st.title("PDF Processing Pipeline")
st.subheader("Process PDFs from Sitemap")

sitemap_url = st.text_input(
    "Enter Sitemap URL",
    value="https://alpinedatalake7.s3.eu-west-3.amazonaws.com/sitemap.xml"
)

if st.button("Start Processing"):
    try:
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
        
        st.write("Fetching sitemap...")
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
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
            st.code(response.text, language="xml")
            st.stop()
            
        st.write("Found PDFs:", pdf_urls)
        
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
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_cols = st.columns(4)
        
        for i, url in enumerate(pdf_urls):
            status_text.text(f"Processing document {i+1}/{len(pdf_urls)}: {unquote(url.split('/')[-1])}")
            
            success = process_document(
                url=url,
                metrics=st.session_state.processing_metrics,
                model=model,
                context_prompt=context_prompt
            )
            if success:
                st.session_state.processing_metrics['processed_documents'] += 1
                st.session_state.processed_urls.add(url)
            
            progress_bar.progress((i + 1) / len(pdf_urls))
            
            with metrics_cols[0]:
                st.metric("Documents Processed", f"{st.session_state.processing_metrics['processed_documents']}/{st.session_state.processing_metrics['total_documents']}")
            with metrics_cols[1]:
                st.metric("Chunks Processed", st.session_state.processing_metrics['successful_chunks'])
            with metrics_cols[2]:
                st.metric("Cache Hits", st.session_state.processing_metrics['cache_hits'])
            with metrics_cols[3]:
                elapsed = datetime.now() - st.session_state.processing_metrics['start_time']
                st.metric("Processing Time", f"{elapsed.total_seconds():.1f}s")
        
        save_processed_urls(st.session_state.processed_urls)
        
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

with st.expander("Current Processing State", expanded=False):
    st.write(f"Previously processed URLs: {len(st.session_state.processed_urls)}")
    if st.session_state.processed_urls:
        for url in sorted(st.session_state.processed_urls):
            st.write(f"- {unquote(url.split('/')[-1])}")

# Cleanup temp directory on exit
for file in TEMP_DIR.glob("*"):
    try:
        os.remove(file)
    except:
        pass
