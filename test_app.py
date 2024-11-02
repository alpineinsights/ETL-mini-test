"""
Streamlit application for building a contextual retrieval ETL pipeline.
Processes PDFs and generates contextual embeddings using various AI models.
"""

import streamlit as st
import anthropic
import voyageai
from llama_parse import LlamaParse
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
from typing import List, Dict, Any

# Configuration Classes
class CompanyConfig:
    """Manages company names for the prompt system"""
    
    COMPANY_NAMES = [
        "Apple Inc",
        "Microsoft Corporation",
        "Google LLC",
        "Amazon.com Inc",
        # ... Add more companies as needed
    ]
    
    @classmethod
    def get_company_names_prompt(cls):
        """Returns formatted company names for use in the system prompt"""
        return "\n".join(cls.COMPANY_NAMES)
    
    @classmethod
    def validate_company_names(cls):
        """Validates that company names are properly formatted"""
        if not cls.COMPANY_NAMES:
            raise ValueError("Company names list cannot be empty")
        for company in cls.COMPANY_NAMES:
            if not isinstance(company, str):
                raise ValueError(f"Invalid company name type: {type(company)}")
            if len(company.strip()) == 0:
                raise ValueError("Company name cannot be empty")
        return True

class PromptConfig:
    """Manages system prompts used in the application"""
    
    DEFAULT_PROMPT_TEMPLATE = """Give a short succinct context to situate this chunk within the overall enclosed document boader context for the purpose of improving similarity search retrieval of the chunk. 

Make sure to list:
1. The name of the main company mentioned AND any other secondary companies mentioned if applicable. ONLY use company names exact spellings from the list below to facilitate similarity search retrieval.
2. The apparent date of the document (YYYY.MM.DD)
3. Any fiscal period mentioned. ALWAYS use BOTH abreviated tags (e.g. Q1 2024, Q2 2024, H1 2024) AND more verbose tags (e.g. first quarter 2024, second quarter 2024, first semester 2024) to improve retrieval.
4. A very succint high level overview (i.e. not a summary) of the chunk's content in no more than 100 characters with a focus on keywords for better similarity search retrieval

Answer only with the succinct context, and nothing else (no introduction, no conclusion, no headings).

List of company names (use exact spelling) : 
{company_names}"""

    @classmethod
    def get_default_prompt(cls):
        """Returns the complete default prompt with company names"""
        try:
            return cls.DEFAULT_PROMPT_TEMPLATE.format(
                company_names=CompanyConfig.get_company_names_prompt()
            )
        except Exception as e:
            raise Exception(f"Error generating default prompt: {str(e)}")

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
        'total_tokens': 0,
        'start_time': None,
        'errors': []
    }

# Initialize clients
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
    
    # Store clients in session state
    if 'clients' not in st.session_state:
        st.session_state.clients = {
            'anthropic': client,
            'llama_parser': llama_parser,
            'embed_model': embed_model
        }
except Exception as e:
    st.error(f"Error initializing clients: {str(e)}")
    st.stop()
def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string"""
    try:
        # The new Claude API returns the count directly as an integer
        token_count = st.session_state.clients['anthropic'].count_tokens(text)
        return token_count  # No need to access .total_tokens anymore
    except Exception as e:
        st.error(f"Error counting tokens: {str(e)}")
        return 0

def create_semantic_chunks(
    text: str,
    max_tokens: int = 1000,
    overlap_tokens: int = 200
) -> List[Dict[str, Any]]:
    """
    Create semantically meaningful chunks from text while respecting markdown structure.
    """
    # Split text into paragraphs using markdown line breaks
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    previous_paragraphs = []  # Store previous paragraphs for overlap
    
    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)
        
        # Handle paragraphs that exceed max tokens by splitting on sentences
        if paragraph_tokens > max_tokens:
            sentences = [s.strip() + '.' for s in paragraph.split('. ') if s.strip()]
            for sentence in sentences:
                sentence_tokens = count_tokens(sentence)
                
                if current_tokens + sentence_tokens > max_tokens:
                    # Create new chunk with overlap
                    if current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        chunk_tokens = count_tokens(chunk_text)
                        chunks.append({
                            'text': chunk_text,
                            'tokens': chunk_tokens
                        })
                        
                        # Add overlap from previous paragraphs
                        overlap_text = []
                        overlap_token_count = 0
                        for prev in reversed(previous_paragraphs):
                            prev_tokens = count_tokens(prev)
                            if overlap_token_count + prev_tokens <= overlap_tokens:
                                overlap_text.insert(0, prev)
                                overlap_token_count += prev_tokens
                            else:
                                break
                        
                        current_chunk = overlap_text
                        current_tokens = overlap_token_count
                    
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
                
        # Normal paragraph handling
        elif current_tokens + paragraph_tokens > max_tokens:
            # Create new chunk with overlap
            chunk_text = '\n\n'.join(current_chunk)
            chunk_tokens = count_tokens(chunk_text)
            chunks.append({
                'text': chunk_text,
                'tokens': chunk_tokens
            })
            
            # Add overlap from previous paragraphs
            overlap_text = []
            overlap_token_count = 0
            for prev in reversed(previous_paragraphs):
                prev_tokens = count_tokens(prev)
                if overlap_token_count + prev_tokens <= overlap_tokens:
                    overlap_text.insert(0, prev)
                    overlap_token_count += prev_tokens
                else:
                    break
            
            current_chunk = overlap_text
            current_tokens = overlap_token_count
            current_chunk.append(paragraph)
            current_tokens += paragraph_tokens
            
        else:
            current_chunk.append(paragraph)
            current_tokens += paragraph_tokens
        
        previous_paragraphs.append(paragraph)
    
    # Add final chunk if there's content
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        chunk_tokens = count_tokens(chunk_text)
        chunks.append({
            'text': chunk_text,
            'tokens': chunk_tokens
        })
    
    return chunks

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=60),
    retry=retry_if_exception(lambda e: "overloaded_error" in str(e))
)
def get_chunk_context(client, chunk: str, full_doc: str, system_prompt: str, model: str):
    """Get context with retry logic for overload errors"""
    try:
        context = st.session_state.clients['anthropic'].messages.create(
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
        
        temp_path = TEMP_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        
        with open(temp_path, 'wb') as f:
            f.write(pdf_response.content)
        
        try:
            st.write("Parsing document...")
            st.write(f"PDF file size: {os.path.getsize(temp_path)} bytes")
            
            parsed_docs = st.session_state.clients['llama_parser'].load_data(str(temp_path))
            
            if not parsed_docs:
                st.warning(f"No sections found in document: {filename}")
                return False
                
            st.write(f"Found {len(parsed_docs)} sections")
            
            for doc in parsed_docs:
                full_doc_text = doc.text
                st.write(f"Full document length: {len(full_doc_text)} characters")
                
                chunks = create_semantic_chunks(
                    text=full_doc_text,
                    max_tokens=chunk_size,
                    overlap_tokens=chunk_overlap
                )
                
                metrics['total_chunks'] += len(chunks)
                st.write(f"Created {len(chunks)} chunks")
                
                chunk_progress = st.progress(0)
                for i, chunk_data in enumerate(chunks):
                    try:
                        chunk_text = chunk_data['text']
                        chunk_tokens = chunk_data['tokens']
                        
                        st.write(f"Processing chunk {i+1}/{len(chunks)}...")
                        context = get_chunk_context(
                            client=st.session_state.clients['anthropic'],
                            chunk=chunk_text,
                            full_doc=full_doc_text,
                            system_prompt=context_prompt,
                            model=model
                        )
                        
                        embedding = st.session_state.clients['embed_model'].get_text_embedding(chunk_text)
                        
                        metrics['successful_chunks'] += 1
                        metrics['total_tokens'] += chunk_tokens
                        chunk_progress.progress((i + 1) / len(chunks))
                        
                        with st.expander(f"Chunk {i+1} Results", expanded=False):
                            st.write("Context:", context.content[0].text)
                            st.write("Embedding size:", len(embedding))
                            st.write("Tokens in chunk:", chunk_tokens)
                            
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
# Page configuration
st.set_page_config(page_title="PDF Processing Pipeline", page_icon="📚", layout="wide")

# Client verification in UI
with st.expander("Client Initialization", expanded=True):
    if 'clients' in st.session_state:
        st.success("✅ All clients initialized successfully")
    else:
        st.error("❌ Clients not properly initialized")
        st.stop()

# Configuration section
with st.expander("Processing Configuration", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input(
            "Chunk Size (in tokens)", 
            value=1000,
            min_value=100,
            max_value=4000,
            help="Maximum number of tokens per chunk"
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap (in tokens)",
            value=200,
            min_value=0,
            max_value=1000,
            help="Number of tokens to overlap between chunks"
        )
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
            value=PromptConfig.get_default_prompt(),
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
        # Reset metrics for new processing run
        st.session_state.processing_metrics = {
            'total_documents': 0,
            'processed_documents': 0,
            'total_chunks': 0,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'cache_hits': 0,
            'total_tokens': 0,
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
            
        st.write(f"Found {len(pdf_urls)} PDFs")
        
        # Display PDF list in an expander
        with st.expander("Show PDF URLs"):
            for url in pdf_urls:
                st.write(f"- {unquote(url.split('/')[-1])}")
        
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
        
        # Create columns for metrics display
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_cols = st.columns(5)
        
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
            
            # Update metrics display
            with metrics_cols[0]:
                st.metric(
                    "Documents Processed", 
                    f"{st.session_state.processing_metrics['processed_documents']}/{st.session_state.processing_metrics['total_documents']}"
                )
            with metrics_cols[1]:
                st.metric(
                    "Chunks Processed", 
                    st.session_state.processing_metrics['successful_chunks']
                )
            with metrics_cols[2]:
                st.metric(
                    "Cache Hits", 
                    st.session_state.processing_metrics['cache_hits']
                )
            with metrics_cols[3]:
                if st.session_state.processing_metrics['successful_chunks'] > 0:
                    avg_chunk_tokens = (
                        st.session_state.processing_metrics['total_tokens'] / 
                        st.session_state.processing_metrics['successful_chunks']
                    )
                    st.metric(
                        "Avg Tokens/Chunk",
                        f"{avg_chunk_tokens:.0f}"
                    )
            with metrics_cols[4]:
                elapsed = datetime.now() - st.session_state.processing_metrics['start_time']
                st.metric(
                    "Processing Time", 
                    f"{elapsed.total_seconds():.1f}s"
                )
        
        save_processed_urls(st.session_state.processed_urls)
        
        # Final success message with detailed metrics
        st.success(f"""
            Processing complete!
            - Documents processed: {st.session_state.processing_metrics['processed_documents']}/{st.session_state.processing_metrics['total_documents']}
            - Successful chunks: {st.session_state.processing_metrics['successful_chunks']}
            - Failed chunks: {st.session_state.processing_metrics['failed_chunks']}
            - Cache hits: {st.session_state.processing_metrics['cache_hits']}
            - Average tokens per chunk: {avg_chunk_tokens:.0f}
            - Total tokens processed: {st.session_state.processing_metrics['total_tokens']:,}
            - Total processing time: {(datetime.now() - st.session_state.processing_metrics['start_time']).total_seconds():.1f}s
        """)
        
        # Display any errors in an expander
        if st.session_state.processing_metrics['errors']:
            with st.expander("Show Errors", expanded=False):
                for error in st.session_state.processing_metrics['errors']:
                    st.error(error)
                    
    except Exception as e:
        st.error(f"Error processing sitemap: {str(e)}")

# Show current processing state
with st.expander("Current Processing State", expanded=False):
    st.write(f"Previously processed URLs: {len(st.session_state.processed_urls)}")
    if st.session_state.processed_urls:
        for url in sorted(st.session_state.processed_urls):
            st.write(f"- {unquote(url.split('/')[-1])}")
            
    # Add token statistics
    if st.session_state.processing_metrics['successful_chunks'] > 0:
        st.write("\nToken Statistics:")
        avg_tokens = (
            st.session_state.processing_metrics['total_tokens'] / 
            st.session_state.processing_metrics['successful_chunks']
        )
        st.write(f"- Average tokens per chunk: {avg_tokens:.0f}")
        st.write(f"- Total tokens processed: {st.session_state.processing_metrics['total_tokens']:,}")

# Cleanup temp directory on exit
for file in TEMP_DIR.glob("*"):
    try:
        os.remove(file)
    except:
        pass
