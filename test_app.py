"""
Streamlit application for building a contextual retrieval ETL pipeline.
Processes PDFs and generates contextual embeddings using various AI models.
"""

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

# Token counting and chunking functions
def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string"""
    try:
        token_count = client.count_tokens(text)
        return token_count.total_tokens
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
    
    Args:
        text: The input text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
    
    Returns:
        List of dictionaries containing chunk text and token count
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
                
                # Create semantic chunks using our new function
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
                            client=client,
                            chunk=chunk_text,
                            full_doc=full_doc_text,
                            system_prompt=context_prompt,
                            model=model
                        )
                        
                        embedding = embed_model.get_text_embedding(chunk_text)
                        
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

# Page configuration and main UI code...
[Previous UI code remains the same, but with these changes in the metrics display:]

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
