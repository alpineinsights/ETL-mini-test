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

# Configure page
st.set_page_config(
    page_title="PDF Processing Test",
    page_icon="📚",
    layout="wide"
)

# Initialize clients section
with st.expander("Client Initialization", expanded=True):
    try:
        client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])
        llama_parser = LlamaParse(api_key=st.secrets['LLAMA_PARSE_API_KEY'])
        embed_model = VoyageEmbedding(
            model_name="voyage-finance-2",  # Updated model name
            voyage_api_key=st.secrets['VOYAGE_API_KEY']
        )
        st.success("✅ All clients initialized successfully")
    except Exception as e:
        st.error(f"❌ Error initializing clients: {str(e)}")
        st.stop()

# Sitemap processing section
with st.expander("Sitemap Processing", expanded=True):
    st.subheader("Process PDFs from Sitemap")
    
    sitemap_url = st.text_input(
        "Enter Sitemap URL",
        value="https://alpinedatalake7.s3.eu-west-3.amazonaws.com/sitemap.xml"
    )
    
    if st.button("Process Sitemap"):
        try:
            # Fetch and parse sitemap
            st.write("Fetching sitemap...")
            response = requests.get(sitemap_url, timeout=30)
            response.raise_for_status()
            
            # Debug raw response
            st.write("Raw sitemap content:")
            st.code(response.text)
            
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
                st.error("No PDF URLs found in sitemap. URL paths found:")
                # Show all URLs found for debugging
                for ns in namespaces.values():
                    if ns:
                        urls = root.findall(f".//{{{ns}}}loc")
                    else:
                        urls = root.findall(".//loc")
                    for url in urls:
                        st.write(f"- {url.text}")
                st.stop()
                
            st.success(f"Found {len(pdf_urls)} PDF documents")
            
            # Display PDF list with processing options
            for url in pdf_urls:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Decode URL for display
                        decoded_url = unquote(url.split('/')[-1])
                        st.write(f"📄 {decoded_url}")
                    with col2:
                        if st.button("Test Process", key=url):
                            try:
                                # Download PDF
                                st.write("Downloading PDF...")
                                pdf_response = requests.get(url, timeout=30)
                                pdf_response.raise_for_status()
                                
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                    tmp_file.write(pdf_response.content)
                                    tmp_path = tmp_file.name
                                
                                try:
                                    # Parse document
                                    st.write("Parsing document...")
                                    parsed_docs = llama_parser.load_data(tmp_path)
                                    st.success(f"✅ Document parsed - {len(parsed_docs)} sections")
                                    
                                    # Process first section as example
                                    if parsed_docs:
                                        doc = parsed_docs[0]
                                        
                                        # Test chunking
                                        chunk_size = 1000
                                        chunks = []
                                        current_chunk = []
                                        current_length = 0
                                        
                                        for line in doc.text.split('\n'):
                                            line_length = len(line)
                                            if current_length + line_length > chunk_size and current_chunk:
                                                chunks.append('\n'.join(current_chunk))
                                                current_chunk = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                                                current_length = sum(len(line) for line in current_chunk)
                                            
                                            current_chunk.append(line)
                                            current_length += line_length
                                        
                                        if current_chunk:
                                            chunks.append('\n'.join(current_chunk))
                                        
                                        st.write(f"Created {len(chunks)} chunks from first section")
                                        
                                        # Test first chunk processing
                                        if chunks:
                                            st.write("Processing first chunk...")
                                            
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
                                                                "text": "\n\nDocument Content:\n" + chunks[0]
                                                            }
                                                        ]
                                                    }
                                                ],
                                                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
                                            )
                                            st.success("✅ Context generated")
                                            with st.expander("Show Context"):
                                                st.write(context.content[0].text)
                                            
                                            # Generate embedding
                                            embedding = embed_model.get_text_embedding(chunks[0])
                                            st.success(f"✅ Embedding generated (length: {len(embedding)})")
                                            
                                finally:
                                    import os
                                    if os.path.exists(tmp_path):
                                        os.unlink(tmp_path)
                                        
                            except Exception as e:
                                st.error(f"❌ Error processing document: {str(e)}")
                                
        except Exception as e:
            st.error(f"❌ Error processing sitemap: {str(e)}")
