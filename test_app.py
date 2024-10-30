import streamlit as st
import anthropic
import voyageai
from llama_parse import LlamaParse
from qdrant_client import QdrantClient
from llama_index.embeddings.voyageai import VoyageEmbedding

# Configure page
st.set_page_config(
    page_title="Dependency Test",
    page_icon="📚",
    layout="wide"
)

st.title("Enhanced Dependency Test")

# Test API keys
if not all(key in st.secrets for key in ['ANTHROPIC_API_KEY', 'VOYAGE_API_KEY', 'LLAMA_PARSE_API_KEY', 'QDRANT_API_KEY']):
    st.error("Missing required API keys in secrets")
    st.stop()

# Container for client initialization tests
with st.expander("Client Initialization Tests", expanded=True):
    try:
        # Test Anthropic
        client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])
        st.success("✅ Anthropic client initialized")
    except Exception as e:
        st.error(f"❌ Anthropic error: {str(e)}")

    try:
        # Test VoyageAI
        voyage_client = voyageai.Client(api_key=st.secrets['VOYAGE_API_KEY'])
        st.success("✅ VoyageAI client initialized")
    except Exception as e:
        st.error(f"❌ VoyageAI error: {str(e)}")

    try:
        # Test LlamaParse
        llama_parser = LlamaParse(api_key=st.secrets['LLAMA_PARSE_API_KEY'])
        st.success("✅ LlamaParse client initialized")
    except Exception as e:
        st.error(f"❌ LlamaParse error: {str(e)}")

    try:
        # Test Qdrant
        qdrant_client = QdrantClient(
            url="https://some-test-url.com",
            api_key=st.secrets['QDRANT_API_KEY']
        )
        st.success("✅ Qdrant client initialized")
    except Exception as e:
        st.error(f"❌ Qdrant error: {str(e)}")

# Test embedding functionality
with st.expander("Embedding Test", expanded=True):
    st.subheader("Test Embedding Generation")
    
    test_text = st.text_area("Enter text to test embedding", value="This is a test sentence.")
    
    if st.button("Generate Embedding"):
        try:
            # Test direct VoyageAI embedding
            voyage_embedding = voyage_client.embed([test_text], model="voyage-2").embeddings[0]
            st.success("✅ Direct VoyageAI embedding generated")
            st.write(f"Embedding length: {len(voyage_embedding)}")
            
            # Test LlamaIndex VoyageEmbedding
            embed_model = VoyageEmbedding(
                model_name="voyage-2",
                voyage_api_key=st.secrets['VOYAGE_API_KEY']
            )
            llama_embedding = embed_model.get_text_embedding(test_text)
            st.success("✅ LlamaIndex VoyageEmbedding generated")
            st.write(f"Embedding length: {len(llama_embedding)}")
            
        except Exception as e:
            st.error(f"❌ Embedding error: {str(e)}")
