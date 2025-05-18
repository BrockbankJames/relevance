import streamlit as st
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import json

# Set page config
st.set_page_config(
    page_title="Universal Sentence Encoder Embedding Generator",
    page_icon="🔤",
    layout="centered"
)

# Add title and description
st.title("Universal Sentence Encoder Embedding Generator")
st.markdown("""
This app generates vector embeddings for text and webpages using Google's Universal Sentence Encoder.
You can either enter a keyword or a webpage URL to analyze content similarity.
""")

# Load the Universal Sentence Encoder model
@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

# Load the model
with st.spinner("Loading the Universal Sentence Encoder model..."):
    model = load_model()

def get_embedding(text):
    """Generate embedding for a single text input"""
    return model([text])[0].numpy()

def scrape_webpage(url):
    """Scrape webpage content using ScrapingBee API"""
    api_key = st.secrets["SCRAPINGBEE_API_KEY"]
    if not api_key:
        st.error("ScrapingBee API key not found. Please add it to your Streamlit secrets.")
        return None
    
    try:
        # Encode the URL
        encoded_url = urllib.parse.quote(url)
        
        # Make the request to ScrapingBee API with enhanced parameters
        params = {
            'api_key': api_key,
            'url': encoded_url,
            'render_js': 'true',  # Enable JavaScript rendering
            'premium_proxy': 'true',  # Use premium proxy
            'wait': '5000',  # Wait 5 seconds for JS to load
            'wait_for': 'body',  # Wait for body to be present
            'block_resources': 'true'  # Block unnecessary resources
        }
        
        api_url = "https://app.scrapingbee.com/api/v1/"
        response = requests.get(api_url, params=params)
        
        if response.status_code != 200:
            st.error(f"Error scraping webpage: HTTP {response.status_code}")
            if response.text:
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        st.error(f"API Error: {error_data['error']}")
                        if 'reason' in error_data:
                            st.error(f"Reason: {error_data['reason']}")
                except:
                    st.error(f"API Error: {response.text}")
            return None
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract sections (h tags and p tags)
        sections = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
            text = tag.get_text().strip()
            if text:  # Only include non-empty sections
                sections.append({
                    'type': tag.name,
                    'text': text
                })
        
        return sections
    except Exception as e:
        st.error(f"Error scraping webpage: {str(e)}")
        return None

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Keyword Embedding", "Webpage Analysis"])

with tab1:
    st.subheader("Generate Embedding for Keyword")
    keyword_input = st.text_area(
        "Enter your keyword:",
        height=100,
        placeholder="Type or paste your keyword here..."
    )
    
    if keyword_input:
        with st.spinner("Generating embedding..."):
            keyword_embedding = get_embedding(keyword_input)
        
        # Display embedding information
        st.subheader("Embedding Information")
        st.write(f"Vector dimension: {len(keyword_embedding)}")
        st.write(f"Vector shape: {keyword_embedding.shape}")
        st.write(f"Data type: {keyword_embedding.dtype}")
        
        # Display the embedding vector
        st.subheader("Embedding Vector")
        st.code(keyword_embedding.tolist())
        
        # Add download button for the embedding
        st.download_button(
            label="Download Embedding as CSV",
            data=",".join(map(str, keyword_embedding)),
            file_name="keyword_embedding.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("Analyze Webpage Content")
    url_input = st.text_input("Enter webpage URL:", placeholder="https://example.com")
    
    if url_input:
        with st.spinner("Scraping webpage..."):
            sections = scrape_webpage(url_input)
        
        if sections:
            st.write(f"Found {len(sections)} sections in the webpage")
            
            # Generate embeddings for all sections
            with st.spinner("Generating embeddings for sections..."):
                section_embeddings = []
                for section in sections:
                    embedding = get_embedding(section['text'])
                    section_embeddings.append(embedding)
            
            # If we have a keyword embedding from tab1, calculate similarities
            if 'keyword_embedding' in locals():
                st.subheader("Similarity Analysis")
                
                # Calculate similarities for each section
                similarities = []
                for embedding in section_embeddings:
                    similarity = calculate_similarity(keyword_embedding, embedding)
                    similarities.append(similarity)
                
                # Calculate and display average similarity
                avg_similarity = np.mean(similarities)
                st.metric(
                    label="Page Similarity Score",
                    value=f"{avg_similarity:.3f}",
                    help="Cosine similarity between the keyword and webpage content (range: -1 to 1)"
                )
            else:
                st.info("Enter a keyword in the first tab to analyze similarities with webpage content.")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and Google's Universal Sentence Encoder") 
