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
        # Clean and validate URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Encode the URL properly
        encoded_url = urllib.parse.quote(url, safe=':/?=&')
        
        # Log the request details (for debugging)
        st.write("Debug info (will be removed in production):")
        st.write(f"Original URL: {url}")
        st.write(f"Encoded URL: {encoded_url}")
        
        # Make the request to ScrapingBee API with enhanced parameters
        params = {
            'api_key': api_key,
            'url': encoded_url,
            'render_js': 'true',
            'premium_proxy': 'true',
            'wait': '5000',
            'wait_for': 'body',
            'block_resources': 'true'
        }
        
        # Log the full API URL (without the API key)
        debug_url = f"https://app.scrapingbee.com/api/v1/?url={encoded_url}&render_js=true&premium_proxy=true"
        st.write(f"API URL (without key): {debug_url}")
        
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
                    st.error(f"Raw API Response: {response.text}")
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
        
        if not sections:
            st.warning("No content sections found on the page. The page might be blocking access or have no text content.")
        
        return sections
    except Exception as e:
        st.error(f"Error scraping webpage: {str(e)}")
        return None

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Keyword Embedding", "Webpage Analysis", "Link Profile Analysis"])

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
            
            # If we have a keyword embedding from tab1, calculate similarity
            if 'keyword_embedding' in locals():
                st.subheader("Similarity Analysis")
                
                # Average all section embeddings
                avg_section_embedding = np.mean(section_embeddings, axis=0)
                
                # Calculate single cosine similarity between keyword and averaged section embeddings
                similarity = calculate_similarity(keyword_embedding, avg_section_embedding)
                
                # Display the similarity score
                st.metric(
                    label="Page Similarity Score",
                    value=f"{similarity:.3f}",
                    help="Cosine similarity between the keyword and the average of all webpage content embeddings (range: -1 to 1)"
                )
                
                # Add explanation
                st.info("""
                This similarity score is calculated by:
                1. Averaging all section embeddings from the webpage
                2. Computing cosine similarity between the keyword embedding and the averaged webpage embedding
                """)
            else:
                st.info("Enter a keyword in the first tab to analyze similarities with webpage content.")

with tab3:
    st.subheader("Analyze Multiple URLs")
    st.markdown("""
    Enter a list of URLs (one per line) to analyze their relevance to your keyword.
    The app will analyze each URL and show:
    - Average relevance across all URLs
    - Most relevant URLs ranked by similarity
    - Individual URL scores
    """)
    
    # Get URLs input
    urls_input = st.text_area(
        "Enter URLs (one per line):",
        height=150,
        placeholder="https://example1.com\nhttps://example2.com\nhttps://example3.com"
    )
    
    if urls_input and 'keyword_embedding' in locals():
        # Process URLs
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        
        if urls:
            # Store results for each URL
            url_results = []
            
            # Process each URL
            with st.spinner(f"Analyzing {len(urls)} URLs..."):
                for url in urls:
                    try:
                        # Scrape and analyze the URL
                        sections = scrape_webpage(url)
                        if sections:
                            # Generate embeddings for all sections
                            section_embeddings = []
                            for section in sections:
                                embedding = get_embedding(section['text'])
                                section_embeddings.append(embedding)
                            
                            # Average section embeddings
                            avg_section_embedding = np.mean(section_embeddings, axis=0)
                            
                            # Calculate similarity
                            similarity = calculate_similarity(keyword_embedding, avg_section_embedding)
                            
                            url_results.append({
                                'url': url,
                                'similarity': similarity,
                                'sections_count': len(sections)
                            })
                    except Exception as e:
                        st.warning(f"Error processing {url}: {str(e)}")
            
            if url_results:
                # Sort URLs by similarity
                url_results.sort(key=lambda x: x['similarity'], reverse=True)
                
                # Calculate average similarity
                avg_similarity = np.mean([r['similarity'] for r in url_results])
                
                # Display overall results
                st.subheader("Overall Results")
                st.metric(
                    label="Average Profile Relevance",
                    value=f"{avg_similarity:.3f}",
                    help="Average similarity across all analyzed URLs"
                )
                
                # Display top 3 most relevant URLs
                st.subheader("Most Relevant URLs")
                for i, result in enumerate(url_results[:3], 1):
                    st.markdown(f"""
                    **{i}. {result['url']}**  
                    Similarity: {result['similarity']:.3f}  
                    Sections analyzed: {result['sections_count']}
                    """)
                
                # Display all URLs in a table
                st.subheader("All URLs Analysis")
                # Create a DataFrame for better display
                import pandas as pd
                df = pd.DataFrame(url_results)
                df['similarity'] = df['similarity'].round(3)
                df = df.rename(columns={
                    'url': 'URL',
                    'similarity': 'Similarity Score',
                    'sections_count': 'Sections Analyzed'
                })
                st.dataframe(df, use_container_width=True)
                
                # Add download button for the results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="url_analysis_results.csv",
                    mime="text/csv"
                )
            else:
                st.error("No URLs were successfully analyzed. Please check the URLs and try again.")
    elif urls_input and 'keyword_embedding' not in locals():
        st.info("Please enter a keyword in the first tab before analyzing URLs.")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and Google's Universal Sentence Encoder") 
