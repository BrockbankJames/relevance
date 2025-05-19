import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import json
import pandas as pd
from google.cloud import aiplatform
from google.oauth2 import service_account
import google.auth

# Set page config
st.set_page_config(
    page_title="Google Vertex AI Embedding Generator",
    page_icon="🔤",
    layout="centered"
)

# Add debug mode
DEBUG = True

def debug_log(message):
    """Print debug messages if DEBUG is True"""
    if DEBUG:
        st.write(f"Debug: {message}")

# Add title and description
st.title("Google Vertex AI Embedding Generator")
st.markdown("""
This app generates vector embeddings for text and webpages using Google's Vertex AI text-embedding-005 model.
You can either enter a keyword or a webpage URL to analyze content similarity.
""")

# Initialize Vertex AI
@st.cache_resource
def init_vertex_ai():
    try:
        debug_log("Starting Vertex AI initialization...")
        
        # Try to get credentials from Streamlit secrets
        if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in st.secrets:
            debug_log("Found credentials in Streamlit secrets")
            try:
                # Create credentials from JSON string
                credentials_info = json.loads(st.secrets['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
                debug_log("Successfully parsed credentials JSON")
                credentials = service_account.Credentials.from_service_account_info(credentials_info)
                debug_log("Successfully created credentials object")
            except json.JSONDecodeError as e:
                st.error(f"Error parsing credentials JSON: {str(e)}")
                return False
            except Exception as e:
                st.error(f"Error creating credentials: {str(e)}")
                return False
        else:
            debug_log("No credentials found in Streamlit secrets, trying default credentials")
            try:
                credentials, project = google.auth.default()
                debug_log("Successfully got default credentials")
            except Exception as e:
                st.error(f"Error getting default credentials: {str(e)}")
                return False
        
        # Get project and location
        project_id = st.secrets.get('GOOGLE_CLOUD_PROJECT')
        location = st.secrets.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
        debug_log(f"Using project: {project_id}, location: {location}")
        
        # Initialize Vertex AI
        debug_log("Initializing Vertex AI...")
        aiplatform.init(
            credentials=credentials,
            project=project_id,
            location=location
        )
        debug_log("Vertex AI initialization complete!")
        return True
    except Exception as e:
        st.error(f"Error initializing Vertex AI: {str(e)}")
        st.info("""
        Please make sure you have set up the following in your Streamlit secrets:
        1. GOOGLE_APPLICATION_CREDENTIALS_JSON: Your service account key JSON
        2. GOOGLE_CLOUD_PROJECT: Your Google Cloud project ID
        3. GOOGLE_CLOUD_LOCATION: Your preferred location (default: us-central1)
        
        Current secrets status:
        - GOOGLE_APPLICATION_CREDENTIALS_JSON: {'present': 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in st.secrets}
        - GOOGLE_CLOUD_PROJECT: {'present': 'GOOGLE_CLOUD_PROJECT' in st.secrets}
        - GOOGLE_CLOUD_LOCATION: {'present': 'GOOGLE_CLOUD_LOCATION' in st.secrets}
        """)
        return False

# Initialize Vertex AI with spinner
with st.spinner("Initializing Vertex AI..."):
    debug_log("Starting app initialization...")
    if not init_vertex_ai():
        st.error("Failed to initialize Vertex AI. Please check the error messages above.")
        st.stop()
    debug_log("App initialization complete!")

def get_embedding(text):
    """Generate embedding using Vertex AI text-embedding-005 model"""
    try:
        debug_log("Starting embedding generation...")
        debug_log(f"Input text length: {len(text)} characters")
        
        # Get project and location from secrets
        project_id = st.secrets.get('GOOGLE_CLOUD_PROJECT')
        location = st.secrets.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        if not project_id:
            st.error("GOOGLE_CLOUD_PROJECT not found in secrets")
            return None
            
        # Initialize the model
        debug_log("Initializing model...")
        # Use the correct model endpoint with project ID from secrets
        model_endpoint = f"projects/{project_id}/locations/{location}/publishers/google/models/textembedding-gecko@001"
        debug_log(f"Using model endpoint: {model_endpoint}")
        
        model = aiplatform.Model(model_endpoint)
        debug_log("Model initialized successfully")
        
        # Get embeddings
        debug_log("Getting embeddings...")
        response = model.predict([text])
        debug_log("Embeddings generated successfully")
        
        # Extract the embedding from the response
        embedding = np.array(response.predictions[0])
        debug_log(f"Embedding shape: {embedding.shape}")
        return embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        debug_log(f"Error details: {str(e)}")
        return None

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
        debug_log(f"Original URL: {url}")
        debug_log(f"Encoded URL: {encoded_url}")
        
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
        debug_log(f"API URL (without key): {debug_url}")
        
        api_url = "https://app.scrapingbee.com/api/v1/"
        response = requests.get(api_url, params=params)
        
        if response.status_code != 200:
            st.error(f"Error scraping webpage: HTTP {response.status_code}")
            try:
                error_data = response.json()
                if 'error' in error_data:
                    st.error(f"API Error: {error_data['error']}")
                    if 'reason' in error_data:
                        st.error(f"Reason: {error_data['reason']}")
            except:
                debug_log(f"Raw API Response: {response.text}")
                st.error("Failed to parse API error response")
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
        debug_log(f"Scraping error details: {str(e)}")
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
    st.subheader("Analyze Link Profile")
    st.markdown("""
    Enter URLs (one per line) to analyze their similarity to the webpage you analyzed in the "Webpage Analysis" tab.
    The app will show:
    - How similar each URL is to the analyzed webpage
    - Most relevant URLs ranked by similarity
    - Individual URL scores
    """)
    
    # Get URLs input
    urls_input = st.text_area(
        "Enter URLs (one per line):",
        height=150,
        placeholder="https://example1.com\nhttps://example2.com\nhttps://example3.com"
    )
    
    if urls_input:
        # Check if we have a webpage analyzed in tab 2
        if 'url_input' not in locals() or not url_input or 'section_embeddings' not in locals() or not section_embeddings:
            st.warning("Please analyze a webpage in the 'Webpage Analysis' tab first.")
        else:
            # Process URLs
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            
            if urls:
                # Get the current page embedding from tab 2
                current_page_embedding = np.mean(section_embeddings, axis=0)
                
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
                                section_embeddings_url = []
                                for section in sections:
                                    embedding = get_embedding(section['text'])
                                    section_embeddings_url.append(embedding)
                                
                                # Average section embeddings
                                avg_section_embedding = np.mean(section_embeddings_url, axis=0)
                                
                                # Calculate similarity with current page
                                similarity = calculate_similarity(current_page_embedding, avg_section_embedding)
                                
                                # Calculate query similarity if keyword exists
                                query_similarity = None
                                if 'keyword_embedding' in locals():
                                    query_similarity = calculate_similarity(keyword_embedding, avg_section_embedding)
                                
                                url_results.append({
                                    'url': url,
                                    'similarity': similarity,
                                    'query_similarity': query_similarity,
                                    'sections_count': len(sections)
                                })
                        except Exception as e:
                            st.warning(f"Error processing {url}: {str(e)}")
                
                if url_results:
                    # Sort URLs by similarity to current page
                    url_results.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    # Calculate average similarity
                    avg_similarity = np.mean([r['similarity'] for r in url_results])
                    
                    # Display overall results
                    st.subheader("Similarity Results")
                    st.metric(
                        label="Average Similarity to Analyzed Webpage",
                        value=f"{avg_similarity:.3f}",
                        help="Average similarity between the analyzed webpage and all URLs"
                    )
                    
                    # Display top 3 most similar URLs
                    st.subheader("Most Similar URLs")
                    for i, result in enumerate(url_results[:3], 1):
                        st.markdown(f"""
                        **{i}. {result['url']}**  
                        Similarity: {result['similarity']:.3f}  
                        Sections analyzed: {result['sections_count']}
                        """)
                    
                    # Display all URLs in a table
                    st.subheader("All URL Similarities")
                    df = pd.DataFrame(url_results)
                    df['similarity'] = df['similarity'].round(3)
                    if 'query_similarity' in df.columns:
                        df['query_similarity'] = df['query_similarity'].round(3)
                    
                    # Rename columns
                    column_rename = {
                        'url': 'URL',
                        'similarity': 'Similarity to Webpage',
                        'sections_count': 'Sections Analyzed'
                    }
                    if 'query_similarity' in df.columns:
                        column_rename['query_similarity'] = 'Query Similarity'
                    
                    df = df.rename(columns=column_rename)
                    st.dataframe(df, use_container_width=True)
                    
                    # Add download button for results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Analysis Results as CSV",
                        data=csv,
                        file_name="url_analysis.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("No URLs were successfully analyzed. Please check the URLs and try again.")
            else:
                st.warning("Please enter at least one URL to analyze.")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and Google's Vertex AI text-embedding-005 model") 
