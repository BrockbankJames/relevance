import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import json
import pandas as pd
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import hashlib
import time

# Set page config
st.set_page_config(
    page_title="Google Vertex AI Embedding Generator",
    page_icon="🔤",
    layout="centered"
)

# Add title and description
st.title("Google Vertex AI Embedding Generator")
st.markdown("""
This app generates vector embeddings for text and webpages using Google's Vertex AI text-embedding-005 model.
You can either enter a keyword or a webpage URL to analyze content similarity.
""")

# Add debug mode
DEBUG = True

# Add supported regions
SUPPORTED_REGIONS = {
    'asia-east1', 'europe-west8', 'europe-west3', 'europe-west2', 'europe-west9',
    'europe-central2', 'us-central1', 'asia-northeast3', 'us-east4', 'australia-southeast2',
    'europe-west1', 'us-west3', 'asia-east2', 'southamerica-west1', 'europe-north1',
    'us-east1', 'me-west1', 'asia-southeast1', 'us-west1', 'australia-southeast1',
    'northamerica-northeast1', 'asia-south1', 'asia-northeast1', 'europe-west6',
    'europe-west4', 'northamerica-northeast2', 'southamerica-east1', 'us-west4',
    'asia-northeast2', 'asia-southeast2', 'us-south1', 'us-west2', 'europe-southwest1'
}

def debug_log(message):
    """Print debug messages if DEBUG is True"""
    if DEBUG:
        st.write(f"Debug: {message}")

# Cache for embeddings to reduce API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_embedding(text):
    """Get embedding from cache or generate new one"""
    # Create a hash of the text for caching
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Check if we have a cached embedding
    if 'embeddings_cache' not in st.session_state:
        st.session_state.embeddings_cache = {}
    
    if text_hash in st.session_state.embeddings_cache:
        debug_log("Using cached embedding")
        return st.session_state.embeddings_cache[text_hash]
    
    # If not in cache, generate new embedding
    embedding = get_embedding(text)
    if embedding is not None:
        st.session_state.embeddings_cache[text_hash] = embedding
    return embedding

def get_vertex_ai_token():
    """Get authentication token for Vertex AI API"""
    try:
        debug_log("Getting authentication token...")
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(st.secrets['GOOGLE_APPLICATION_CREDENTIALS_JSON']),
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        credentials.refresh(Request())
        token = credentials.token
        debug_log("Successfully obtained authentication token")
        return token
    except Exception as e:
        st.error(f"Error getting authentication token: {str(e)}")
        debug_log(f"Token error details: {str(e)}")
        return None

def get_embedding(text):
    """Generate embedding using Vertex AI text-embedding-005 model via REST API"""
    if not text or not isinstance(text, str):
        st.error("Invalid input: text must be a non-empty string")
        return None
        
    text = text.strip()
    if not text:
        st.error("Invalid input: text cannot be empty or whitespace only")
        return None
        
    try:
        debug_log("Starting embedding generation...")
        debug_log(f"Input text length: {len(text)} characters")
        
        # Get project and location from secrets
        project_id = st.secrets.get('GOOGLE_CLOUD_PROJECT')
        location = st.secrets.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        if not project_id:
            st.error("GOOGLE_CLOUD_PROJECT not found in secrets")
            return None
            
        # Get authentication token
        token = get_vertex_ai_token()
        if not token:
            return None
            
        # Prepare the API request
        endpoint = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/textembedding-gecko:predict"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "instances": [
                {"content": text}
            ]
        }
        
        debug_log(f"Making request to: {endpoint}")
        
        # Make the API request with retry logic for quota limits
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(endpoint, headers=headers, json=data)
                
                if response.status_code == 429:  # Quota exceeded
                    if attempt < max_retries - 1:
                        debug_log(f"Quota exceeded, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        st.error("""
                        Quota exceeded for Vertex AI. Please try again later or request a quota increase.
                        You can request a quota increase here: https://cloud.google.com/vertex-ai/docs/generative-ai/quotas-genai
                        """)
                        return None
                
                if response.status_code != 200:
                    st.error(f"Error from Vertex AI API: {response.status_code}")
                    debug_log(f"API Response: {response.text}")
                    return None
                    
                # Parse the response
                result = response.json()
                debug_log("Successfully received API response")
                
                if 'predictions' not in result or not result['predictions']:
                    st.error("No predictions in API response")
                    debug_log(f"API Response: {result}")
                    return None
                    
                # Extract the embedding
                embedding = np.array(result['predictions'][0]['embeddings']['values'])
                debug_log(f"Successfully extracted embedding, shape: {embedding.shape}")
                
                return embedding
                
            except Exception as e:
                if attempt < max_retries - 1:
                    debug_log(f"Request failed, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e
        
    except Exception as e:
        st.error(f"Error in get_embedding: {str(e)}")
        debug_log(f"Full error details: {str(e)}")
        return None

def extract_sections(html_content):
    """Extract content sections from HTML, properly handling nested tags"""
    soup = BeautifulSoup(html_content, 'html.parser')
    sections = []
    
    # Find all heading tags
    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        section = {
            'type': heading.name,
            'text': '',
            'content': []
        }
        
        # Get the heading text
        heading_text = heading.get_text().strip()
        if heading_text:
            section['text'] = heading_text
        
        # Get all content until the next heading of same or higher level
        current = heading.next_sibling
        while current and not (hasattr(current, 'name') and 
                             current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and 
                             int(current.name[1]) <= int(heading.name[1])):
            if hasattr(current, 'name') and current.name == 'p':
                text = current.get_text().strip()
                if text:
                    section['content'].append(text)
            current = current.next_sibling
        
        # Combine heading and content
        if section['content']:
            section['text'] = f"{section['text']} {' '.join(section['content'])}"
        
        if section['text']:  # Only add non-empty sections
            sections.append(section)
    
    # If no sections found, try to get paragraphs
    if not sections:
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if text:
                sections.append({
                    'type': 'p',
                    'text': text
                })
    
    debug_log(f"Extracted {len(sections)} sections from HTML")
    for i, section in enumerate(sections):
        debug_log(f"Section {i+1} ({section['type']}): {section['text'][:100]}...")
    
    return sections

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
        debug_log(f"Scraping URL: {url}")
        
        # Make the request to ScrapingBee API with enhanced parameters
        params = {
            'api_key': api_key,
            'url': encoded_url,
            'render_js': 'true',  # Enable JavaScript rendering
            'premium_proxy': 'true',  # Use premium proxies
            'wait': '5000',  # Wait 5 seconds for JavaScript
            'wait_for': 'body',  # Wait for body element
            'block_resources': 'true',  # Block unnecessary resources
            'extract_rules': json.dumps({  # Extract specific elements
                'headings': {
                    'selector': 'h1, h2, h3, h4, h5, h6',
                    'type': 'list',
                    'output': 'text'
                },
                'paragraphs': {
                    'selector': 'p',
                    'type': 'list',
                    'output': 'text'
                }
            })
        }
        
        debug_log("Making request to ScrapingBee API...")
        api_url = "https://app.scrapingbee.com/api/v1/"
        response = requests.get(api_url, params=params)
        
        if response.status_code != 200:
            st.error(f"Error scraping webpage: HTTP {response.status_code}")
            try:
                error_data = response.json()
                if 'error' in error_data:
                    st.error(f"ScrapingBee API Error: {error_data['error']}")
                    if 'message' in error_data:
                        st.error(f"Details: {error_data['message']}")
            except:
                debug_log(f"Raw ScrapingBee Response: {response.text}")
            return None
        
        debug_log("Successfully received response from ScrapingBee")
        
        # Extract sections using the new function
        sections = extract_sections(response.content)
        
        if not sections:
            st.warning("""
            No content sections found on the page. This could be because:
            1. The page is blocking access
            2. The page has no text content
            3. The page structure is different than expected
            
            Try using a different URL or check if the page is accessible.
            """)
            debug_log("Raw HTML content received:")
            debug_log(response.text[:500] + "...")  # Log first 500 chars of HTML
        
        return sections
    except Exception as e:
        st.error(f"Error scraping webpage: {str(e)}")
        debug_log(f"Scraping error details: {str(e)}")
        debug_log(f"Error type: {type(e)}")
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
            keyword_embedding = get_cached_embedding(keyword_input)
            
        if keyword_embedding is not None:
            # Display embedding information
            st.subheader("Embedding Information")
            st.write(f"Vector dimension: {keyword_embedding.shape[0]}")
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
        else:
            st.error("Failed to generate embedding. Please check the error messages above.")

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
                    embedding = get_cached_embedding(section['text'])
                    if embedding is not None:
                        section_embeddings.append(embedding)
            
            if section_embeddings:
                # If we have a keyword embedding from tab1, calculate similarity
                if 'keyword_embedding' in locals() and keyword_embedding is not None:
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
            else:
                st.error("Failed to generate embeddings for any sections. Please check the error messages above.")
        else:
            st.error("Failed to scrape webpage content. Please check the URL and try again.")

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
        if 'url_input' not in locals() or not url_input:
            st.warning("Please analyze a webpage in the 'Webpage Analysis' tab first.")
        elif 'section_embeddings' not in locals() or not section_embeddings:
            st.warning("No valid embeddings found for the analyzed webpage. Please try analyzing the webpage again.")
        else:
            # Process URLs
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            
            if not urls:
                st.warning("Please enter at least one valid URL to analyze.")
            else:
                # Get the current page embedding from tab 2
                try:
                    current_page_embedding = np.mean(section_embeddings, axis=0)
                    if current_page_embedding is None or current_page_embedding.size == 0:
                        st.error("Failed to calculate average embedding for the current page.")
                        st.stop()
                except Exception as e:
                    st.error(f"Error calculating average embedding: {str(e)}")
                    st.stop()
                
                # Store results for each URL
                url_results = []
                
                # Process each URL
                with st.spinner(f"Analyzing {len(urls)} URLs..."):
                    for url in urls:
                        try:
                            # Scrape and analyze the URL
                            sections = scrape_webpage(url)
                            if not sections:
                                st.warning(f"Could not scrape content from {url}")
                                continue
                                
                            # Generate embeddings for all sections
                            section_embeddings_url = []
                            for section in sections:
                                embedding = get_cached_embedding(section['text'])
                                if embedding is not None:
                                    section_embeddings_url.append(embedding)
                            
                            if not section_embeddings_url:
                                st.warning(f"Could not generate embeddings for {url}")
                                continue
                                
                            try:
                                # Average section embeddings
                                avg_section_embedding = np.mean(section_embeddings_url, axis=0)
                                if avg_section_embedding is None or avg_section_embedding.size == 0:
                                    st.warning(f"Failed to calculate average embedding for {url}")
                                    continue
                                    
                                # Calculate similarity with current page
                                similarity = calculate_similarity(current_page_embedding, avg_section_embedding)
                                
                                # Calculate query similarity if keyword exists
                                query_similarity = None
                                if 'keyword_embedding' in locals() and keyword_embedding is not None:
                                    query_similarity = calculate_similarity(keyword_embedding, avg_section_embedding)
                                
                                url_results.append({
                                    'url': url,
                                    'similarity': similarity,
                                    'query_similarity': query_similarity,
                                    'sections_count': len(sections)
                                })
                            except Exception as e:
                                st.warning(f"Error processing embeddings for {url}: {str(e)}")
                                continue
                                
                        except Exception as e:
                            st.warning(f"Error processing {url}: {str(e)}")
                            continue
                
                if url_results:
                    # Sort URLs by similarity to current page
                    url_results.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    # Calculate average similarity
                    try:
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
                    except Exception as e:
                        st.error(f"Error calculating or displaying results: {str(e)}")
                else:
                    st.error("No URLs were successfully analyzed. Please check the URLs and try again.")
    else:
        st.warning("Please enter at least one URL to analyze.")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and Google's Vertex AI text-embedding-005 model") 
