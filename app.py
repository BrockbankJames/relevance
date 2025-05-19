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
    Enter URLs (one per line) to analyze their relevance to your keyword and to each other.
    You can also compare a single URL to the page you analyzed in the "Webpage Analysis" tab.
    The app will show:
    - Query relevance: How similar each URL is to your keyword
    - URL similarity: How similar the URLs are to each other and to the analyzed page
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
        # Process URLs
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        
        if urls:
            # Store results for each URL
            url_results = []
            url_embeddings = {}  # Store embeddings for URL-to-URL comparison
            
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
                            url_embeddings[url] = avg_section_embedding
                            
                            # Calculate query similarity if keyword exists
                            query_similarity = None
                            if 'keyword_embedding' in locals():
                                query_similarity = calculate_similarity(keyword_embedding, avg_section_embedding)
                            
                            url_results.append({
                                'url': url,
                                'query_similarity': query_similarity,
                                'sections_count': len(sections)
                            })
                    except Exception as e:
                        st.warning(f"Error processing {url}: {str(e)}")
            
            if url_results:
                # Calculate URL-to-URL similarities
                url_similarities = []
                
                # If we have a current page from tab 2, add it to comparisons
                current_page_url = None
                if 'url_input' in locals() and url_input and 'section_embeddings' in locals() and section_embeddings:
                    current_page_url = "Current Page"
                    current_page_embedding = np.mean(section_embeddings, axis=0)
                    url_embeddings[current_page_url] = current_page_embedding
                
                # Compare URLs with each other and with current page
                for i, url1 in enumerate(urls):
                    # Compare with current page if available
                    if current_page_url:
                        similarity = calculate_similarity(url_embeddings[url1], url_embeddings[current_page_url])
                        url_similarities.append({
                            'url1': url1,
                            'url2': current_page_url,
                            'similarity': similarity
                        })
                    
                    # Compare with other URLs
                    for url2 in urls[i+1:]:
                        if url1 in url_embeddings and url2 in url_embeddings:
                            similarity = calculate_similarity(url_embeddings[url1], url_embeddings[url2])
                            url_similarities.append({
                                'url1': url1,
                                'url2': url2,
                                'similarity': similarity
                            })
                
                # Create tabs for different views
                query_tab, url_tab = st.tabs(["Query Relevance", "URL Similarity"])
                
                with query_tab:
                    if 'keyword_embedding' in locals():
                        # Sort URLs by query similarity
                        url_results.sort(key=lambda x: x['query_similarity'] if x['query_similarity'] is not None else -1, reverse=True)
                        
                        # Calculate average query similarity
                        query_similarities = [r['query_similarity'] for r in url_results if r['query_similarity'] is not None]
                        if query_similarities:
                            avg_query_similarity = np.mean(query_similarities)
                            
                            # Display overall results
                            st.subheader("Query Relevance Results")
                            st.metric(
                                label="Average Query Relevance",
                                value=f"{avg_query_similarity:.3f}",
                                help="Average similarity to your keyword across all analyzed URLs"
                            )
                            
                            # Display top 3 most relevant URLs
                            st.subheader("Most Relevant URLs to Query")
                            for i, result in enumerate(url_results[:3], 1):
                                if result['query_similarity'] is not None:
                                    st.markdown(f"""
                                    **{i}. {result['url']}**  
                                    Query Similarity: {result['query_similarity']:.3f}  
                                    Sections analyzed: {result['sections_count']}
                                    """)
                            
                            # Display all URLs in a table
                            st.subheader("All URLs Query Relevance")
                            import pandas as pd
                            df_query = pd.DataFrame(url_results)
                            df_query['query_similarity'] = df_query['query_similarity'].round(3)
                            df_query = df_query.rename(columns={
                                'url': 'URL',
                                'query_similarity': 'Query Similarity Score',
                                'sections_count': 'Sections Analyzed'
                            })
                            st.dataframe(df_query, use_container_width=True)
                            
                            # Add download button for query results
                            csv_query = df_query.to_csv(index=False)
                            st.download_button(
                                label="Download Query Results as CSV",
                                data=csv_query,
                                file_name="url_query_analysis.csv",
                                mime="text/csv"
                            )
                    else:
                        st.info("Enter a keyword in the first tab to see query relevance analysis.")
                
                with url_tab:
                    if url_similarities:
                        # Sort URL similarities
                        url_similarities.sort(key=lambda x: x['similarity'], reverse=True)
                        
                        # Calculate average URL similarity
                        avg_url_similarity = np.mean([s['similarity'] for s in url_similarities])
                        
                        # Display overall results
                        st.subheader("URL Similarity Results")
                        st.metric(
                            label="Average URL Similarity",
                            value=f"{avg_url_similarity:.3f}",
                            help="Average similarity between URLs across all pairs"
                        )
                        
                        # Display top 3 most similar URL pairs
                        st.subheader("Most Similar URL Pairs")
                        for i, pair in enumerate(url_similarities[:3], 1):
                            st.markdown(f"""
                            **{i}. {pair['url1']}**  
                            **   {pair['url2']}**  
                            Similarity: {pair['similarity']:.3f}
                            """)
                        
                        # If we have current page comparisons, show them separately
                        if current_page_url:
                            st.subheader("Similarity to Current Page")
                            current_page_similarities = [s for s in url_similarities if s['url2'] == current_page_url]
                            current_page_similarities.sort(key=lambda x: x['similarity'], reverse=True)
                            
                            for similarity in current_page_similarities:
                                st.markdown(f"""
                                **URL: {similarity['url1']}**  
                                Similarity to Current Page: {similarity['similarity']:.3f}
                                """)
                        
                        # Display all URL pairs in a table
                        st.subheader("All URL Pair Similarities")
                        df_url = pd.DataFrame(url_similarities)
                        df_url['similarity'] = df_url['similarity'].round(3)
                        df_url = df_url.rename(columns={
                            'url1': 'URL 1',
                            'url2': 'URL 2',
                            'similarity': 'Similarity Score'
                        })
                        st.dataframe(df_url, use_container_width=True)
                        
                        # Add download button for URL similarity results
                        csv_url = df_url.to_csv(index=False)
                        st.download_button(
                            label="Download URL Similarity Results as CSV",
                            data=csv_url,
                            file_name="url_similarity_analysis.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Not enough URLs were successfully analyzed to calculate URL similarities.")
            else:
                st.error("No URLs were successfully analyzed. Please check the URLs and try again.")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and Google's Universal Sentence Encoder") 
