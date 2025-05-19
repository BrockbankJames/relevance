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

def get_embedding(texts, batch_size=250):
    """Generate embeddings using Vertex AI text-embedding-005 model via REST API"""
    if isinstance(texts, str):
        texts = [texts]  # Convert single text to list
        
    if not texts or not all(isinstance(text, str) for text in texts):
        st.error("Invalid input: all texts must be non-empty strings")
        return None
        
    # Clean and validate texts
    texts = [text.strip() for text in texts if text.strip()]
    if not texts:
        st.error("Invalid input: no valid texts provided")
        return None
        
    try:
        debug_log("Starting batch embedding generation...")
        debug_log(f"Number of texts to process: {len(texts)}")
        
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
            
        # Prepare the API endpoint
        endpoint = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/textembedding-gecko:predict"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Process texts in batches
        all_embeddings = []
        current_batch = []
        current_batch_tokens = 0
        
        # Rough estimate: 1 token ≈ 4 characters for English text
        # This is a conservative estimate to stay well under the 20,000 token limit
        TOKENS_PER_CHAR = 0.25
        MAX_TOKENS_PER_TEXT = 8000  # Slightly under 8,192 to be safe
        MAX_TOKENS_PER_BATCH = 19000  # Slightly under 20,000 to be safe
        
        for text in texts:
            # Truncate text if it's too long
            if len(text) * TOKENS_PER_CHAR > MAX_TOKENS_PER_TEXT:
                debug_log(f"Truncating text from {len(text)} to {int(MAX_TOKENS_PER_TEXT / TOKENS_PER_CHAR)} characters")
                text = text[:int(MAX_TOKENS_PER_TEXT / TOKENS_PER_CHAR)]
            
            text_tokens = len(text) * TOKENS_PER_CHAR
            
            # If adding this text would exceed batch limits, process current batch
            if (current_batch_tokens + text_tokens > MAX_TOKENS_PER_BATCH or 
                len(current_batch) >= batch_size):
                if current_batch:
                    debug_log(f"\nProcessing batch of {len(current_batch)} texts")
                    debug_log(f"Batch token estimate: {current_batch_tokens:.0f}")
                    
                    data = {
                        "instances": [{"content": text} for text in current_batch]
                    }
                    
                    # Make the API request with retry logic for quota limits
                    max_retries = 3
                    retry_delay = 5  # Start with 5 second delay
                    
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
                            
                            # Extract and validate embeddings
                            batch_embeddings = []
                            for pred in result['predictions']:
                                if 'embeddings' not in pred or 'values' not in pred['embeddings']:
                                    st.error("Invalid prediction format in API response")
                                    debug_log(f"Prediction structure: {pred}")
                                    return None
                                    
                                values = pred['embeddings']['values']
                                if not isinstance(values, list):
                                    st.error("Embedding values is not a list")
                                    debug_log(f"Values type: {type(values)}")
                                    return None
                                    
                                # Convert to numpy array and validate
                                embedding = np.array(values, dtype=np.float32)
                                if embedding.ndim != 1:
                                    st.error(f"Invalid embedding dimension: {embedding.ndim}")
                                    debug_log(f"Embedding shape: {embedding.shape}")
                                    return None
                                    
                                batch_embeddings.append(embedding)
                            
                            all_embeddings.extend(batch_embeddings)
                            debug_log(f"Successfully extracted {len(batch_embeddings)} embeddings")
                            
                            # Add a delay between batches to avoid rate limits
                            time.sleep(2)  # 2 second delay between batches
                            
                            break  # Success, exit retry loop
                            
                        except Exception as e:
                            if attempt < max_retries - 1:
                                debug_log(f"Request failed, retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                retry_delay *= 2
                            else:
                                raise e
                    
                    # Reset batch
                    current_batch = []
                    current_batch_tokens = 0
            
            # Add text to current batch
            current_batch.append(text)
            current_batch_tokens += text_tokens
        
        # Process any remaining texts
        if current_batch:
            debug_log(f"\nProcessing final batch of {len(current_batch)} texts")
            debug_log(f"Batch token estimate: {current_batch_tokens:.0f}")
            
            data = {
                "instances": [{"content": text} for text in current_batch]
            }
            
            # Make the API request with retry logic
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(endpoint, headers=headers, json=data)
                    
                    if response.status_code == 429:  # Quota exceeded
                        if attempt < max_retries - 1:
                            debug_log(f"Quota exceeded, retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
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
                    
                    if 'predictions' not in result or not result['predictions']:
                        st.error("No predictions in API response")
                        debug_log(f"API Response: {result}")
                        return None
                    
                    # Extract and validate embeddings
                    batch_embeddings = []
                    for pred in result['predictions']:
                        if 'embeddings' not in pred or 'values' not in pred['embeddings']:
                            st.error("Invalid prediction format in API response")
                            debug_log(f"Prediction structure: {pred}")
                            return None
                            
                        values = pred['embeddings']['values']
                        if not isinstance(values, list):
                            st.error("Embedding values is not a list")
                            debug_log(f"Values type: {type(values)}")
                            return None
                            
                        # Convert to numpy array and validate
                        embedding = np.array(values, dtype=np.float32)
                        if embedding.ndim != 1:
                            st.error(f"Invalid embedding dimension: {embedding.ndim}")
                            debug_log(f"Embedding shape: {embedding.shape}")
                            return None
                            
                        batch_embeddings.append(embedding)
                    
                    all_embeddings.extend(batch_embeddings)
                    debug_log(f"Successfully extracted {len(batch_embeddings)} embeddings")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        debug_log(f"Request failed, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise e
        
        if len(all_embeddings) != len(texts):
            st.error(f"Expected {len(texts)} embeddings but got {len(all_embeddings)}")
            return None
            
        debug_log(f"Successfully generated all {len(all_embeddings)} embeddings")
        # Return single embedding for single text input, list for multiple texts
        return all_embeddings[0] if len(texts) == 1 else all_embeddings
        
    except Exception as e:
        st.error(f"Error in get_embedding: {str(e)}")
        debug_log(f"Full error details: {str(e)}")
        return None

def get_cached_embedding(texts, batch_size=250):
    """Get embeddings from cache or generate new ones, maintaining section structure"""
    if isinstance(texts, str):
        texts = [texts]  # Convert single text to list
    
    # If texts is a list of sections (from webpage analysis)
    is_section_list = all(isinstance(text, dict) and 'text' in text for text in texts)
    
    if is_section_list:
        # Extract just the text content for embedding generation
        section_texts = [section['text'] for section in texts]
        # Create hashes for caching using the full section text
        text_hashes = [hashlib.md5(section['text'].encode()).hexdigest() for section in texts]
    else:
        section_texts = texts
        # Create hashes for caching
        text_hashes = [hashlib.md5(text.encode()).hexdigest() for text in texts]
    
    # Check if we have cached embeddings
    if 'embeddings_cache' not in st.session_state:
        st.session_state.embeddings_cache = {}
    
    # Check which texts need new embeddings
    texts_to_generate = []
    indices_to_generate = []
    cached_embeddings = []
    
    for i, (text, text_hash) in enumerate(zip(section_texts, text_hashes)):
        if text_hash in st.session_state.embeddings_cache:
            cached_embedding = st.session_state.embeddings_cache[text_hash]
            # Ensure cached embedding is a numpy array
            if isinstance(cached_embedding, (np.ndarray, list, tuple)):
                if isinstance(cached_embedding, np.ndarray):
                    if cached_embedding.ndim == 1:
                        debug_log(f"Using cached embedding for text {i+1}, shape: {cached_embedding.shape}")
                        cached_embeddings.append(cached_embedding)
                        continue
                    else:
                        debug_log(f"Invalid cached embedding dimension for text {i+1}, regenerating")
                else:
                    # Convert list/tuple to numpy array
                    try:
                        cached_embedding = np.array(cached_embedding, dtype=np.float32)
                        if cached_embedding.ndim == 1:
                            debug_log(f"Converted cached embedding to numpy array for text {i+1}, shape: {cached_embedding.shape}")
                            cached_embeddings.append(cached_embedding)
                            st.session_state.embeddings_cache[text_hash] = cached_embedding
                            continue
                        else:
                            debug_log(f"Invalid converted embedding dimension for text {i+1}, regenerating")
                    except Exception as e:
                        debug_log(f"Error converting cached embedding for text {i+1}: {str(e)}")
            else:
                debug_log(f"Invalid cached embedding type for text {i+1}: {type(cached_embedding)}")
            
            # If we get here, the cached embedding was invalid
            texts_to_generate.append(text)
            indices_to_generate.append(i)
        else:
            texts_to_generate.append(text)
            indices_to_generate.append(i)
    
    if texts_to_generate:
        debug_log(f"Generating new embeddings for {len(texts_to_generate)} texts")
        new_embeddings = get_embedding(texts_to_generate, batch_size)
        
        if new_embeddings is not None:
            # Ensure new embeddings are numpy arrays
            if isinstance(new_embeddings, list):
                for i, embedding in enumerate(new_embeddings):
                    if not isinstance(embedding, np.ndarray):
                        try:
                            embedding = np.array(embedding, dtype=np.float32)
                        except Exception as e:
                            debug_log(f"Error converting new embedding to numpy array: {str(e)}")
                            continue
                    if embedding.ndim != 1:
                        debug_log(f"Invalid new embedding dimension: {embedding.ndim}")
                        continue
                    st.session_state.embeddings_cache[text_hashes[indices_to_generate[i]]] = embedding
                    cached_embeddings.append(embedding)
            else:
                # Single embedding case
                if not isinstance(new_embeddings, np.ndarray):
                    try:
                        new_embeddings = np.array(new_embeddings, dtype=np.float32)
                    except Exception as e:
                        debug_log(f"Error converting single new embedding to numpy array: {str(e)}")
                        return None
                if new_embeddings.ndim != 1:
                    debug_log(f"Invalid single new embedding dimension: {new_embeddings.ndim}")
                    return None
                st.session_state.embeddings_cache[text_hashes[indices_to_generate[0]]] = new_embeddings
                cached_embeddings.append(new_embeddings)
            
            debug_log("Successfully cached new embeddings")
    
    if not cached_embeddings:
        debug_log("No valid embeddings generated or cached")
        return None
    
    # If input was a list of sections, return embeddings with section structure
    if is_section_list:
        for i, section in enumerate(texts):
            section['embedding'] = cached_embeddings[i]
        return texts
    else:
        # Return single embedding for single text input, list for multiple texts
        result = cached_embeddings[0] if len(texts) == 1 else cached_embeddings
        debug_log(f"Returning embeddings of type: {type(result)}")
        if isinstance(result, np.ndarray):
            debug_log(f"Returning numpy array with shape: {result.shape}")
        elif isinstance(result, list):
            debug_log(f"Returning list of {len(result)} embeddings")
            for i, emb in enumerate(result):
                debug_log(f"Embedding {i+1} type: {type(emb)}, shape: {emb.shape if hasattr(emb, 'shape') else 'no shape'}")
        return result

def extract_sections(html_content):
    """Extract content sections from HTML, properly handling nested tags"""
    debug_log("Starting HTML content extraction...")
    debug_log(f"HTML content length: {len(html_content)} characters")
    debug_log("First 500 characters of HTML:")
    debug_log(html_content[:500])
    
    soup = BeautifulSoup(html_content, 'html.parser')
    sections = []
    
    # Debug the HTML structure
    debug_log("\nChecking for headings...")
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    debug_log(f"Found {len(headings)} heading tags")
    for h in headings:
        debug_log(f"Found {h.name} tag with text: {h.get_text().strip()[:100]}")
    
    debug_log("\nChecking for paragraphs...")
    paragraphs = soup.find_all('p')
    debug_log(f"Found {len(paragraphs)} paragraph tags")
    for p in paragraphs[:3]:  # Show first 3 paragraphs
        debug_log(f"Paragraph text: {p.get_text().strip()[:100]}")
    
    # Find all heading tags
    for heading in headings:
        section = {
            'type': heading.name,
            'text': '',
            'content': []
        }
        
        # Get the heading text
        heading_text = heading.get_text().strip()
        if heading_text:
            section['text'] = heading_text
            debug_log(f"\nProcessing {heading.name} section: {heading_text[:100]}")
        
        # Get all content until the next heading of same or higher level
        current = heading.next_sibling
        content_count = 0
        while current and not (hasattr(current, 'name') and 
                             current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and 
                             int(current.name[1]) <= int(heading.name[1])):
            if hasattr(current, 'name') and current.name == 'p':
                text = current.get_text().strip()
                if text:
                    section['content'].append(text)
                    content_count += 1
                    if content_count <= 2:  # Log first 2 content items
                        debug_log(f"  Added content: {text[:100]}")
            current = current.next_sibling
        
        # Combine heading and content
        if section['content']:
            section['text'] = f"{section['text']} {' '.join(section['content'])}"
            debug_log(f"  Combined section text: {section['text'][:100]}")
        
        if section['text']:  # Only add non-empty sections
            sections.append(section)
            debug_log(f"  Added section with {len(section['content'])} content items")
    
    # If no sections found, try to get paragraphs
    if not sections:
        debug_log("\nNo sections found with headings, trying paragraphs only...")
        for p in paragraphs:
            text = p.get_text().strip()
            if text:
                sections.append({
                    'type': 'p',
                    'text': text
                })
                debug_log(f"Added paragraph: {text[:100]}")
    
    debug_log(f"\nExtracted {len(sections)} total sections")
    for i, section in enumerate(sections):
        debug_log(f"Section {i+1} ({section['type']}): {section['text'][:100]}...")
    
    return sections

def extract_sections_from_json(json_data):
    """Extract sections from ScrapingBee JSON response, grouping content by H2 tags"""
    debug_log("Processing JSON response...")
    sections = []
    
    try:
        data = json.loads(json_data)
        debug_log(f"JSON keys found: {list(data.keys())}")
        
        # First, get all headings and paragraphs from the main content HTML
        headings = []
        paragraphs = []
        
        # Process main content HTML first to get proper heading levels
        if 'main_content' in data and data['main_content']:
            debug_log("Processing main content HTML...")
            debug_log(f"Number of main_content items: {len(data['main_content'])}")
            
            for i, content_html in enumerate(data['main_content']):
                if not content_html:
                    debug_log(f"Skipping empty content_html at index {i}")
                    continue
                    
                debug_log(f"\nProcessing content_html {i+1}:")
                debug_log(f"Content length: {len(content_html)} characters")
                debug_log(f"First 200 characters: {content_html[:200]}")
                
                try:
                    soup = BeautifulSoup(content_html, 'html.parser')
                    
                    # First, remove all header, footer, nav, and other non-content elements
                    # This is done before any content extraction to ensure we only process main content
                    for tag in soup.find_all(['header', 'footer', 'nav', 'aside']):
                        debug_log(f"Removing {tag.name} element")
                        tag.decompose()
                    
                    # Also remove any elements that are children of header/footer/nav
                    for parent in soup.find_all(['header', 'footer', 'nav']):
                        for child in parent.find_all():
                            debug_log(f"Removing child element {child.name} from {parent.name}")
                            child.decompose()
                    
                    # Remove elements with common non-content classes
                    non_content_classes = [
                        'header', 'footer', 'nav', 'menu', 'sidebar', 'widget', 'advertisement',
                        'banner', 'promo', 'cookie-notice', 'popup', 'modal', 'overlay',
                        'newsletter', 'subscription', 'social-share', 'related-posts',
                        'comments', 'recommendations', 'trending', 'popular', 'featured',
                        'latest', 'recent', 'preview', 'list-item', 'card', 'teaser',
                        'summary', 'excerpt', 'snippet', 'thumbnail', 'image', 'video',
                        'gallery', 'carousel', 'slider', 'tabs', 'accordion', 'dropdown',
                        'tooltip', 'notification', 'alert', 'message', 'status', 'progress',
                        'loading', 'spinner', 'icon', 'button', 'link', 'badge', 'label',
                        'tag', 'category', 'meta', 'author', 'date', 'time', 'location',
                        'price', 'rating', 'review', 'comment', 'share', 'like', 'follow',
                        'subscribe', 'search', 'filter', 'sort', 'pagination', 'breadcrumb',
                        'sitemap'
                    ]
                    
                    # Remove elements with non-content classes
                    for tag in soup.find_all(class_=lambda x: x and any(cls in str(x).lower() for cls in non_content_classes)):
                        debug_log(f"Removing element with non-content class: {tag.get('class', [])}")
                        tag.decompose()
                    
                    # Remove elements that are children of non-content elements
                    for parent in soup.find_all(class_=lambda x: x and any(cls in str(x).lower() for cls in non_content_classes)):
                        for child in parent.find_all():
                            debug_log(f"Removing child element {child.name} from non-content parent")
                            child.decompose()
                    
                    # Now extract headings only from the cleaned content
                    for tag in soup.find_all(['h1', 'h2', 'h3']):
                        # Skip if the heading is inside a non-content element
                        if any(parent.name in ['header', 'footer', 'nav'] or 
                              (parent.get('class') and any(cls in str(parent.get('class')).lower() for cls in non_content_classes))
                              for parent in tag.parents):
                            debug_log(f"Skipping heading in non-content element: {tag.get_text().strip()[:100]}")
                            continue
                            
                        heading_text = tag.get_text().strip()
                        if heading_text:
                            level = int(tag.name[1])  # h1 -> 1, h2 -> 2, etc.
                            headings.append({
                                'text': heading_text,
                                'level': level,
                                'html_tag': tag.name,
                                'position': len(headings),
                                'parent_tag': tag.parent.name if tag.parent else None,
                                'parent_class': tag.parent.get('class', []) if tag.parent else None
                            })
                            debug_log(f"Added heading from HTML ({tag.name}): {heading_text[:100]}")
                            debug_log(f"  Parent tag: {tag.parent.name if tag.parent else 'None'}")
                            debug_log(f"  Parent class: {tag.parent.get('class', []) if tag.parent else 'None'}")
                    
                    # Extract paragraphs only from the cleaned content
                    for p in soup.find_all('p'):
                        # Skip if the paragraph is inside a non-content element
                        if any(parent.name in ['header', 'footer', 'nav'] or 
                              (parent.get('class') and any(cls in str(parent.get('class')).lower() for cls in non_content_classes))
                              for parent in p.parents):
                            debug_log(f"Skipping paragraph in non-content element: {p.get_text().strip()[:100]}")
                            continue
                            
                        paragraph_text = p.get_text().strip()
                        if paragraph_text:
                            paragraphs.append({
                                'text': paragraph_text,
                                'position': len(paragraphs),
                                'parent_tag': p.parent.name if p.parent else None,
                                'parent_class': p.parent.get('class', []) if p.parent else None
                            })
                            debug_log(f"Added paragraph from HTML: {paragraph_text[:100]}")
                            debug_log(f"  Parent tag: {p.parent.name if p.parent else 'None'}")
                            debug_log(f"  Parent class: {p.parent.get('class', []) if p.parent else 'None'}")
                            
                except Exception as e:
                    debug_log(f"Error processing content_html {i+1}: {str(e)}")
                    continue
        
        # If we didn't get any headings from HTML, fall back to the headings list
        if not headings and 'headings' in data and isinstance(data['headings'], list):
            debug_log("No headings found in HTML, using headings list...")
            for heading in data['headings']:
                if heading and isinstance(heading, str):
                    # Skip headings that look like they're from header/footer
                    if any(non_content in heading.lower() for non_content in ['menu', 'navigation', 'header', 'footer', 'copyright']):
                        debug_log(f"Skipping non-content heading: {heading[:100]}")
                        continue
                    headings.append({
                        'text': heading.strip(),
                        'level': 2,  # Default to h2
                        'html_tag': 'h2',
                        'position': len(headings)
                    })
                    debug_log(f"Added heading from list (h2): {heading[:100]}")
        
        # Process paragraphs from JSON if we didn't get enough from HTML
        if len(paragraphs) < 3 and 'paragraphs' in data and isinstance(data['paragraphs'], list):
            debug_log(f"Found {len(data['paragraphs'])} paragraphs in JSON")
            for paragraph in data['paragraphs']:
                if paragraph and isinstance(paragraph, str):
                    # Skip paragraphs that look like they're from header/footer
                    if any(non_content in paragraph.lower() for non_content in ['menu', 'navigation', 'header', 'footer', 'copyright']):
                        debug_log(f"Skipping non-content paragraph: {paragraph[:100]}")
                        continue
                    paragraphs.append({
                        'text': paragraph.strip(),
                        'position': len(paragraphs)
                    })
                    debug_log(f"Added paragraph: {paragraph[:100]}")
        
        # Sort all content by position
        all_content = sorted(headings + paragraphs, key=lambda x: x['position'])
        debug_log(f"\nTotal content items found: {len(all_content)}")
        debug_log(f"Headings: {len(headings)}, Paragraphs: {len(paragraphs)}")
        
        # Group content into sections based on H2 tags
        current_section = None
        current_content = []
        current_h3_section = None
        current_h3_content = []
        
        for item in all_content:
            if 'html_tag' in item:  # This is a heading
                # Skip headings that are part of navigation or previews
                if item.get('parent_class') and any(cls in str(item['parent_class']).lower() 
                                                  for cls in ['nav', 'menu', 'preview', 'list-item']):
                    debug_log(f"Skipping navigation/preview heading: {item['text'][:100]}")
                    continue
                    
                if item['html_tag'] == 'h1':
                    # If we have a current section, save it
                    if current_section:
                        if current_h3_section:
                            current_h3_section['text'] = f"{current_h3_section['heading']} {' '.join(current_h3_content)}"
                            current_section['subsections'].append(current_h3_section)
                        # Combine all content for embedding
                        section_text = [current_section['heading']]
                        section_text.extend(current_content)
                        for subsection in current_section['subsections']:
                            section_text.append(subsection['heading'])
                            section_text.extend(subsection['content'])
                        current_section['text'] = ' '.join(section_text)
                        sections.append(current_section)
                        debug_log(f"Completed section: {current_section['heading'][:100]}")
                        debug_log(f"Combined text length: {len(current_section['text'])} characters")
                    
                    # Start new H1 section
                    current_section = {
                        'type': 'h1',
                        'heading': item['text'],
                        'text': '',
                        'content': [],
                        'subsections': []
                    }
                    current_content = []
                    current_h3_section = None
                    current_h3_content = []
                    debug_log(f"\nStarting new H1 section: {item['text'][:100]}")
                
                elif item['html_tag'] == 'h2':
                    # If we have a current section, save it
                    if current_section:
                        if current_h3_section:
                            current_h3_section['text'] = f"{current_h3_section['heading']} {' '.join(current_h3_content)}"
                            current_section['subsections'].append(current_h3_section)
                        # Combine all content for embedding
                        section_text = [current_section['heading']]
                        section_text.extend(current_content)
                        for subsection in current_section['subsections']:
                            section_text.append(subsection['heading'])
                            section_text.extend(subsection['content'])
                        current_section['text'] = ' '.join(section_text)
                        sections.append(current_section)
                        debug_log(f"Completed section: {current_section['heading'][:100]}")
                        debug_log(f"Combined text length: {len(current_section['text'])} characters")
                    
                    # Start new H2 section
                    current_section = {
                        'type': 'h2',
                        'heading': item['text'],
                        'text': '',
                        'content': [],
                        'subsections': []
                    }
                    current_content = []
                    current_h3_section = None
                    current_h3_content = []
                    debug_log(f"\nStarting new H2 section: {item['text'][:100]}")
                
                elif item['html_tag'] == 'h3':
                    # Skip H3s that are part of navigation or previews
                    if item.get('parent_class') and any(cls in str(item['parent_class']).lower() 
                                                      for cls in ['nav', 'menu', 'preview', 'list-item']):
                        debug_log(f"Skipping navigation/preview H3: {item['text'][:100]}")
                        continue
                        
                    # If we have a current H3 section, save it
                    if current_h3_section:
                        current_h3_section['text'] = f"{current_h3_section['heading']} {' '.join(current_h3_content)}"
                        current_section['subsections'].append(current_h3_section)
                    
                    # Start new H3 subsection
                    current_h3_section = {
                        'type': 'h3',
                        'heading': item['text'],
                        'text': '',
                        'content': []
                    }
                    current_h3_content = []
                    debug_log(f"  Starting new H3 subsection: {item['text'][:100]}")
            
            else:  # This is a paragraph
                # Skip paragraphs that are part of navigation or previews
                if item.get('parent_class') and any(cls in str(item['parent_class']).lower() 
                                                  for cls in ['nav', 'menu', 'preview', 'list-item']):
                    debug_log(f"Skipping navigation/preview paragraph: {item['text'][:100]}")
                    continue
                    
                if current_h3_section:
                    current_h3_content.append(item['text'])
                    current_h3_section['content'].append(item['text'])
                    debug_log(f"  Added paragraph to H3 section: {item['text'][:100]}")
                else:
                    current_content.append(item['text'])
                    debug_log(f"Added paragraph to main section: {item['text'][:100]}")
        
        # Save the last section
        if current_section:
            if current_h3_section:
                current_h3_section['text'] = f"{current_h3_section['heading']} {' '.join(current_h3_content)}"
                current_section['subsections'].append(current_h3_section)
            # Combine all content for embedding
            section_text = [current_section['heading']]
            section_text.extend(current_content)
            for subsection in current_section['subsections']:
                section_text.append(subsection['heading'])
                section_text.extend(subsection['content'])
            current_section['text'] = ' '.join(section_text)
            sections.append(current_section)
            debug_log(f"Completed final section: {current_section['heading'][:100]}")
            debug_log(f"Combined text length: {len(current_section['text'])} characters")
        
        # If we have no sections but have paragraphs, create one section
        elif paragraphs:
            section_text = ' '.join(p['text'] for p in paragraphs)
            sections.append({
                'type': 'p',
                'text': section_text,
                'subsections': []
            })
            debug_log("Created section from paragraphs only")
            debug_log(f"Combined text length: {len(section_text)} characters")
        
        debug_log(f"\nExtracted {len(sections)} total sections")
        for i, section in enumerate(sections):
            debug_log(f"Section {i+1} ({section['type']}): {section['heading'][:100] if 'heading' in section else 'No heading'}")
            if section['subsections']:
                debug_log(f"  Contains {len(section['subsections'])} subsections")
                for j, subsection in enumerate(section['subsections']):
                    debug_log(f"  Subsection {j+1} ({subsection['type']}): {subsection['heading'][:100]}")
            debug_log(f"  Total text length for embedding: {len(section['text'])} characters")
        
        return sections
    except json.JSONDecodeError as e:
        debug_log(f"Error decoding JSON: {str(e)}")
        return None
    except Exception as e:
        debug_log(f"Error processing JSON: {str(e)}")
        debug_log(f"Error type: {type(e)}")
        debug_log(f"Full error: {repr(e)}")
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
        debug_log(f"\nScraping URL: {url}")
        debug_log(f"Encoded URL: {encoded_url}")
        
        # Make the request to ScrapingBee API with simpler parameters
        params = {
            'api_key': api_key,
            'url': encoded_url,
            'render_js': 'true',  # Enable JavaScript rendering
            'premium_proxy': 'true',  # Use premium proxies
            'wait': '5000',  # Wait 5 seconds for JavaScript
            'wait_for': 'body',  # Wait for body element
            'block_resources': 'false',  # Don't block resources to avoid 500 errors
            'extract_rules': json.dumps({
                # Get all content first, we'll filter it later
                'main_content': {
                    'selector': 'body',
                    'type': 'list',
                    'output': 'html'
                },
                # Get all headings
                'headings': {
                    'selector': 'h1, h2, h3',
                    'type': 'list',
                    'output': 'text'
                },
                # Get all paragraphs
                'paragraphs': {
                    'selector': 'p',
                    'type': 'list',
                    'output': 'text'
                }
            })
        }
        
        debug_log("\nMaking request to ScrapingBee API...")
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
        
        debug_log("\nSuccessfully received response from ScrapingBee")
        debug_log(f"Response status code: {response.status_code}")
        debug_log(f"Response content type: {response.headers.get('content-type', 'unknown')}")
        debug_log(f"Response length: {len(response.content)} bytes")
        
        # Check content type and process accordingly
        content_type = response.headers.get('content-type', '').lower()
        if 'application/json' in content_type:
            debug_log("Processing JSON response...")
            try:
                data = json.loads(response.text)
                debug_log(f"JSON keys found: {list(data.keys())}")
                
                # Process the HTML content to remove header/footer before extracting sections
                if 'main_content' in data and data['main_content']:
                    debug_log("\nProcessing main content HTML...")
                    processed_content = []
                    
                    for i, content_html in enumerate(data['main_content']):
                        if not content_html:
                            continue
                            
                        debug_log(f"\nProcessing content block {i+1}:")
                        debug_log(f"Content length: {len(content_html)} characters")
                        
                        try:
                            soup = BeautifulSoup(content_html, 'html.parser')
                            
                            # Remove header, footer, nav, and their children
                            for tag in soup.find_all(['header', 'footer', 'nav']):
                                debug_log(f"Removing {tag.name} element and its children")
                                tag.decompose()
                            
                            # Remove elements with common non-content classes
                            non_content_classes = [
                                'header', 'footer', 'nav', 'menu', 'sidebar', 'widget', 'advertisement',
                                'banner', 'promo', 'cookie-notice', 'popup', 'modal', 'overlay',
                                'newsletter', 'subscription', 'social-share', 'related-posts',
                                'comments', 'recommendations', 'trending', 'popular', 'featured',
                                'latest', 'recent', 'preview', 'list-item', 'card', 'teaser',
                                'summary', 'excerpt', 'snippet', 'thumbnail', 'image', 'video',
                                'gallery', 'carousel', 'slider', 'tabs', 'accordion', 'dropdown',
                                'tooltip', 'notification', 'alert', 'message', 'status', 'progress',
                                'loading', 'spinner', 'icon', 'button', 'link', 'badge', 'label',
                                'tag', 'category', 'meta', 'author', 'date', 'time', 'location',
                                'price', 'rating', 'review', 'comment', 'share', 'like', 'follow',
                                'subscribe', 'search', 'filter', 'sort', 'pagination', 'breadcrumb',
                                'sitemap'
                            ]
                            
                            # Remove elements with non-content classes
                            for tag in soup.find_all(class_=lambda x: x and any(cls in str(x).lower() for cls in non_content_classes)):
                                debug_log(f"Removing element with non-content class: {tag.get('class', [])}")
                                tag.decompose()
                            
                            # Get the cleaned HTML
                            cleaned_html = str(soup)
                            if cleaned_html.strip():
                                processed_content.append(cleaned_html)
                                debug_log(f"Added processed content block {i+1}, length: {len(cleaned_html)}")
                            
                        except Exception as e:
                            debug_log(f"Error processing content block {i+1}: {str(e)}")
                            continue
                    
                    # Update the data with processed content
                    data['main_content'] = processed_content
                    debug_log(f"\nProcessed {len(processed_content)} content blocks")
                
                # Now extract sections from the cleaned content
                sections = extract_sections_from_json(json.dumps(data))
                
            except json.JSONDecodeError as e:
                debug_log(f"Error decoding JSON: {str(e)}")
                debug_log("Raw response content:")
                debug_log(response.text[:1000] + "...")
                return None
        else:
            debug_log("Processing HTML response...")
            sections = extract_sections(response.content)
        
        if not sections:
            st.warning("""
            No content sections found on the page. This could be because:
            1. The page is blocking access
            2. The page has no text content
            3. The page structure is different than expected
            
            Try using a different URL or check if the page is accessible.
            """)
            debug_log("\nRaw response content:")
            debug_log(response.text[:1000] + "...")
            return None
        
        # Filter out very short sections and sections that look like they're from header/footer
        filtered_sections = []
        for section in sections:
            # Skip sections that are too short
            if len(section['text'].split()) <= 3:
                debug_log(f"Skipping short section: {section['text'][:100]}")
                continue
                
            # Skip sections that look like they're from header/footer
            if any(non_content in section['text'].lower() for non_content in 
                  ['menu', 'navigation', 'header', 'footer', 'copyright', 'privacy policy', 'terms of use']):
                debug_log(f"Skipping non-content section: {section['text'][:100]}")
                continue
                
            # Skip sections that are part of navigation or previews
            if section.get('parent_class') and any(cls in str(section['parent_class']).lower() 
                                                 for cls in ['nav', 'menu', 'preview', 'list-item']):
                debug_log(f"Skipping navigation/preview section: {section['text'][:100]}")
                continue
                
            filtered_sections.append(section)
        
        debug_log(f"\nAfter filtering: {len(filtered_sections)} sections remaining")
        for i, section in enumerate(filtered_sections):
            debug_log(f"Section {i+1}: {section['text'][:100]}")
        
        return filtered_sections
    except Exception as e:
        st.error(f"Error scraping webpage: {str(e)}")
        debug_log(f"Scraping error details: {str(e)}")
        debug_log(f"Error type: {type(e)}")
        debug_log(f"Full error: {repr(e)}")
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
            debug_log(f"Getting embedding for keyword: {keyword_input[:100]}")
            embeddings = get_cached_embedding(keyword_input)
            debug_log(f"Received embeddings type: {type(embeddings)}")
            if embeddings is not None:
                debug_log(f"Embeddings content: {str(embeddings)[:200]}")
            
        if embeddings is not None:
            try:
                # Handle both single embedding and list of embeddings
                if isinstance(embeddings, list):
                    if not embeddings:
                        st.error("Received empty list of embeddings")
                        st.stop()
                    keyword_embedding = embeddings[0]
                    debug_log(f"Extracted first embedding from list, type: {type(keyword_embedding)}")
                else:
                    keyword_embedding = embeddings
                    debug_log(f"Using single embedding, type: {type(keyword_embedding)}")
                
                if not isinstance(keyword_embedding, np.ndarray):
                    st.error(f"Invalid embedding type: {type(keyword_embedding)}")
                    debug_log(f"Embedding content: {str(keyword_embedding)[:200]}")
                    st.stop()
                
                # Display embedding information
                st.subheader("Embedding Information")
                try:
                    st.write(f"Vector dimension: {keyword_embedding.shape[0]}")
                    st.write(f"Vector shape: {keyword_embedding.shape}")
                    st.write(f"Data type: {keyword_embedding.dtype}")
                except Exception as e:
                    st.error(f"Error displaying embedding information: {str(e)}")
                    debug_log(f"Embedding shape: {getattr(keyword_embedding, 'shape', 'No shape')}")
                    debug_log(f"Embedding type: {type(keyword_embedding)}")
                    debug_log(f"Embedding content: {str(keyword_embedding)[:200]}")
                    st.stop()
                
                # Display the embedding vector
                st.subheader("Embedding Vector")
                try:
                    st.code(keyword_embedding.tolist())
                except Exception as e:
                    st.error(f"Error displaying embedding vector: {str(e)}")
                    debug_log(f"Embedding content: {str(keyword_embedding)[:200]}")
                    st.stop()
                
                # Add download button for the embedding
                try:
                    st.download_button(
                        label="Download Embedding as CSV",
                        data=",".join(map(str, keyword_embedding)),
                        file_name="keyword_embedding.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error creating download button: {str(e)}")
                    debug_log(f"Embedding content: {str(keyword_embedding)[:200]}")
                    st.stop()
            except Exception as e:
                st.error(f"Error processing embedding: {str(e)}")
                debug_log(f"Full error details: {str(e)}")
                debug_log(f"Embeddings type: {type(embeddings)}")
                debug_log(f"Embeddings content: {str(embeddings)[:200]}")
                st.stop()
        else:
            st.error("Failed to generate embedding. Please check the error messages above.")
            debug_log("get_cached_embedding returned None")

with tab2:
    st.subheader("Analyze Webpage Content")
    url_input = st.text_input("Enter webpage URL:", placeholder="https://example.com")
    
    if url_input:
        with st.spinner("Scraping webpage..."):
            sections = scrape_webpage(url_input)
        
        if sections:
            st.write(f"Found {len(sections)} sections in the webpage")
            
            # Generate embeddings for all sections in batches
            with st.spinner("Generating embeddings for sections..."):
                # Pass the full section objects to get_cached_embedding
                sections_with_embeddings = get_cached_embedding(sections, batch_size=250)
                
                if sections_with_embeddings is None:
                    st.error("""
                    Failed to generate embeddings. This could be due to:
                    1. Vertex AI quota limits - Please try again later or request a quota increase
                    2. Invalid content - Please check if the webpage has valid text content
                    
                    You can request a quota increase here: https://cloud.google.com/vertex-ai/docs/generative-ai/quotas-genai
                    """)
                    st.stop()
                
                # Extract embeddings from sections
                section_embeddings = [section['embedding'] for section in sections_with_embeddings]
            
            # If we have a keyword embedding from tab1, calculate similarity
            if 'keyword_embedding' in locals() and keyword_embedding is not None:
                st.subheader("Similarity Analysis")
                
                try:
                    # Calculate similarity for each section
                    section_similarities = []
                    for section in sections_with_embeddings:
                        similarity = calculate_similarity(keyword_embedding, section['embedding'])
                        section_similarities.append({
                            'heading': section.get('heading', 'No heading'),
                            'type': section['type'],
                            'similarity': similarity,
                            'text_preview': section['text'][:200] + '...' if len(section['text']) > 200 else section['text']
                        })
                    
                    # Sort sections by similarity
                    section_similarities.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    # Display overall similarity
                    avg_similarity = np.mean([s['similarity'] for s in section_similarities])
                    st.metric(
                        label="Average Page Similarity Score",
                        value=f"{avg_similarity:.3f}",
                        help="Average cosine similarity between the keyword and all webpage sections (range: -1 to 1)"
                    )
                    
                    # Display most similar sections
                    st.subheader("Most Similar Sections")
                    for i, section in enumerate(section_similarities[:3], 1):
                        st.markdown(f"""
                        **{i}. {section['heading']}** ({section['type']})  
                        Similarity: {section['similarity']:.3f}  
                        Preview: {section['text_preview']}
                        """)
                    
                    # Display all sections in a table
                    st.subheader("All Section Similarities")
                    df = pd.DataFrame(section_similarities)
                    df['similarity'] = df['similarity'].round(3)
                    df = df.rename(columns={
                        'heading': 'Section Heading',
                        'type': 'Section Type',
                        'similarity': 'Similarity Score',
                        'text_preview': 'Content Preview'
                    })
                    st.dataframe(df, use_container_width=True)
                    
                    # Add download button for results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Section Analysis as CSV",
                        data=csv,
                        file_name="section_analysis.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error calculating similarity: {str(e)}")
                    debug_log(f"Error details: {str(e)}")
            else:
                st.info("Enter a keyword in the first tab to analyze similarities with webpage content.")
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
            st.warning("""
            No valid embeddings found for the analyzed webpage. This could be due to:
            1. Vertex AI quota limits - Please try again later or request a quota increase
            2. Invalid content - Please check if the webpage has valid text content
            
            You can request a quota increase here: https://cloud.google.com/vertex-ai/docs/generative-ai/quotas-genai
            """)
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
                                
                            # Generate embeddings for all sections in batches
                            section_texts = [section['text'] for section in sections]
                            embeddings = get_cached_embedding(section_texts, batch_size=5)
                            
                            if embeddings is None:
                                st.warning(f"""
                                Could not generate embeddings for {url}. This could be due to:
                                1. Vertex AI quota limits - Please try again later
                                2. Invalid content - Please check if the webpage has valid text content
                                """)
                                continue
                            
                            section_embeddings_url = embeddings if isinstance(embeddings, list) else [embeddings]
                            
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
                    st.error("""
                    No URLs were successfully analyzed. This could be due to:
                    1. Vertex AI quota limits - Please try again later or request a quota increase
                    2. Invalid content - Please check if the webpages have valid text content
                    3. Access issues - Some URLs may be blocking access
                    
                    You can request a quota increase here: https://cloud.google.com/vertex-ai/docs/generative-ai/quotas-genai
                    """)
    else:
        st.warning("Please enter at least one URL to analyze.")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and Google's Vertex AI text-embedding-005 model") 
