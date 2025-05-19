import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
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
        debug_log("\nStarting batch embedding generation...")
        debug_log(f"Number of texts to process: {len(texts)}")
        for i, text in enumerate(texts[:3]):  # Show first 3 texts
            debug_log(f"\nText {i+1} preview:")
            debug_log(f"Length: {len(text)} characters")
            debug_log(f"First 200 chars: {text[:200]}...")
        
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
                    
                    # Make the API request with retry logic
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
            if hasattr(current, 'name'):
                if current.name == 'p':
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
    """Extract sections from ScrapingBee JSON response, maintaining proper heading hierarchy"""
    debug_log("Processing JSON response...")
    sections = []
    
    try:
        data = json.loads(json_data)
        debug_log(f"JSON keys found: {list(data.keys())}")
        
        if 'main_content' in data and data['main_content']:
            debug_log("Processing main content HTML...")
            debug_log(f"Number of main_content items: {len(data['main_content'])}")
            
            # First, combine all content
            combined_html = ''
            for content_html in data['main_content']:
                if content_html:
                    combined_html += content_html
            
            if combined_html:
                debug_log("\nProcessing combined HTML content...")
                debug_log(f"Combined content length: {len(combined_html)} characters")
                
                try:
                    # Create a single soup object from all content
                    soup = BeautifulSoup(combined_html, 'html.parser')
                    
                    # Debug: Log all elements with classes
                    debug_log("\nSearching for elements with classes...")
                    all_elements_with_classes = soup.find_all(class_=True)
                    debug_log(f"Found {len(all_elements_with_classes)} elements with classes")
                    
                    # Debug: Log all elements that might be navigation
                    debug_log("\nChecking for navigation elements...")
                    for element in all_elements_with_classes:
                        classes = element.get('class', [])
                        if classes:
                            debug_log(f"\nElement: {element.name}")
                            debug_log(f"Classes: {classes}")
                            debug_log(f"HTML: {element.prettify()[:200]}")
                            debug_log(f"Text content: {element.get_text().strip()[:200]}")
                            
                            # Check each class individually
                            for class_name in classes:
                                if 'nav' in class_name.lower() or 'navigation' in class_name.lower():
                                    debug_log(f"FOUND NAVIGATION CLASS: {class_name}")
                    
                    # Remove header, nav, and footer elements
                    for tag in ['header', 'nav', 'footer']:
                        for element in soup.find_all(tag):
                            debug_log(f"\nRemoving {tag} element:")
                            debug_log(f"Content: {element.get_text().strip()[:200]}")
                            element.decompose()
                    
                    # Remove elements with nav/navigation in class names (as substrings)
                    debug_log("\nRemoving elements with navigation classes...")
                    for element in soup.find_all(class_=True):
                        classes = element.get('class', [])
                        if classes:
                            debug_log(f"\nChecking element: {element.name}")
                            debug_log(f"Classes: {classes}")
                            
                            # Check each class individually and log the decision
                            should_remove = False
                            for class_name in classes:
                                class_name_lower = class_name.lower()
                                # More specific checks for actual navigation elements
                                if (
                                    # Common navigation class names
                                    class_name_lower in ['nav', 'navigation', 'navbar', 'navbar-nav', 'nav-menu', 'nav-wrapper', 'nav-container'] or
                                    # Common navigation role attributes
                                    element.get('role') == 'navigation' or
                                    # Common navigation ARIA roles
                                    element.get('aria-label', '').lower() in ['navigation', 'main navigation', 'primary navigation'] or
                                    # Common navigation IDs
                                    element.get('id', '').lower() in ['nav', 'navigation', 'navbar', 'main-nav', 'primary-nav']
                                ):
                                    debug_log(f"MATCH FOUND - Element is actual navigation: {class_name}")
                                    should_remove = True
                                    break
                                # Skip elements that have 'nav' in class but aren't navigation
                                elif 'nav' in class_name_lower:
                                    # Check if this is a legitimate content element with 'nav' in class
                                    if (
                                        element.name in ['div', 'section', 'article'] and
                                        not any(nav_term in element.get('role', '').lower() for nav_term in ['nav', 'navigation']) and
                                        not any(nav_term in element.get('aria-label', '').lower() for nav_term in ['nav', 'navigation'])
                                    ):
                                        debug_log(f"Keeping element - 'nav' in class but not navigation: {class_name}")
                                        continue
                                    else:
                                        debug_log(f"MATCH FOUND - Class '{class_name}' contains navigation term")
                                        should_remove = True
                                        break
                            
                            if should_remove:
                                debug_log(f"REMOVING element with classes: {classes}")
                                debug_log(f"Content being removed: {element.get_text().strip()[:200]}")
                                element.decompose()
                            else:
                                debug_log("Keeping element - no navigation classes found")
                    
                    # Get the cleaned HTML
                    cleaned_html = str(soup)
                    debug_log(f"\nCleaned HTML length: {len(cleaned_html)} characters")
                    
                    # Verify removal
                    remaining_nav_elements = soup.find_all(class_=True)
                    nav_elements_found = False
                    for element in remaining_nav_elements:
                        classes = element.get('class', [])
                        if any('nav' in c.lower() or 'navigation' in c.lower() for c in classes):
                            nav_elements_found = True
                            debug_log(f"\nWARNING: Found remaining navigation element!")
                            debug_log(f"Element: {element.name}")
                            debug_log(f"Classes: {classes}")
                            debug_log(f"Content: {element.get_text().strip()[:200]}")
                    
                    if not nav_elements_found:
                        debug_log("\nNo remaining navigation elements found")
                    
                    # Create new soup from cleaned HTML
                    soup = BeautifulSoup(cleaned_html, 'html.parser')
                    
                    # Find all heading tags in order
                    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    debug_log(f"\nFound {len(headings)} heading tags")
                    
                    # Process headings in order, maintaining hierarchy
                    current_section = None
                    current_content = []
                    
                    for heading in headings:
                        if not heading or not heading.name:  # Skip None or invalid headings
                            continue
                        
                        # Skip headings that are part of navigation
                        if heading.get('class'):
                            if any(nav_term in class_name for class_name in heading.get('class', []) for nav_term in ['nav', 'navigation']):
                                debug_log(f"\nSkipping navigation heading: {heading.get_text().strip()}")
                                debug_log(f"Classes: {heading.get('class', [])}")
                                continue
                        
                        # Get heading text
                        heading_text = heading.get_text().strip()
                        if not heading_text:
                            continue
                        
                        # Save previous section if it exists
                        if current_section:
                            # Add accumulated content to the section
                            if current_content:
                                current_section['content'].extend(current_content)
                                current_section['text'] = f"{current_section['heading']} {' '.join(current_section['content'])}"
                            
                            # Only add meaningful sections
                            if len(current_section['text'].split()) > 3:
                                sections.append(current_section)
                                debug_log(f"Completed section: {current_section['heading'][:100]}")
                                debug_log(f"Content items: {len(current_section['content'])}")
                            
                            # Reset content for next section
                            current_content = []
                        
                        # Start new section
                        current_section = {
                            'type': heading.name,
                            'heading': heading_text,
                            'text': heading_text,
                            'content': []
                        }
                        debug_log(f"\nStarting new {heading.name} section: {heading_text[:100]}")
                        
                        # Get all content until next heading of same or higher level
                        current = heading.next_sibling
                        while current and not (hasattr(current, 'name') and 
                                             current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and 
                                             int(current.name[1]) <= int(heading.name[1])):
                            if hasattr(current, 'name'):
                                # Skip elements that are part of navigation
                                if current.get('class'):
                                    if any(nav_term in class_name for class_name in current.get('class', []) for nav_term in ['nav', 'navigation']):
                                        debug_log(f"\nSkipping navigation content: {current.get_text().strip()[:100]}")
                                        debug_log(f"Classes: {current.get('class', [])}")
                                        current = current.next_sibling
                                        continue
                                
                                # Get text from any element
                                text = current.get_text().strip()
                                if text and len(text.split()) > 2:  # Skip very short text
                                    current_content.append(text)
                                    debug_log(f"Added content: {text[:100]}")
                            current = current.next_sibling
                    
                    # Save final section
                    if current_section:
                        if current_content:
                            current_section['content'].extend(current_content)
                            current_section['text'] = f"{current_section['heading']} {' '.join(current_section['content'])}"
                        
                        if len(current_section['text'].split()) > 3:
                            sections.append(current_section)
                            debug_log(f"Completed final section: {current_section['heading'][:100]}")
                            debug_log(f"Content items: {len(current_section['content'])}")
                    
                except Exception as e:
                    debug_log(f"Error processing HTML content: {str(e)}")
                    debug_log(f"Error type: {type(e)}")
                    debug_log(f"Full error: {repr(e)}")
                    return None
        
        # If no sections found, try to get content from the cleaned HTML
        if not sections:
            debug_log("\nNo sections found with headings, trying content from cleaned HTML...")
            try:
                # Get all text content from the body
                body = soup.find('body')
                if body:
                    # Get all text nodes and elements
                    content_elements = []
                    for element in body.descendants:
                        if element and element.name and element.name not in ['script', 'style', 'meta', 'link']:
                            # Skip elements that are part of navigation
                            if element.get('class'):
                                if any(nav_term in class_name for class_name in element.get('class', []) for nav_term in ['nav', 'navigation']):
                                    continue
                            
                            text = element.get_text().strip()
                            if text and len(text.split()) > 2:  # Skip very short text
                                content_elements.append(text)
                    
                    if content_elements:
                        # Create a single section with all content
                        sections.append({
                            'type': 'content',
                            'heading': 'Content Section',
                            'text': ' '.join(content_elements),
                            'content': content_elements
                        })
            except Exception as e:
                debug_log(f"Error processing fallback content: {str(e)}")
                debug_log(f"Error type: {type(e)}")
                debug_log(f"Full error: {repr(e)}")
        
        debug_log(f"\nExtracted {len(sections)} total sections")
        for i, section in enumerate(sections):
            debug_log(f"Section {i+1} ({section['type']}): {section['heading'][:100]}")
            debug_log(f"  Text length: {len(section['text'])} characters")
            debug_log(f"  Content items: {len(section['content'])}")
        
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
        
        # Make the request to ScrapingBee API with extraction rules that exclude header/footer/nav
        params = {
            'api_key': api_key,
            'url': encoded_url,
            'render_js': 'true',  # Enable JavaScript rendering
            'premium_proxy': 'true',  # Use premium proxies
            'wait': '5000',  # Wait 5 seconds for JavaScript
            'wait_for': 'body',  # Wait for body element
            'block_resources': 'false',  # Don't block resources to avoid 500 errors
            'extract_rules': json.dumps({
                'main_content': {
                    'selector': 'body',
                    'type': 'list',
                    'output': 'html'
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
                
                # Process the HTML content
                if 'main_content' in data and data['main_content']:
                    debug_log("\nProcessing main content HTML...")
                    processed_content = []
                    
                    for i, content_html in enumerate(data['main_content']):
                        if not content_html:
                            continue
                            
                        debug_log(f"\nProcessing content_html {i+1}:")
                        debug_log(f"Content length: {len(content_html)} characters")
                        
                        try:
                            # Create soup object and log initial structure
                            soup = BeautifulSoup(content_html, 'html.parser')
                            
                            # Ensure we have a valid body element
                            body = soup.find('body')
                            if not body:
                                debug_log("No body element found, creating one")
                                body = soup.new_tag('body')
                                body.append(soup)
                                soup = BeautifulSoup(str(body), 'html.parser')
                            
                            # Log all header, footer, and nav elements found
                            debug_log("\nSearching for unwanted elements...")
                            header_elements = soup.find_all('header')
                            footer_elements = soup.find_all('footer')
                            nav_elements = soup.find_all('nav')
                            
                            debug_log(f"Found {len(header_elements)} header elements")
                            debug_log(f"Found {len(footer_elements)} footer elements")
                            debug_log(f"Found {len(nav_elements)} nav elements")
                            
                            # Remove all unwanted elements
                            for tag in header_elements + footer_elements + nav_elements:
                                if tag and tag.parent:  # Check if tag exists and has a parent
                                    debug_log(f"\nRemoving {tag.name} element:")
                                    debug_log(f"Content: {tag.get_text().strip()[:200]}")
                                    tag.decompose()
                            
                            # Get all text content from the body
                            content_elements = []
                            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article', 'section', 'div']):
                                if element and element.parent:  # Check if element exists and has a parent
                                    # Skip elements that are part of navigation
                                    if element.get('class'):
                                        classes = element.get('class', [])
                                        if any(
                                            class_name.lower() in ['nav', 'navigation', 'navbar', 'navbar-nav', 'nav-menu', 'nav-wrapper', 'nav-container', 
                                                                 'cookie', 'cookies', 'cookie-banner', 'cookie-notice', 'cookie-consent', 'cookie-policy', 'cookie-bar', 'cookie-warning', 'cookie-popup', 'cookie-modal',
                                                                 'login', 'log-in', 'login-form', 'login-modal', 'login-popup', 'login-wrapper', 'login-container', 'login-box', 'login-panel', 'login-dialog', 'login-overlay', 'login-banner', 'login-notice',
                                                                 'preference', 'preferences', 'preference-panel', 'preference-modal', 'preference-popup', 'preference-wrapper', 'preference-container', 'preference-box', 'preference-settings', 'preference-menu', 'preference-dialog', 'preference-overlay', 'preference-banner', 'preference-notice']
                                            or element.get('role') == 'navigation'
                                            or element.get('aria-label', '').lower() in ['navigation', 'main navigation', 'primary navigation', 
                                                                                       'cookie consent', 'cookie notice', 'cookie policy',
                                                                                       'login', 'log in', 'login form', 'login modal', 'login popup', 'login panel', 'login dialog',
                                                                                       'preference', 'preferences', 'preference panel', 'preference modal', 'preference popup', 'preference settings', 'preference menu', 'preference dialog']
                                            or element.get('id', '').lower() in ['nav', 'navigation', 'navbar', 'main-nav', 'primary-nav', 
                                                                               'cookie', 'cookies', 'cookie-banner', 'cookie-notice', 'cookie-consent', 'cookie-policy', 'cookie-bar', 'cookie-warning', 'cookie-popup', 'cookie-modal',
                                                                               'login', 'log-in', 'login-form', 'login-modal', 'login-popup', 'login-wrapper', 'login-container', 'login-box', 'login-panel', 'login-dialog', 'login-overlay', 'login-banner', 'login-notice',
                                                                               'preference', 'preferences', 'preference-panel', 'preference-modal', 'preference-popup', 'preference-wrapper', 'preference-container', 'preference-box', 'preference-settings', 'preference-menu', 'preference-dialog', 'preference-overlay', 'preference-banner', 'preference-notice']
                                            for class_name in classes
                                        ):
                                            continue
                                    
                                    text = element.get_text().strip()
                                    if text and len(text.split()) > 2:  # Skip very short text
                                        content_elements.append({
                                            'type': element.name,
                                            'text': text,
                                            'heading': element.name.startswith('h') and text or None
                                        })
                            
                            if content_elements:
                                # Create sections from content elements
                                sections = []
                                current_section = None
                                
                                for element in content_elements:
                                    if element['type'].startswith('h'):  # This is a heading
                                        if current_section:
                                            sections.append(current_section)
                                        current_section = {
                                            'type': element['type'],
                                            'heading': element['text'],
                                            'text': element['text'],
                                            'content': []
                                        }
                                    elif current_section:
                                        current_section['content'].append(element['text'])
                                        current_section['text'] = f"{current_section['heading']} {' '.join(current_section['content'])}"
                                    else:
                                        # Create a default section for content before first heading
                                        current_section = {
                                            'type': 'content',
                                            'heading': 'Content Section',
                                            'text': element['text'],
                                            'content': [element['text']]
                                        }
                                
                                if current_section:
                                    sections.append(current_section)
                                
                                if sections:
                                    processed_content.extend(sections)
                                    debug_log(f"\nExtracted {len(sections)} sections from content block {i+1}")
                                else:
                                    debug_log(f"\nNo sections found in content block {i+1}")
                            
                        except Exception as e:
                            debug_log(f"Error processing content block {i+1}: {str(e)}")
                            debug_log(f"Error type: {type(e)}")
                            debug_log(f"Full error: {repr(e)}")
                            continue
                    
                    if processed_content:
                        debug_log(f"\nSuccessfully processed {len(processed_content)} total sections")
                        return processed_content
                    else:
                        debug_log("\nNo content sections found after processing")
                        return None
                else:
                    debug_log("No main_content found in JSON response")
                    return None
                
            except json.JSONDecodeError as e:
                debug_log(f"Error decoding JSON: {str(e)}")
                debug_log("Raw response content:")
                debug_log(response.text[:1000] + "...")
                return None
        else:
            debug_log("Processing HTML response...")
            # For direct HTML responses, create soup and process content
            try:
                soup = BeautifulSoup(response.content, 'html.parser')
                body = soup.find('body')
                if not body:
                    debug_log("No body element found in direct HTML response")
                    return None
                
                # Process content using the same logic as above
                content_elements = []
                for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article', 'section', 'div']):
                    if element and element.parent:
                        # Skip navigation elements
                        if element.get('class'):
                            classes = element.get('class', [])
                            if any(
                                class_name.lower() in ['nav', 'navigation', 'navbar', 'navbar-nav', 'nav-menu', 'nav-wrapper', 'nav-container', 
                                                                 'cookie', 'cookies', 'cookie-banner', 'cookie-notice', 'cookie-consent', 'cookie-policy', 'cookie-bar', 'cookie-warning', 'cookie-popup', 'cookie-modal',
                                                                 'login', 'log-in', 'login-form', 'login-modal', 'login-popup', 'login-wrapper', 'login-container', 'login-box', 'login-panel', 'login-dialog', 'login-overlay', 'login-banner', 'login-notice',
                                                                 'preference', 'preferences', 'preference-panel', 'preference-modal', 'preference-popup', 'preference-wrapper', 'preference-container', 'preference-box', 'preference-settings', 'preference-menu', 'preference-dialog', 'preference-overlay', 'preference-banner', 'preference-notice']
                                            or element.get('role') == 'navigation'
                                            or element.get('aria-label', '').lower() in ['navigation', 'main navigation', 'primary navigation', 
                                                                                       'cookie consent', 'cookie notice', 'cookie policy',
                                                                                       'login', 'log in', 'login form', 'login modal', 'login popup', 'login panel', 'login dialog',
                                                                                       'preference', 'preferences', 'preference panel', 'preference modal', 'preference popup', 'preference settings', 'preference menu', 'preference dialog']
                                            or element.get('id', '').lower() in ['nav', 'navigation', 'navbar', 'main-nav', 'primary-nav', 
                                                                               'cookie', 'cookies', 'cookie-banner', 'cookie-notice', 'cookie-consent', 'cookie-policy', 'cookie-bar', 'cookie-warning', 'cookie-popup', 'cookie-modal',
                                                                               'login', 'log-in', 'login-form', 'login-modal', 'login-popup', 'login-wrapper', 'login-container', 'login-box', 'login-panel', 'login-dialog', 'login-overlay', 'login-banner', 'login-notice',
                                                                               'preference', 'preferences', 'preference-panel', 'preference-modal', 'preference-popup', 'preference-wrapper', 'preference-container', 'preference-box', 'preference-settings', 'preference-menu', 'preference-dialog', 'preference-overlay', 'preference-banner', 'preference-notice']
                                            for class_name in classes
                                        ):
                                            continue
                        
                        text = element.get_text().strip()
                        if text and len(text.split()) > 2:
                            content_elements.append({
                                'type': element.name,
                                'text': text,
                                'heading': element.name.startswith('h') and text or None
                            })
                
                if content_elements:
                    # Create sections from content elements
                    sections = []
                    current_section = None
                    
                    for element in content_elements:
                        if element['type'].startswith('h'):
                            if current_section:
                                sections.append(current_section)
                            current_section = {
                                'type': element['type'],
                                'heading': element['text'],
                                'text': element['text'],
                                'content': []
                            }
                        elif current_section:
                            current_section['content'].append(element['text'])
                            current_section['text'] = f"{current_section['heading']} {' '.join(current_section['content'])}"
                        else:
                            current_section = {
                                'type': 'content',
                                'heading': 'Content Section',
                                'text': element['text'],
                                'content': [element['text']]
                            }
                    
                    if current_section:
                        sections.append(current_section)
                    
                    if sections:
                        debug_log(f"\nExtracted {len(sections)} sections from direct HTML")
                        return sections
                
                debug_log("\nNo content sections found in direct HTML")
                return None
                
            except Exception as e:
                debug_log(f"Error processing direct HTML: {str(e)}")
                debug_log(f"Error type: {type(e)}")
                debug_log(f"Full error: {repr(e)}")
                return None
        
    except Exception as e:
        st.error(f"Error scraping webpage: {str(e)}")
        debug_log(f"Scraping error details: {str(e)}")
        debug_log(f"Error type: {type(e)}")
        debug_log(f"Full error: {repr(e)}")
        return None

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings with additional validation"""
    try:
        # Validate inputs
        if embedding1 is None or embedding2 is None:
            debug_log("Invalid input: one or both embeddings are None")
            return 0.0
            
        if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
            debug_log("Invalid input: embeddings must be numpy arrays")
            return 0.0
            
        # Debug logging for input embeddings
        debug_log(f"\nCalculating similarity between embeddings:")
        debug_log(f"Embedding 1 type: {type(embedding1)}, shape: {embedding1.shape}")
        debug_log(f"Embedding 2 type: {type(embedding2)}, shape: {embedding2.shape}")
        
        # Ensure embeddings are numpy arrays of float32
        embedding1 = np.array(embedding1, dtype=np.float32)
        embedding2 = np.array(embedding2, dtype=np.float32)
            
        # Debug logging for numpy arrays
        debug_log(f"After conversion - Embedding 1: mean={float(np.mean(embedding1)):.4f}, std={float(np.std(embedding1)):.4f}")
        debug_log(f"After conversion - Embedding 2: mean={float(np.mean(embedding2)):.4f}, std={float(np.std(embedding2)):.4f}")
        
        # Normalize embeddings
        norm1 = float(np.linalg.norm(embedding1))
        norm2 = float(np.linalg.norm(embedding2))
        debug_log(f"Norms before normalization: {norm1:.4f}, {norm2:.4f}")
        
        if norm1 == 0 or norm2 == 0:
            debug_log("Invalid input: zero norm detected")
            return 0.0
        
        embedding1 = embedding1 / norm1
        embedding2 = embedding2 / norm2
        
        # Debug logging after normalization
        debug_log(f"After normalization - Embedding 1: mean={float(np.mean(embedding1)):.4f}, std={float(np.std(embedding1)):.4f}")
        debug_log(f"After normalization - Embedding 2: mean={float(np.mean(embedding2)):.4f}, std={float(np.std(embedding2)):.4f}")
        debug_log(f"Norms after normalization: {float(np.linalg.norm(embedding1)):.4f}, {float(np.linalg.norm(embedding2)):.4f}")
        
        # Calculate cosine similarity
        similarity = float(np.dot(embedding1, embedding2))
        debug_log(f"Raw similarity score: {similarity:.4f}")
        
        # Add validation to ensure similarity is in expected range
        if not -1.0 <= similarity <= 1.0:
            debug_log(f"Warning: Similarity score {similarity} outside expected range [-1, 1]")
            similarity = np.clip(similarity, -1.0, 1.0)
            
        return float(similarity)
    except Exception as e:
        debug_log(f"Error in calculate_similarity: {str(e)}")
        debug_log(f"Error type: {type(e)}")
        debug_log(f"Full error: {repr(e)}")
        return 0.0

def calculate_weighted_similarity(sections, keyword_embedding):
    """Calculate weighted similarity score for a set of sections"""
    if not sections or keyword_embedding is None or not isinstance(keyword_embedding, np.ndarray):
        debug_log("Invalid input to calculate_weighted_similarity")
        return 0.0, [], {}
        
    try:
        debug_log("\nCalculating weighted similarity:")
        debug_log(f"Number of sections: {len(sections)}")
        debug_log(f"Keyword embedding type: {type(keyword_embedding)}, shape: {keyword_embedding.shape if hasattr(keyword_embedding, 'shape') else 'no shape'}")
        
        # Calculate raw similarities
        section_similarities = []
        for i, section in enumerate(sections):
            if 'embedding' not in section or not isinstance(section['embedding'], np.ndarray):
                debug_log(f"Skipping section {i+1}: Invalid embedding")
                continue
                
            debug_log(f"\nProcessing section {i+1}:")
            debug_log(f"Section heading: {section.get('heading', 'No heading')}")
            debug_log(f"Section text length: {len(section['text'].split())} words")
            debug_log(f"Section text preview: {section['text'][:200]}...")
            
            similarity = calculate_similarity(keyword_embedding, section['embedding'])
            debug_log(f"Section similarity score: {similarity:.4f}")
            
            section_similarities.append({
                'section': section,
                'similarity': similarity,
                'text_length': len(section['text'].split())
            })
        
        if not section_similarities:
            debug_log("No valid sections found for similarity calculation")
            return 0.0, [], {}
        
        # Sort sections by similarity
        section_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        debug_log("\nSorted section similarities:")
        for i, item in enumerate(section_similarities[:3]):  # Show top 3
            debug_log(f"Top {i+1}: {item['section'].get('heading', 'No heading')}")
            debug_log(f"  Similarity: {item['similarity']:.4f}")
            debug_log(f"  Length: {item['text_length']} words")
        
        # Calculate weights based on similarity and content length
        total_weight = 0
        weighted_sum = 0
        detailed_scores = []
        
        debug_log("\nCalculating weights:")
        for i, item in enumerate(section_similarities):
            # Higher weight for more similar sections
            similarity_weight = 1.0 / (i + 1)  # 1.0, 0.5, 0.33, 0.25, etc.
            
            # Weight based on content length (normalized)
            length_weight = min(1.0, item['text_length'] / 100)  # Cap at 100 words
            
            # Combined weight
            weight = similarity_weight * length_weight
            total_weight += weight
            weighted_sum += item['similarity'] * weight
            
            debug_log(f"\nSection {i+1} weights:")
            debug_log(f"  Similarity weight: {similarity_weight:.4f}")
            debug_log(f"  Length weight: {length_weight:.4f}")
            debug_log(f"  Combined weight: {weight:.4f}")
            debug_log(f"  Weighted contribution: {item['similarity'] * weight:.4f}")
            
            detailed_scores.append({
                'heading': item['section'].get('heading', 'No heading'),
                'similarity': item['similarity'],
                'weight': weight,
                'length': item['text_length'],
                'text_preview': item['section']['text'][:200] + '...' if len(item['section']['text']) > 200 else item['section']['text']
            })
        
        # Calculate weighted average
        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0.0
        debug_log(f"\nFinal weighted average: {weighted_avg:.4f}")
        
        # Calculate additional metrics
        raw_similarities = [s['similarity'] for s in section_similarities]
        metrics = {
            'raw_avg': float(np.mean(raw_similarities)),
            'raw_max': float(max(raw_similarities)),
            'raw_min': float(min(raw_similarities)),
            'std_dev': float(np.std(raw_similarities)),
            'section_count': len(sections),
            'weighted_avg': float(weighted_avg)
        }
        
        debug_log("\nFinal metrics:")
        debug_log(f"  Raw average: {metrics['raw_avg']:.4f}")
        debug_log(f"  Raw max: {metrics['raw_max']:.4f}")
        debug_log(f"  Raw min: {metrics['raw_min']:.4f}")
        debug_log(f"  Std dev: {metrics['std_dev']:.4f}")
        debug_log(f"  Section count: {metrics['section_count']}")
        debug_log(f"  Weighted average: {metrics['weighted_avg']:.4f}")
        
        return weighted_avg, detailed_scores, metrics
        
    except Exception as e:
        debug_log(f"Error in calculate_weighted_similarity: {str(e)}")
        debug_log(f"Error type: {type(e)}")
        debug_log(f"Full error: {repr(e)}")
        return 0.0, [], {}

# Create tabs for different input methods
tab1, tab2, tab3, tab4 = st.tabs(["Keyword Embedding", "Webpage Analysis", "Link Profile Analysis", "URL Comparison"])

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
            
            # If we have a keyword embedding from tab1, calculate similarity
            if 'keyword_embedding' in locals() and isinstance(keyword_embedding, np.ndarray):
                st.subheader("Similarity Analysis")
                
                try:
                    # Calculate individual section similarities
                    section_results = []
                    for section in sections_with_embeddings:
                        if 'embedding' not in section or not isinstance(section['embedding'], np.ndarray):
                            continue
                            
                        # Calculate similarity for this section only
                        similarity = calculate_similarity(keyword_embedding, section['embedding'])
                        text_length = len(section['text'].split())
                        
                        # Calculate weight based on content length
                        length_weight = min(1.0, text_length / 100)  # Cap at 100 words
                        
                        section_results.append({
                            'section': section,
                            'raw_similarity': similarity,
                            'weighted_similarity': similarity * length_weight,  # Weight by content length
                            'text_length': text_length
                        })
                    
                    if not section_results:
                        st.warning("No valid sections found for similarity analysis")
                        st.stop()
                    
                    # Sort sections by weighted similarity
                    section_results.sort(key=lambda x: x['weighted_similarity'], reverse=True)
                    
                    # Calculate overall metrics
                    raw_similarities = [r['raw_similarity'] for r in section_results]
                    metrics = {
                        'raw_avg': float(np.mean(raw_similarities)),
                        'raw_max': float(max(raw_similarities)),
                        'raw_min': float(min(raw_similarities)),
                        'std_dev': float(np.std(raw_similarities)),
                        'section_count': len(section_results)
                    }
                    
                    # Display overall results
                    st.subheader("Comparison Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            label="Average Similarity",
                            value=f"{metrics['raw_avg']:.3f}",
                            help="Average similarity between the keyword and all analyzed sections"
                        )
                    with col2:
                        st.metric(
                            label="Highest Similarity",
                            value=f"{metrics['raw_max']:.3f}",
                            help="Highest similarity score found across all sections"
                        )
                    with col3:
                        st.metric(
                            label="Std Dev",
                            value=f"{metrics['std_dev']:.3f}",
                            help="Standard deviation of similarity scores"
                        )
                    
                    # Display top 3 most similar sections
                    st.subheader("Most Similar Sections")
                    for i, result in enumerate(section_results[:3], 1):
                        st.markdown(f"""
                        **{i}. {result['section'].get('heading', 'No heading')}**  
                        Raw Similarity: {result['raw_similarity']:.3f}  
                        Weighted Similarity: {result['weighted_similarity']:.3f}  
                        Length: {result['text_length']} words
                        """)
                        
                        # Show section preview
                        st.markdown(f"**Preview:** {result['section']['text'][:200]}...")
                    
                    # Display all sections in a table
                    st.subheader("All Section Similarities")
                    df = pd.DataFrame(section_results)
                    # Round numeric columns
                    numeric_cols = ['raw_similarity', 'weighted_similarity']
                    df[numeric_cols] = df[numeric_cols].round(3)
                    
                    # Add section heading and text preview
                    df['heading'] = df['section'].apply(lambda x: x.get('heading', 'No heading'))
                    df['text_preview'] = df['section'].apply(lambda x: x['text'][:200] + '...' if len(x['text']) > 200 else x['text'])
                    
                    # Rename columns
                    df = df.rename(columns={
                        'heading': 'Section',
                        'raw_similarity': 'Raw Similarity',
                        'weighted_similarity': 'Weighted Similarity',
                        'text_length': 'Length',
                        'text_preview': 'Preview'
                    })
                    
                    # Reorder columns
                    df = df[['Section', 'Raw Similarity', 'Weighted Similarity', 'Length', 'Preview']]
                    
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
                    st.error(f"Error calculating or displaying results: {str(e)}")
                    debug_log(f"Error type: {type(e)}")
                    debug_log(f"Full error: {repr(e)}")
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
                                # Calculate weighted similarity across all sections
                                weighted_similarity, detailed_scores, metrics = calculate_weighted_similarity(
                                    sections_with_embeddings, 
                                    current_page_embedding
                                )
                                
                                url_results.append({
                                    'url': url,
                                    'weighted_similarity': weighted_similarity,
                                    'raw_avg_similarity': metrics['raw_avg'],
                                    'raw_max_similarity': metrics['raw_max'],
                                    'raw_min_similarity': metrics['raw_min'],
                                    'similarity_std_dev': metrics['std_dev'],
                                    'sections_count': metrics['section_count'],
                                    'most_similar_section': detailed_scores[0]['heading'] if detailed_scores else 'No heading',
                                    'most_similar_text': detailed_scores[0]['text_preview'] if detailed_scores else '',
                                    'detailed_scores': detailed_scores
                                })
                            except Exception as e:
                                st.warning(f"Error processing embeddings for {url}: {str(e)}")
                                continue
                                
                        except Exception as e:
                            st.warning(f"Error processing {url}: {str(e)}")
                            continue
                
                if url_results:
                    # Sort URLs by similarity
                    url_results.sort(key=lambda x: x['weighted_similarity'], reverse=True)
                    
                    # Display overall results
                    st.subheader("Comparison Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            label="Average Weighted Similarity",
                            value=f"{np.mean([r['weighted_similarity'] for r in url_results]):.3f}",
                            help="Weighted average similarity between the keyword and all analyzed URLs"
                        )
                    with col2:
                        st.metric(
                            label="Highest Similarity",
                            value=f"{np.max([r['raw_max_similarity'] for r in url_results]):.3f}",
                            help="Highest similarity score found across all URLs"
                        )
                    with col3:
                        st.metric(
                            label="Average Std Dev",
                            value=f"{np.mean([r['similarity_std_dev'] for r in url_results]):.3f}",
                            help="Average standard deviation of similarity scores"
                        )
                    
                    # Display top 3 most similar URLs
                    st.subheader("Most Similar URLs")
                    for i, result in enumerate(url_results[:3], 1):
                        st.markdown(f"""
                        **{i}. {result['url']}**  
                        Weighted Similarity: {result['weighted_similarity']:.3f}  
                        Raw Average: {result['raw_avg_similarity']:.3f}  
                        Raw Max: {result['raw_max_similarity']:.3f}  
                        Raw Min: {result['raw_min_similarity']:.3f}  
                        Std Dev: {result['similarity_std_dev']:.3f}  
                        Sections analyzed: {result['sections_count']}  
                        Most similar section: {result['most_similar_section']}  
                        Preview: {result['most_similar_text']}
                        """)
                        
                        # Show detailed scores for top sections
                        if result['detailed_scores']:
                            st.markdown("**Top 3 Most Similar Sections:**")
                            for j, score in enumerate(result['detailed_scores'][:3], 1):
                                st.markdown(f"""
                                {j}. {score['heading']}  
                                Similarity: {score['similarity']:.3f}  
                                Weight: {score['weight']:.3f}  
                                Length: {score['length']} words  
                                Preview: {score['text_preview']}
                                """)
                    
                    # Display all URLs in a table
                    st.subheader("All URL Similarities")
                    df = pd.DataFrame(url_results)
                    # Round numeric columns
                    numeric_cols = ['weighted_similarity', 'raw_avg_similarity', 'raw_max_similarity', 
                                  'raw_min_similarity', 'similarity_std_dev']
                    df[numeric_cols] = df[numeric_cols].round(3)
                    
                    # Rename columns
                    df = df.rename(columns={
                        'url': 'URL',
                        'weighted_similarity': 'Weighted Similarity',
                        'raw_avg_similarity': 'Raw Average',
                        'raw_max_similarity': 'Raw Maximum',
                        'raw_min_similarity': 'Raw Minimum',
                        'similarity_std_dev': 'Std Deviation',
                        'sections_count': 'Sections Analyzed',
                        'most_similar_section': 'Most Similar Section',
                        'most_similar_text': 'Section Preview'
                    })
                    
                    # Reorder columns
                    df = df[['URL', 'Weighted Similarity', 'Raw Average', 'Raw Maximum', 
                            'Raw Minimum', 'Std Deviation', 'Sections Analyzed', 
                            'Most Similar Section', 'Section Preview']]
                    
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
                    st.error("""
                    No URLs were successfully analyzed. This could be due to:
                    1. Vertex AI quota limits - Please try again later or request a quota increase
                    2. Invalid content - Please check if the webpages have valid text content
                    3. Access issues - Some URLs may be blocking access
                    
                    You can request a quota increase here: https://cloud.google.com/vertex-ai/docs/generative-ai/quotas-genai
                    """)
    else:
        st.warning("Please enter at least one URL to analyze.")

with tab4:
    st.subheader("Compare Multiple URLs")
    st.markdown("""
    Enter a keyword and up to 10 URLs to compare their similarity to the keyword.
    The app will analyze each URL and show:
    - How similar each URL is to the keyword
    - Most relevant URLs ranked by similarity
    - Individual URL scores and section details
    """)
    
    # Get keyword input
    keyword_input = st.text_input("Enter keyword:", placeholder="Enter your keyword here...")
    
    # Get URLs input
    urls_input = st.text_area(
        "Enter URLs to compare (one per line, max 10):",
        height=150,
        placeholder="https://example1.com\nhttps://example2.com\nhttps://example3.com"
    )
    
    if keyword_input and urls_input:
        # Process URLs
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        
        if len(urls) > 10:
            st.warning("Maximum 10 URLs allowed. Only the first 10 will be analyzed.")
            urls = urls[:10]
        
        if not urls:
            st.warning("Please enter at least one valid URL to analyze.")
        else:
            # Generate embedding for keyword
            with st.spinner("Generating keyword embedding..."):
                keyword_embedding = get_cached_embedding(keyword_input)
                if keyword_embedding is None:
                    st.error("Failed to generate keyword embedding. Please try again.")
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
                        sections_with_embeddings = get_cached_embedding(sections, batch_size=5)
                        
                        if sections_with_embeddings is None:
                            st.warning(f"""
                            Could not generate embeddings for {url}. This could be due to:
                            1. Vertex AI quota limits - Please try again later
                            2. Invalid content - Please check if the webpage has valid text content
                            """)
                            continue
                        
                        try:
                            # Calculate weighted similarity across all sections
                            weighted_similarity, detailed_scores, metrics = calculate_weighted_similarity(
                                sections_with_embeddings, 
                                keyword_embedding
                            )
                            
                            url_results.append({
                                'url': url,
                                'weighted_similarity': weighted_similarity,
                                'raw_avg_similarity': metrics['raw_avg'],
                                'raw_max_similarity': metrics['raw_max'],
                                'raw_min_similarity': metrics['raw_min'],
                                'similarity_std_dev': metrics['std_dev'],
                                'sections_count': metrics['section_count'],
                                'most_similar_section': detailed_scores[0]['heading'] if detailed_scores else 'No heading',
                                'most_similar_text': detailed_scores[0]['text_preview'] if detailed_scores else '',
                                'detailed_scores': detailed_scores
                            })
                        except Exception as e:
                            st.warning(f"Error processing embeddings for {url}: {str(e)}")
                            continue
                            
                    except Exception as e:
                        st.warning(f"Error processing {url}: {str(e)}")
                        continue
            
            if url_results:
                # Sort URLs by similarity
                url_results.sort(key=lambda x: x['weighted_similarity'], reverse=True)
                
                # Display overall results
                st.subheader("Comparison Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Average Weighted Similarity",
                        value=f"{np.mean([r['weighted_similarity'] for r in url_results]):.3f}",
                        help="Weighted average similarity between the keyword and all analyzed URLs"
                    )
                with col2:
                    st.metric(
                        label="Highest Similarity",
                        value=f"{np.max([r['raw_max_similarity'] for r in url_results]):.3f}",
                        help="Highest similarity score found across all URLs"
                    )
                with col3:
                    st.metric(
                        label="Average Std Dev",
                        value=f"{np.mean([r['similarity_std_dev'] for r in url_results]):.3f}",
                        help="Average standard deviation of similarity scores"
                    )
                
                # Display top 3 most similar URLs
                st.subheader("Most Similar URLs")
                for i, result in enumerate(url_results[:3], 1):
                    st.markdown(f"""
                    **{i}. {result['url']}**  
                    Weighted Similarity: {result['weighted_similarity']:.3f}  
                    Raw Average: {result['raw_avg_similarity']:.3f}  
                    Raw Max: {result['raw_max_similarity']:.3f}  
                    Raw Min: {result['raw_min_similarity']:.3f}  
                    Std Dev: {result['similarity_std_dev']:.3f}  
                    Sections analyzed: {result['sections_count']}  
                    Most similar section: {result['most_similar_section']}  
                    Preview: {result['most_similar_text']}
                    """)
                    
                    # Show detailed scores for top sections
                    if result['detailed_scores']:
                        st.markdown("**Top 3 Most Similar Sections:**")
                        for j, score in enumerate(result['detailed_scores'][:3], 1):
                            st.markdown(f"""
                            {j}. {score['heading']}  
                            Similarity: {score['similarity']:.3f}  
                            Weight: {score['weight']:.3f}  
                            Length: {score['length']} words  
                            Preview: {score['text_preview']}
                            """)
                
                # Display all URLs in a table
                st.subheader("All URL Similarities")
                df = pd.DataFrame(url_results)
                # Round numeric columns
                numeric_cols = ['weighted_similarity', 'raw_avg_similarity', 'raw_max_similarity', 
                              'raw_min_similarity', 'similarity_std_dev']
                df[numeric_cols] = df[numeric_cols].round(3)
                
                # Rename columns
                df = df.rename(columns={
                    'url': 'URL',
                    'weighted_similarity': 'Weighted Similarity',
                    'raw_avg_similarity': 'Raw Average',
                    'raw_max_similarity': 'Raw Maximum',
                    'raw_min_similarity': 'Raw Minimum',
                    'similarity_std_dev': 'Std Deviation',
                    'sections_count': 'Sections Analyzed',
                    'most_similar_section': 'Most Similar Section',
                    'most_similar_text': 'Section Preview'
                })
                
                # Reorder columns
                df = df[['URL', 'Weighted Similarity', 'Raw Average', 'Raw Maximum', 
                        'Raw Minimum', 'Std Deviation', 'Sections Analyzed', 
                        'Most Similar Section', 'Section Preview']]
                
                st.dataframe(df, use_container_width=True)
                
                # Add download button for results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Analysis Results as CSV",
                    data=csv,
                    file_name="url_comparison.csv",
                    mime="text/csv"
                )
            else:
                st.error("""
                No URLs were successfully analyzed. This could be due to:
                1. Vertex AI quota limits - Please try again later or request a quota increase
                2. Invalid content - Please check if the webpages have valid text content
                3. Access issues - Some URLs may be blocking access
                
                You can request a quota increase here: https://cloud.google.com/vertex-ai/docs/generative-ai/quotas-genai
                """)
    else:
        st.warning("Please enter both a keyword and at least one URL to analyze.")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and Google's Vertex AI text-embedding-005 model") 
