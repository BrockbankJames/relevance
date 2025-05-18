# Universal Sentence Encoder Embedding Generator

This Streamlit app allows you to generate vector embeddings for text using Google's Universal Sentence Encoder (USE). The app provides a simple interface to input text and get the corresponding 512-dimensional embedding vector.

## Features

- Generate embeddings for any text input
- View embedding vector dimensions and shape
- Download embeddings as CSV files
- Uses the multilingual version of Universal Sentence Encoder
- Simple and intuitive user interface

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

To run the app locally:

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501` by default.

## Usage

1. Enter your text in the text area
2. The app will automatically generate the embedding
3. View the embedding vector and its properties
4. Optionally download the embedding as a CSV file

## Dependencies

- streamlit==1.32.0
- tensorflow-hub==0.15.0
- tensorflow-text==2.15.0
- numpy==1.24.3

## Note

The first time you run the app, it will download the Universal Sentence Encoder model, which might take a few minutes depending on your internet connection. The model is then cached for subsequent uses. 
