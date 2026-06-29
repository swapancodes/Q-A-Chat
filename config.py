import streamlit as st
GROQ_API_KEY = st.secrets['groq_api_key']

MILVUS_URI = server_uri
MILVUS_TOKEN = server_token

COLLECTION_NAME = "new_collection"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.1-8b-instant"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150