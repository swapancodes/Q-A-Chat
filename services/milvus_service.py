import streamlit as st
from pymilvus import MilvusClient
from config import MILVUS_URI, MILVUS_TOKEN


@st.cache_resource
def get_milvus_client():
    return MilvusClient(
        uri=MILVUS_URI,
        token=MILVUS_TOKEN
    )