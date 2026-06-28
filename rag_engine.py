from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pymilvus import MilvusClient
import streamlit as st

api_key = st.secrets["GROQ_API_KEY"]

if not api_key:
    raise ValueError("GROQ_API_KEY not found")

# ---------------------------
# Embedding Model
# ---------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# Milvus Connection
# ---------------------------
@st.cache_resource
def get_milvus_client():
    return MilvusClient(
       uri=st.secrets["MILVUS_URI"],
       token=st.secrets["MILVUS_TOKEN"]
    )

client = get_milvus_client()

# ---------------------------
# Groq LLM
# ---------------------------
llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant",
    temperature=0,
    max_retries=2
)

# ---------------------------
# Prompt Template
# ---------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for question-answering tasks.

Use the retrieved context below to answer the question.

If the answer is not present in the context, reply:
"I don't know."

Context:
{context}
"""
        ),
        ("human", "{question}")
    ]
)

# ---------------------------
# Chain
# ---------------------------
chain = prompt | llm | StrOutputParser()

# ---------------------------
# QA Function
# ---------------------------
def ask_question(question: str):

    # Create query embedding
    query_vector = embeddings.embed_query(question)

    # Search Milvus
    results = client.search(
        collection_name="new_collection",
        data=[query_vector],
        limit=5,
        output_fields=["text"]
    )

    # Extract retrieved text
    retrieved_docs = []

    for hit in results[0]:
        if "entity" in hit:
            retrieved_docs.append(hit["entity"]["text"])
        else:
            retrieved_docs.append(hit["text"])

    context = "\n\n".join(retrieved_docs)

    # Generate answer
    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return response