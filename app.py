import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient

#st.title("Document Upload and Store in Milvus")

uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt", "csv", "xlsx", "xls", "md", "html"],
    accept_multiple_files=True,
)

# ---------------- Loader Function ---------------- #
@st.cache_data
def load_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyMuPDFLoader(temp_path)

        elif suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(temp_path)

        elif suffix == ".txt":
            loader = TextLoader(temp_path, encoding="utf-8")

        elif suffix == ".csv":
            loader = CSVLoader(temp_path)

        elif suffix in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(temp_path)

        elif suffix == ".html":
            loader = UnstructuredHTMLLoader(temp_path)

        elif suffix == ".md":
            loader = UnstructuredMarkdownLoader(temp_path)

        else:
            st.warning(f"Unsupported file: {uploaded_file.name}")
            return []

        docs = loader.load()

    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {e}")
        docs = []

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return docs


# ---------------- Process Uploaded Files ---------------- #

if uploaded_files:

    all_docs = []

    for file in uploaded_files:
        docs = load_uploaded_file(file)
        all_docs.extend(docs)

    if len(all_docs) == 0:
        st.warning("No documents could be loaded.")
        st.stop()

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(all_docs)

    texts = [chunk.page_content for chunk in chunks]

    # Embedding model
    @st.cache_resource
    def load_embeddings():
        return HuggingFaceEmbeddings(
             model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    embeddings = load_embeddings()
    vectors = embeddings.embed_documents(texts)

    # Connect Milvus
    @st.cache_resource
    def get_client():
       return MilvusClient(
          uri=st.secrets["MILVUS_URI"],
          token=st.secrets["MILVUS_TOKEN"]
       )
    client = get_client()

    collection_name = "new_collection"

    if client.has_collection(collection_name):
       client.drop_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        dimension=len(vectors[0]),
    )

    data = []
    import uuid
    for chunk, vector in zip(chunks, vectors):
        data.append(
            {
                "id": uuid.uuid4().int & ((1 << 63) - 1),
                "vector": vector,
                "text": chunk.page_content,
            }
        )

    client.insert(
        collection_name=collection_name,
        data=data,
    )

    #st.success(f"Inserted {len(data)} document chunks into Milvus.")
    st.success('Documents Uploaded Successfully')

else:
    st.info("Please upload one or more documents.")


#------------Streamlit Chat UI (app.py)----------
from rag_engine import ask_question
st.title("📚 PDF RAG Chatbot")

question = st.text_input("Ask a question")

if st.button("Submit") and question:
    answer = ask_question(question)
    st.write(answer)