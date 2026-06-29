import streamlit as st

from utils.loaders import load_uploaded_file
from rag.ingest import ingest_documents
from rag.qa import ask_question

st.title("📚 Document RAG Chatbot")

uploaded_files = st.file_uploader(
    "Upload Documents",
    accept_multiple_files=True,
    type=["pdf","docx","txt","csv","xlsx","xls","md","html",],
)

if uploaded_files:

    docs = []

    for file in uploaded_files:
        docs.extend(load_uploaded_file(file))

    if st.button("Upload"):

        with st.spinner("Processing..."):
            ingest_documents(docs)

        st.success("Documents uploaded successfully.")

st.divider()

question = st.text_input("Ask a question")

if st.button("Submit"):

    with st.spinner("Thinking..."):
        answer = ask_question(question)

    st.write(answer)