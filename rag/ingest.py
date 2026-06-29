import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import *
from services.embeddings import get_embedding_model
from services.milvus_service import get_milvus_client


def ingest_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)

    embedding = get_embedding_model()

    vectors = embedding.embed_documents(
        [c.page_content for c in chunks]
    )

    client = get_milvus_client()

    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=len(vectors[0]),
    )

    data = []

    for chunk, vector in zip(chunks, vectors):
        data.append(
            {
                "id": uuid.uuid4().int & ((1 << 63) - 1),
                "vector": vector,
                "text": chunk.page_content,
            }
        )

    client.insert(
        collection_name=COLLECTION_NAME,
        data=data,
    )