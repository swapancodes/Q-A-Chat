from services.embeddings import get_embedding_model
from services.milvus_service import get_milvus_client
from config import COLLECTION_NAME


def retrieve(question):

    embedding = get_embedding_model()

    client = get_milvus_client()

    vector = embedding.embed_query(question)

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[vector],
        limit=5,
        output_fields=["text"]
    )

    docs = []

    for hit in results[0]:
        docs.append(hit["entity"]["text"])

    return "\n\n".join(docs)