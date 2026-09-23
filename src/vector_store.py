import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "data_engineering_docs"

CHROMA_PATH = "data/chroma"


# Load the embedding model once when this module is imported.
_embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


def get_embedding_model():
    return _embedding_model


def add_documents(documents, metadatas=None, ids=None):
    if not documents:
        return

    if metadatas is not None and len(metadatas) != len(documents):
        raise ValueError("Each document must have matching metadata.")

    collection = get_collection()
    model = get_embedding_model()

    if ids is None:
        ids = [f"doc_{i}" for i in range(len(documents))]

    if len(ids) != len(documents):
        raise ValueError("Each document must have a matching ID.")

    embeddings = model.encode(documents).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Stored {len(documents)} documents.")


def search(query, top_k=3, technology=None):
    collection = get_collection()
    model = get_embedding_model()

    query_embedding = model.encode(query).tolist()

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k
    }

    if technology:
        query_kwargs["where"] = {
            "technology": technology
        }

    results = collection.query(**query_kwargs)

    return results
