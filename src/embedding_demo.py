from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def main():

    model = SentenceTransformer(MODEL_NAME)

    query = "Why did my Spark executor run out of memory?"

    documents = [
        "Spark executors perform tasks and store data in memory.",
        "Executor out-of-memory errors can occur when workloads require more memory than the executor has available.",
        "Spark SQL allows users to query structured data using SQL.",
        "Spark shuffle operations redistribute data across executors.",
        "Kafka partitions allow messages to be distributed across consumers.",
        "Delta Lake provides ACID transactions and schema enforcement.",
        "Microsoft Fabric provides an integrated analytics platform.",
        "Airflow DAGs define workflows as directed acyclic graphs.",
    ]
    query_embedding = model.encode(query)

    document_embeddings = model.encode(documents)

    results = []

    for document, embedding in zip(
        documents,
        document_embeddings
    ):
        similarity = cosine_similarity(
            query_embedding,
            embedding
        )

        results.append(
            (document, similarity)
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("=" * 70)
    print("Query:")
    print(query)

    print("\nSemantic Search Results:")

    for document, similarity in results:
        print(f"\nSimilarity: {similarity:.4f}")
        print(f"Document: {document}")


if __name__ == "__main__":
    main()