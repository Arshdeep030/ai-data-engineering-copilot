from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "BAAI/bge-base-en-v1.5"


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def retrieve(query, documents, model, top_k=3):

    query_embedding = model.encode(query)

    document_embeddings = model.encode(documents)

    results = []

    for document, embedding in zip(
        documents,
        document_embeddings
    ):
        score = cosine_similarity(
            query_embedding,
            embedding
        )

        results.append({
            "document": document,
            "score": score
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


def calculate_recall_at_k(
    evaluation_data,
    documents,
    model,
    k
):
    correct = 0

    for item in evaluation_data:

        results = retrieve(
            item["query"],
            documents,
            model,
            top_k=k
        )

        retrieved_documents = [
            result["document"]
            for result in results
        ]

        if item["relevant_document"] in retrieved_documents:
            correct += 1

    return correct / len(evaluation_data)


def main():

    model = SentenceTransformer(MODEL_NAME)

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

    evaluation_data = [
        {
            "query": "Why did my Spark executor run out of memory?",
            "relevant_document": "Executor out-of-memory errors can occur when workloads require more memory than the executor has available."
        },
        {
            "query": "What causes executor out-of-memory errors in Spark?",
            "relevant_document": "Executor out-of-memory errors can occur when workloads require more memory than the executor has available."
        },
        {
            "query": "My Spark executor ran out of heap memory. What happened?",
            "relevant_document": "Executor out-of-memory errors can occur when workloads require more memory than the executor has available."
        },
        {
            "query": "How does Spark distribute data between executors?",
            "relevant_document": "Spark shuffle operations redistribute data across executors."
        },
        {
            "query": "How does Spark SQL work?",
            "relevant_document": "Spark SQL allows users to query structured data using SQL."
        },
        {
            "query": "How does Kafka distribute messages?",
            "relevant_document": "Kafka partitions allow messages to be distributed across consumers."
        },
        {
            "query": "What does Delta Lake provide?",
            "relevant_document": "Delta Lake provides ACID transactions and schema enforcement."
        }
    ]

    # ----------------------------------------
    # Retrieval results
    # ----------------------------------------

    for item in evaluation_data:

        query = item["query"]

        results = retrieve(
            query,
            documents,
            model,
            top_k=3
        )

        print("\n" + "=" * 70)
        print("QUERY")
        print("=" * 70)
        print(query)

        print("\nTOP 3 RETRIEVED DOCUMENTS")
        print("=" * 70)

        for i, result in enumerate(results, start=1):

            print(f"\nRank: {i}")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Document: {result['document']}")

    # ----------------------------------------
    # Evaluation
    # ----------------------------------------

    recall_at_1 = calculate_recall_at_k(
        evaluation_data,
        documents,
        model,
        k=1
    )

    recall_at_3 = calculate_recall_at_k(
        evaluation_data,
        documents,
        model,
        k=3
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    print(f"Recall@1: {recall_at_1:.2%}")
    print(f"Recall@3: {recall_at_3:.2%}")


if __name__ == "__main__":
    main()