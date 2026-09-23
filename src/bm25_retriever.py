import re

from rank_bm25 import BM25Okapi

from vector_store import get_collection


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_documents(technology=None):
    collection = get_collection()
    results = collection.get()

    records = zip(
        results["documents"],
        results["ids"],
        results["metadatas"],
    )

    if technology:
        records = (
            record
            for record in records
            if record[2].get("technology") == technology
        )

    records = list(records)

    return (
        [document for document, _, _ in records],
        [document_id for _, document_id, _ in records],
        [metadata for _, _, metadata in records],
    )


def build_bm25(technology=None):
    documents, ids, metadatas = load_documents(technology)

    if not documents:
        return None, documents, ids, metadatas

    tokenized_documents = [tokenize(document) for document in documents]
    bm25 = BM25Okapi(tokenized_documents)

    return bm25, documents, ids, metadatas


def search_bm25(query, top_k=3, technology=None):
    bm25, documents, ids, metadatas = build_bm25(technology)

    if bm25 is None:
        return []

    scores = bm25.get_scores(tokenize(query))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    return [
        {
            "id": ids[index],
            "document": documents[index],
            "metadata": metadatas[index],
            "score": float(scores[index]),
        }
        for index in ranked_indices[:top_k]
    ]


def main():
    query = input("\nYou: ")
    technology = input(
        "Technology filter (press Enter for none): "
    ).strip() or None

    results = search_bm25(query, top_k=3, technology=technology)

    print("\nBM25 RESULTS")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"ID: {result['id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Metadata: {result['metadata']}")
        print(f"Document: {result['document']}")


if __name__ == "__main__":
    main()
