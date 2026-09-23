from bm25_retriever import search_bm25
from vector_store import search


def dense_search(query, top_k=5, technology=None):
    results = search(query, top_k=top_k, technology=technology)

    return [
        {
            "id": document_id,
            "document": document,
            "metadata": metadata,
            "rank": rank,
            "distance": float(distance),
        }
        for rank, (document_id, document, distance, metadata) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            ),
            start=1,
        )
    ]


def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    scores = {}
    documents = {}

    for result in dense_results:
        document_id = result["id"]
        scores[document_id] = scores.get(document_id, 0) + 1 / (
            k + result["rank"]
        )
        documents[document_id] = result

    for rank, result in enumerate(bm25_results, start=1):
        document_id = result["id"]
        scores[document_id] = scores.get(document_id, 0) + 1 / (k + rank)
        documents.setdefault(document_id, result)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)

    fused_results = []
    for document_id in ranked_ids:
        result = documents[document_id].copy()
        result["rrf_score"] = scores[document_id]
        fused_results.append(result)

    return fused_results


def hybrid_search(query, top_k=3, technology=None, candidate_k=5):
    dense_results = dense_search(
        query,
        top_k=candidate_k,
        technology=technology,
    )
    bm25_results = search_bm25(
        query,
        top_k=candidate_k,
        technology=technology,
    )

    return reciprocal_rank_fusion(dense_results, bm25_results)[:top_k]


def main():
    query = input("\nYou: ")
    technology = input(
        "Technology filter (press Enter for none): "
    ).strip() or None
    results = hybrid_search(query, top_k=3, technology=technology)

    print("\nHYBRID RESULTS")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"ID: {result['id']}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Metadata: {result['metadata']}")
        print(f"Document: {result['document']}")


if __name__ == "__main__":
    main()
