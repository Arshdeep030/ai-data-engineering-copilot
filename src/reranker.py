from sentence_transformers import CrossEncoder

from hybrid_retriever import hybrid_search


RERANKER_MODEL = "BAAI/bge-reranker-base"
_reranker = None


def get_reranker():
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)

    return _reranker


def rerank(query, candidates, top_k=3):
    if not candidates:
        return []

    pairs = [(query, candidate["document"]) for candidate in candidates]
    scores = get_reranker().predict(pairs)

    reranked = []
    for candidate, score in zip(candidates, scores):
        result = candidate.copy()
        result["reranker_score"] = float(score)
        reranked.append(result)

    reranked.sort(key=lambda result: result["reranker_score"], reverse=True)
    return reranked[:top_k]


def main():
    query = input("\nYou: ")
    technology = input(
        "Technology filter (press Enter for none): "
    ).strip() or None
    candidates = hybrid_search(query, top_k=5, technology=technology)
    results = rerank(query, candidates, top_k=3)

    print("\nRERANKED RESULTS")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"ID: {result['id']}")
        print(f"Reranker Score: {result['reranker_score']:.4f}")
        print(f"Metadata: {result['metadata']}")
        print(f"Document: {result['document']}")


if __name__ == "__main__":
    main()
