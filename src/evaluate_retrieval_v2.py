from bm25_retriever import search_bm25
from hybrid_retriever import dense_search, hybrid_search
from reranker import rerank


TEST_CASES = [
    {
        "query": "Why did my Spark executor run out of memory?",
        "expected_id": "spark_executor_out_of_memory_errors_000",
    },
    {
        "query": "What causes executor out-of-memory errors in Spark?",
        "expected_id": "spark_executor_out_of_memory_errors_000",
    },
    {
        "query": "How does Spark distribute data between executors?",
        "expected_id": "spark_spark_shuffle_000",
    },
    {
        "query": "How does Spark SQL work?",
        "expected_id": "spark_spark_sql_000",
    },
]


def retrieve_dense(query):
    return dense_search(query, top_k=5)


def retrieve_bm25(query):
    return search_bm25(query, top_k=5)


def retrieve_hybrid(query):
    return hybrid_search(query, top_k=5)


def retrieve_hybrid_reranked(query):
    candidates = hybrid_search(query, top_k=5)
    return rerank(query, candidates, top_k=5)


def evaluate_method(retrieve):
    ranks = []

    for test_case in TEST_CASES:
        retrieved_ids = [
            result["id"] for result in retrieve(test_case["query"])
        ]

        try:
            rank = retrieved_ids.index(test_case["expected_id"]) + 1
        except ValueError:
            rank = None

        ranks.append(rank)

    total = len(TEST_CASES)
    return {
        "recall_at_1": sum(rank == 1 for rank in ranks) / total,
        "recall_at_3": sum(
            rank is not None and rank <= 3 for rank in ranks
        ) / total,
        "recall_at_5": sum(
            rank is not None and rank <= 5 for rank in ranks
        ) / total,
        "mrr": sum(1 / rank if rank else 0 for rank in ranks) / total,
        "ranks": ranks,
    }


def main():
    methods = {
        "Dense": retrieve_dense,
        "BM25": retrieve_bm25,
        "Hybrid (RRF)": retrieve_hybrid,
        "Hybrid + Reranker": retrieve_hybrid_reranked,
    }

    print("\nRETRIEVAL EVALUATION: DAY 7")
    print("=" * 78)
    print(f"{'Method':<22} {'Recall@1':>10} {'Recall@3':>10} {'Recall@5':>10} {'MRR':>10}")
    print("-" * 78)

    for name, retrieve in methods.items():
        metrics = evaluate_method(retrieve)
        print(
            f"{name:<22} "
            f"{metrics['recall_at_1']:>9.2%} "
            f"{metrics['recall_at_3']:>9.2%} "
            f"{metrics['recall_at_5']:>9.2%} "
            f"{metrics['mrr']:>10.4f}"
        )
        print(f"  Relevant chunk ranks: {metrics['ranks']}")


if __name__ == "__main__":
    main()
