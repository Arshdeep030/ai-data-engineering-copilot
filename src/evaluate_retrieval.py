from vector_store import search


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


def evaluate_recall_at_k(k):
    correct = 0

    print("\n" + "=" * 70)
    print(f"RECALL@{k}")
    print("=" * 70)

    for test_case in TEST_CASES:

        query = test_case["query"]
        expected_id = test_case["expected_id"]

        results = search(
            query,
            top_k=k
        )

        retrieved_ids = results["ids"][0]

        is_correct = expected_id in retrieved_ids

        if is_correct:
            correct += 1

        print("\nQuery:")
        print(query)

        print(f"Expected: {expected_id}")
        print(f"Retrieved: {retrieved_ids}")

        print(
            f"Result: {'PASS' if is_correct else 'FAIL'}"
        )

    recall = correct / len(TEST_CASES)

    print("\n" + "-" * 70)
    print(
        f"Recall@{k}: "
        f"{recall * 100:.2f}%"
    )

    return recall


def main():

    recall_at_1 = evaluate_recall_at_k(1)
    recall_at_3 = evaluate_recall_at_k(3)
    recall_at_5 = evaluate_recall_at_k(5)

    print("\n" + "=" * 70)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 70)

    print(
        f"Recall@1: {recall_at_1 * 100:.2f}%"
    )

    print(
        f"Recall@3: {recall_at_3 * 100:.2f}%"
    )

    print(
        f"Recall@5: {recall_at_5 * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
