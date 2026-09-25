import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from hybrid_retriever import hybrid_search
from llm_client import ask_llm
from reranker import rerank


DATASET_PATH = Path("data/evaluation/rag_eval.json")
RESULTS_PATH = Path("data/evaluation/results/day8_results.json")
TOP_K = 5

GROUNDED_SYSTEM_PROMPT = """
You are an AI Data Engineering Copilot.

Answer the user's question using only the retrieved documentation.

Rules:
- Do not use outside knowledge or add unsupported details.
- Cite source IDs after factual statements.
- If the retrieved context does not contain enough information, say:
  "The retrieved context does not provide enough information to answer
  this question."
- Keep the answer concise and technically accurate.
"""


def load_dataset(path=DATASET_PATH):
    with path.open(encoding="utf-8") as dataset_file:
        test_cases = json.load(dataset_file)

    for index, test_case in enumerate(test_cases, start=1):
        required_fields = {
            "question",
            "relevant_ids",
            "expected_answer",
            "answerable",
        }
        missing_fields = required_fields - test_case.keys()

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Case {index} is missing: {missing}")

    return test_cases


def retrieve_chunks(question):
    candidates = hybrid_search(question, top_k=TOP_K)
    return rerank(question, candidates, top_k=TOP_K)


def build_context(chunks):
    context_parts = []

    for chunk in chunks:
        metadata = chunk["metadata"]
        context_parts.append(
            f"""Source ID: {chunk['id']}
Source: {metadata['source']}
Technology: {metadata['technology']}
Section: {metadata['section']}
Chunk Index: {metadata['chunk_index']}

Content:
{chunk['document']}"""
        )

    return "\n\n---\n\n".join(context_parts)


def generate_grounded_answer(question, chunks):
    context = build_context(chunks)
    prompt = f"""Retrieved documentation:

{context}

Question:
{question}"""

    return ask_llm(prompt, system_prompt=GROUNDED_SYSTEM_PROMPT)


def serialize_chunk(chunk):
    serialized = {
        "id": chunk["id"],
        "document": chunk["document"],
        "metadata": chunk["metadata"],
        "rrf_score": chunk.get("rrf_score"),
        "reranker_score": chunk.get("reranker_score"),
    }

    if "distance" in chunk:
        serialized["dense_distance"] = chunk["distance"]

    if "score" in chunk:
        serialized["bm25_score"] = chunk["score"]

    return serialized


def relevant_positions(retrieved_ids, relevant_ids):
    return {
        chunk_id: retrieved_ids.index(chunk_id) + 1
        for chunk_id in relevant_ids
        if chunk_id in retrieved_ids
    }


def calculate_retrieval_metrics(results):
    answerable_results = [result for result in results if result["answerable"]]

    if not answerable_results:
        return {
            "answerable_case_count": 0,
            "unsupported_case_count": len(results),
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
        }

    def recall_at_k(k):
        per_case_recall = []

        for result in answerable_results:
            relevant_ids = set(result["relevant_ids"])
            retrieved_ids = set(result["retrieved_ids"][:k])
            per_case_recall.append(len(relevant_ids & retrieved_ids) / len(relevant_ids))

        return sum(per_case_recall) / len(per_case_recall)

    reciprocal_ranks = []
    for result in answerable_results:
        positions = result["relevant_positions"].values()
        reciprocal_ranks.append(1 / min(positions) if positions else 0.0)

    return {
        "answerable_case_count": len(answerable_results),
        "unsupported_case_count": len(results) - len(answerable_results),
        "recall_at_1": recall_at_k(1),
        "recall_at_3": recall_at_k(3),
        "recall_at_5": recall_at_k(5),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def evaluate_case(test_case, skip_generation=False):
    started_at = perf_counter()
    chunks = retrieve_chunks(test_case["question"])
    retrieval_latency_seconds = perf_counter() - started_at

    answer = None
    usage = None
    answer_error = None
    generation_latency_seconds = 0.0

    if not skip_generation:
        generation_started_at = perf_counter()
        try:
            answer, usage = generate_grounded_answer(test_case["question"], chunks)
        except Exception as error:
            answer_error = str(error)
        generation_latency_seconds = perf_counter() - generation_started_at

    retrieved_ids = [chunk["id"] for chunk in chunks]

    return {
        "question": test_case["question"],
        "answerable": test_case["answerable"],
        "relevant_ids": test_case["relevant_ids"],
        "expected_answer": test_case["expected_answer"],
        "retrieved_ids": retrieved_ids,
        "relevant_positions": relevant_positions(
            retrieved_ids,
            test_case["relevant_ids"],
        ),
        "retrieved_chunks": [serialize_chunk(chunk) for chunk in chunks],
        "answer": answer,
        "answer_error": answer_error,
        "usage": usage,
        "retrieval_latency_seconds": round(retrieval_latency_seconds, 4),
        "generation_latency_seconds": round(generation_latency_seconds, 4),
        "latency_seconds": round(
            retrieval_latency_seconds + generation_latency_seconds,
            4,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the current Day 7 RAG pipeline."
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Run retrieval metrics without calling the local Qwen3 model.",
    )
    args = parser.parse_args()

    test_cases = load_dataset()
    results = []

    for index, test_case in enumerate(test_cases, start=1):
        print(f"[{index}/{len(test_cases)}] {test_case['question']}")
        results.append(
            evaluate_case(test_case, skip_generation=args.skip_generation)
        )

    metrics = calculate_retrieval_metrics(results)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET_PATH),
        "pipeline": "dense + BM25 + RRF + cross-encoder reranker",
        "top_k": TOP_K,
        "generation_skipped": args.skip_generation,
        "retrieval_metrics": metrics,
        "results": results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\nDAY 8 RETRIEVAL BASELINE")
    print("=" * 70)
    print(f"Recall@1: {metrics['recall_at_1']:.2%}")
    print(f"Recall@3: {metrics['recall_at_3']:.2%}")
    print(f"Recall@5: {metrics['recall_at_5']:.2%}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Saved results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
