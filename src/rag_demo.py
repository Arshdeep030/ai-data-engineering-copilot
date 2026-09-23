from ollama import chat

from hybrid_retriever import hybrid_search
from reranker import rerank


LLM_MODEL = "qwen3:8b"


def generate_answer(query, results):

    context_parts = []

    for result in results:
        context_parts.append(
            f"""
Source ID: {result['id']}
Source: {result['metadata']['source']}
Technology: {result['metadata']['technology']}
Section: {result['metadata']['section']}
Chunk Index: {result['metadata']['chunk_index']}

Content:
{result['document']}
"""
        )

    context = "\n".join(context_parts)

    system_prompt = """
You are an AI Data Engineering Copilot.

Answer the user's question using ONLY the information explicitly
supported by the retrieved documentation.

Rules:
- Do not use outside knowledge.
- Do not guess or add unsupported technical details.
- Every factual statement must be supported by retrieved documentation.
- Cite the source ID after factual statements and include the source
  section when useful.
- If the retrieved context does not contain enough information, say:
  "The retrieved context does not provide enough information to answer
  this question."
- Keep the answer concise and technically accurate.
"""

    user_prompt = f"""
Context:

{context}

Question:

{query}
"""

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response["message"]["content"]


def main():

    query = input("\nYou: ")
    technology = input(
        "Technology filter (press Enter for none): "
    ).strip()

    if technology == "":
        technology = None

    candidates = hybrid_search(
        query,
        top_k=5,
        technology=technology
    )

    results = rerank(query, candidates, top_k=3)

    if not results:
        print("\nNo chunks matched the retrieval filter.")
        return

    print("\nRETRIEVED CONTEXT")
    print("=" * 60)

    for i, result in enumerate(results, start=1):
        print(f"\nRank {i}")
        print(f"ID: {result['id']}")
        print(f"RRF score: {result['rrf_score']:.6f}")
        print(f"Reranker score: {result['reranker_score']:.4f}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Section: {result['metadata']['section']}")
        print(f"Document: {result['document']}")

    answer = generate_answer(
        query,
        results
    )

    print("\nCOPILOT ANSWER")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
