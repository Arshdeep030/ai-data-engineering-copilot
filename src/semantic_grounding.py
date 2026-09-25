from functools import lru_cache

from transformers import pipeline


MODEL_NAME = "cross-encoder/nli-deberta-v3-base"


@lru_cache(maxsize=1)
def get_nli_pipeline():
    """
    Load the NLI model once and reuse it.
    """

    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )


def evaluate_claim(premise, hypothesis):
    """
    Evaluate the relationship between retrieved evidence
    and an answer claim.

    Returns:
        label:
            entailment
            contradiction
            neutral

        score:
            model confidence
    """

    classifier = get_nli_pipeline()

    result = classifier(
        premise,
        text_pair=hypothesis,
    )
    if isinstance(result, list):
        result = result[0]

    return {
        "label": result["label"].lower(),
        "score": float(result["score"]),
    }


def find_best_evidence(hypothesis, retrieved_chunks):
    """
    Compare one answer sentence against every retrieved chunk.

    The chunk with the highest entailment score is selected
    as the strongest evidence.

    If no chunk is strongly entailing, we still retain the
    highest-confidence NLI result for inspection.
    """

    evidence_results = []

    for chunk in retrieved_chunks:

        document = chunk.get("document", "")

        if not document:
            continue

        result = evaluate_claim(
            premise=document,
            hypothesis=hypothesis,
        )

        evidence_results.append(
            {
                "chunk_id": chunk.get("id"),
                "document": document,
                "label": result["label"],
                "score": result["score"],
            }
        )

    if not evidence_results:
        return {
            "best_label": "unknown",
            "best_score": 0.0,
            "best_chunk_id": None,
            "evidence": [],
        }

    # Prefer entailment when selecting evidence.
    entailments = [
        result
        for result in evidence_results
        if result["label"] == "entailment"
    ]

    if entailments:
        best = max(
            entailments,
            key=lambda result: result["score"],
        )
    else:
        # If nothing entails the claim, retain the strongest
        # NLI result for diagnostics.
        best = max(
            evidence_results,
            key=lambda result: result["score"],
        )

    return {
        "best_label": best["label"],
        "best_score": round(best["score"], 4),
        "best_chunk_id": best["chunk_id"],
        "evidence": evidence_results,
    }
