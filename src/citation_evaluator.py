import re


def extract_citation_ids(answer):
    """
    Extract chunk IDs from generated answer text.
    Matches IDs formatted like: spark_spark_executors_000, spark_apache_spark_000, etc.
    """
    return set(
        re.findall(
            r"\b(?:spark|[a-zA-Z]+)_[a-zA-Z0-9_]+_\d{3}\b",
            answer,
        )
    )


def evaluate_citations(answer, retrieved_ids, relevant_ids):
    """
    Evaluate citation precision (cited IDs in retrieved context)
    and citation recall (relevant reference IDs cited).
    """
    cited_ids = extract_citation_ids(answer)

    retrieved_ids = set(retrieved_ids)
    relevant_ids = set(relevant_ids)

    valid_citations = cited_ids.intersection(retrieved_ids)
    relevant_citations = cited_ids.intersection(relevant_ids)

    precision = (
        len(valid_citations) / len(cited_ids)
        if cited_ids
        else (1.0 if not relevant_ids else 0.0)
    )

    recall = (
        len(relevant_citations) / len(relevant_ids)
        if relevant_ids
        else (1.0 if not cited_ids else 0.0)
    )

    return {
        "cited_ids": sorted(cited_ids),
        "valid_cited_ids": sorted(valid_citations),
        "relevant_cited_ids": sorted(relevant_citations),
        "citation_precision": round(precision, 4),
        "citation_recall": round(recall, 4),
    }
