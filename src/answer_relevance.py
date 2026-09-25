from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.reranker import get_reranker


EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def calculate_embedding_relevance(question, answer):
    """
    Calculate cosine similarity between the question
    and generated answer embeddings.
    """

    model = get_embedding_model()

    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    answer_embedding = model.encode(
        answer,
        normalize_embeddings=True,
    )

    similarity = float(
        np.dot(
            question_embedding,
            answer_embedding,
        )
    )

    return similarity


def calculate_cross_encoder_relevance(question, answer):
    """
    Score question-answer relevance using the
    existing cross-encoder.
    """

    reranker = get_reranker()

    score = float(
        reranker.predict(
            [(question, answer)]
        )[0]
    )

    return score


def evaluate_relevance(question, answer):
    embedding_score = calculate_embedding_relevance(
        question,
        answer,
    )

    cross_encoder_score = calculate_cross_encoder_relevance(
        question,
        answer,
    )

    return {
        "embedding_cosine_similarity": round(
            embedding_score,
            4,
        ),
        "cross_encoder_score": round(
            cross_encoder_score,
            4,
        ),
    }
