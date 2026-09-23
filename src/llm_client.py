from ollama import chat
from config import MODEL, SYSTEM_PROMPT


def ask_llm(question: str):

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response["message"]["content"]

    usage = {
        "input_tokens": response.get("prompt_eval_count", 0),
        "output_tokens": response.get("eval_count", 0),
        "total_tokens": (
            response.get("prompt_eval_count", 0)
            + response.get("eval_count", 0)
        ),
    }

    return answer, usage