import os
import httpx

try:
    from config import MODEL, SYSTEM_PROMPT
except ImportError:
    from .config import MODEL, SYSTEM_PROMPT


def ask_llm(question: str, system_prompt: str = SYSTEM_PROMPT):
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Cloud LLM Provider (Groq or OpenAI-compatible) for 24/7 cloud deployments
    if groq_api_key or openai_api_key:
        api_key = groq_api_key or openai_api_key
        base_url = (
            "https://api.groq.com/openai/v1"
            if groq_api_key
            else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        cloud_model = os.getenv(
            "CLOUD_LLM_MODEL",
            "llama-3.1-8b-instant" if groq_api_key else "gpt-4o-mini",
        )

        payload = {
            "model": cloud_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": 0.1,
            "max_tokens": 400,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]["message"]
            answer = choice.get("content", "")
            usage_data = data.get("usage", {})
            usage = {
                "input_tokens": usage_data.get("prompt_tokens", 0),
                "output_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }
            return answer, usage

    # Default fallback: Local Ollama (qwen3:8b)
    from ollama import chat

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        options={
            "temperature": 0.1,
            "num_predict": 300,
        },
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

