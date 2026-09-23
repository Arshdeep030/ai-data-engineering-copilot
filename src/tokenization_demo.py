from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-8B"


def main():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    texts = [
        "Spark",
        "Spark SQL",
        "Spark is fast",
        "data engineering",
        "executor",
        "executor memory",
        "Apache Spark is a distributed computing engine.",
    ]

    for text in texts:

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        print("=" * 60)
        print(f"Text: {text}")
        print(f"Token count: {len(token_ids)}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")


if __name__ == "__main__":
    main()