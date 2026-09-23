from llm_client import ask_llm


def main():

    print("======================================")
    print(" AI Data Engineering Copilot - v0.1")
    print("======================================")
    print("Running locally with Ollama.")
    print("Token tracking enabled.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        if not question.strip():
            continue

        answer, usage = ask_llm(question)

        print("\nCopilot:")
        print(answer)

        print("\n--------------------------------------")
        print("Token Usage")
        print("--------------------------------------")
        print(f"Input tokens:  {usage['input_tokens']}")
        print(f"Output tokens: {usage['output_tokens']}")
        print(f"Total tokens:  {usage['total_tokens']}")
        print("--------------------------------------\n")


if __name__ == "__main__":
    main()