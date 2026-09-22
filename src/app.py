from llm_client import ask_llm


def main():

    print("======================================")
    print(" AI Data Engineering Copilot - v0")
    print("======================================")
    print("Running locally with Ollama.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        if not question.strip():
            continue

        answer = ask_llm(question)

        print("\nCopilot:")
        print(answer)
        print()


if __name__ == "__main__":
    main()