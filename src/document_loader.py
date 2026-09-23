from pathlib import Path


DOCS_DIR = Path("data/raw/docs")


def load_markdown_files():
    documents = []

    for file_path in DOCS_DIR.glob("*.md"):
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "source": file_path.name,
                "text": text,
            }
        )

    return documents


def main():
    documents = load_markdown_files()

    print(f"Loaded {len(documents)} documents.")

    for document in documents:
        print("\n" + "=" * 60)
        print(f"Source: {document['source']}")
        print("=" * 60)
        print(document["text"][:500])


if __name__ == "__main__":
    main()