from chunker import chunk_document
from document_loader import load_markdown_files
from vector_store import add_documents


def get_technology(source):
    source = source.lower()

    for technology in ("spark", "kafka", "airflow", "fabric", "databricks"):
        if technology in source:
            return technology

    return "unknown"


def main():
    documents = load_markdown_files()
    all_chunks = []

    for document in documents:
        source = document["source"]
        technology = get_technology(source)

        chunks = chunk_document(document["text"], source)

        for chunk in chunks:
            chunk["technology"] = technology
            all_chunks.append(chunk)

    if not all_chunks:
        print("No documents found.")
        return

    documents_to_store = [chunk["text"] for chunk in all_chunks]
    ids_to_store = [chunk["id"] for chunk in all_chunks]
    metadata_to_store = [
        {
            "source": chunk["source"],
            "technology": chunk["technology"],
            "section": chunk["section"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in all_chunks
    ]

    add_documents(
        documents=documents_to_store,
        metadatas=metadata_to_store,
        ids=ids_to_store,
    )

    print(f"\nSuccessfully ingested {len(all_chunks)} chunks.")


if __name__ == "__main__":
    main()
