from vector_store import search


def main():
    query = input("\nYou: ")

    technology = input(
        "Technology filter (press Enter for none): "
    ).strip()

    if technology == "":
        technology = None

    results = search(
        query,
        top_k=3,
        technology=technology,
    )

    print("\nRETRIEVED CHUNKS")
    print("=" * 60)

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]

    for i, (
        document,
        distance,
        metadata,
        document_id
    ) in enumerate(
        zip(
            documents,
            distances,
            metadatas,
            ids
        ),
        start=1
    ):
        print(f"\nRank {i}")
        print(f"ID: {document_id}")
        print(f"Distance: {distance:.4f}")
        print(f"Metadata: {metadata}")
        print(f"Document: {document}")


if __name__ == "__main__":
    main()
