from vector_store import get_collection


def main():
    collection = get_collection()

    print("Collection:", collection.name)
    print("Document count:", collection.count())

    results = collection.get()

    print("\nStored documents:\n")

    for document_id, document, metadata in zip(
        results["ids"],
        results["documents"],
        results["metadatas"]
    ):
        print("ID:", document_id)
        print("Document:", document)
        print("Metadata:", metadata)
        print("-" * 60)


if __name__ == "__main__":
    main()