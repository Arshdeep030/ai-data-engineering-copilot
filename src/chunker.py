import re


def create_chunk_id(source, section, chunk_index):
    source_name = source.rsplit(".", 1)[0]

    safe_source = re.sub(r"[^a-z0-9]+", "_", source_name.lower()).strip("_")
    safe_section = re.sub(r"[^a-z0-9]+", "_", section.lower()).strip("_")

    return f"{safe_source}_{safe_section}_{chunk_index:03d}"


def parse_markdown_sections(text):
    sections = []

    current_title = "Introduction"
    current_content = []

    for line in text.splitlines():

        # Detect Markdown headings
        match = re.match(r"^#{1,6}\s+(.+)$", line)

        if match:
            # Save previous section
            if current_content:
                sections.append(
                    {
                        "section": current_title,
                        "text": "\n".join(current_content).strip()
                    }
                )

            current_title = match.group(1).strip()
            current_content = []

        else:
            if line.strip():
                current_content.append(line)

    # Save final section
    if current_content:
        sections.append(
            {
                "section": current_title,
                "text": "\n".join(current_content).strip()
            }
        )

    return sections


def chunk_text(text, chunk_size=100, chunk_overlap=20):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


def chunk_document(text, source):

    sections = parse_markdown_sections(text)

    chunks = []

    for section in sections:

        section_chunks = chunk_text(
            section["text"],
            chunk_size=100,
            chunk_overlap=20
        )

        for chunk_index, chunk in enumerate(section_chunks):

            chunk_id = create_chunk_id(
                source,
                section["section"],
                chunk_index
            )

            chunks.append(
                {
                    "id": chunk_id,
                    "source": source,
                    "section": section["section"],
                    "chunk_index": chunk_index,
                    "text": chunk
                }
            )

    return chunks


def main():

    from document_loader import load_markdown_files

    documents = load_markdown_files()

    for document in documents:

        chunks = chunk_document(
            document["text"],
            document["source"]
        )

        print("\n" + "=" * 70)
        print(f"DOCUMENT: {document['source']}")
        print("=" * 70)

        for i, chunk in enumerate(chunks):

            print(f"\nCHUNK {i}")
            print("-" * 70)

            print(f"ID: {chunk['id']}")
            print(f"Source: {chunk['source']}")
            print(f"Section: {chunk['section']}")
            print(f"Chunk index: {chunk['chunk_index']}")
            print(f"Text: {chunk['text']}")


if __name__ == "__main__":
    main()
