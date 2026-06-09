def chunk_text_by_size(text, source, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "source": source,
                "chunk_id": chunk_id,
                "title": f"{source} part {chunk_id}",
                "text": chunk
            })

        chunk_id += 1
        start = end - overlap

    return chunks


def chunk_sections_by_heading(text, source, max_section_size=1200, overlap=150):
    chunks = []

    sections = text.split("\n## ")

    for i, section in enumerate(sections):
        section = section.strip()

        if not section:
            continue

        chunk_text = "## " + section if not section.startswith("## ") else section

        lines = chunk_text.splitlines()
        title = lines[0].replace("##", "").strip()

        # If section is short, keep it whole
        if len(chunk_text) <= max_section_size:
            chunks.append({
                "source": source,
                "chunk_id": len(chunks),
                "title": title,
                "text": chunk_text
            })

        # If section is too long, split it further but preserve title
        else:
            subchunks = chunk_text_by_size(
                text=chunk_text,
                source=source,
                chunk_size=max_section_size,
                overlap=overlap
            )

            for subchunk in subchunks:
                subchunk["title"] = title
                chunks.append(subchunk)

    return chunks

def build_chunks(raw_docs):
    all_chunks = []

    for doc in raw_docs:
        source = doc["source"]
        text = doc["text"]

        # If the file contains heading markers, use heading-based chunking
        if "\n## " in text or text.strip().startswith("## "):
            chunks = chunk_sections_by_heading(text, source)

        # If no headings, fall back to size-based chunking
        else:
            chunks = chunk_text_by_size(text, source)

        all_chunks.extend(chunks)

    return all_chunks