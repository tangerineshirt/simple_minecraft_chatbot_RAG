# ⛏️ Minecraft RAG Chatbot

A local Minecraft question-answering chatbot built with **Retrieval-Augmented Generation (RAG)**.

This project uses local `.txt` knowledge files, converts them into chunks, embeds those chunks using Sentence Transformers, retrieves the most relevant chunks based on the user's question, and generates an answer using a Hugging Face language model.

The chatbot can answer Minecraft-related questions such as crafting recipes, mobs, items, and other knowledge stored in the local knowledge base.

---

## Features

- Local document-based RAG system
- Supports `.txt` Minecraft knowledge files
- Heading-based chunking using `##` sections
- Fallback chunking for normal unstructured text
- Sentence embedding using `sentence-transformers/all-MiniLM-L6-v2`
- Cosine similarity retrieval
- Hugging Face LLM answer generation
- Terminal chatbot interface
- Streamlit web chatbot interface
- Retrieved source display for debugging and transparency
- Persistent saved index using `.pkl` and `.npy` files

---

## Project Structure

```text
minecraft-rag-chatbot/
├── terminal_app.py
├── streamlit_app.py
├── build_index.py
├── requirements.txt
├── README.md
├── mc_knowledge/
│   ├── minecraft_crafting_recipes_heading_format.txt
│   └── minecraft_mobs_heading_format.txt
├── vector_store/
│   ├── chunks.pkl
│   └── embeddings.npy
└── src/
    ├── __init__.py
    ├── loader.py
    ├── chunker.py
    ├── storage.py
    ├── retriever.py
    ├── generator.py
    └── rag.py
