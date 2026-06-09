import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from src.loader import load_text_files
from src.chunker import build_chunks

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

raw_docs = load_text_files("mc_knowledge")
chunks = build_chunks(raw_docs)

chunk_texts = [
    f"Title: {chunk.get('title', '')}\nSource: {chunk['source']}\n{chunk['text']}"
    for chunk in chunks
]

chunk_embeddings = embedding_model.encode(chunk_texts)

os.makedirs("vector_store", exist_ok=True)

with open("vector_store/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

np.save("vector_store/embeddings.npy", chunk_embeddings)

print("Index built successfully.")
print("Chunks:", len(chunks))
print("Embeddings shape:", chunk_embeddings.shape)