import pickle
import numpy as np

def load_index():
    with open("vector_store/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    chunk_embeddings = np.load("vector_store/embeddings.npy")

    return chunks, chunk_embeddings