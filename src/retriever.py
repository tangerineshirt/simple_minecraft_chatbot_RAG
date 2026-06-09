from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def retrieve_chunks(question, chunks, chunk_embeddings, embedding_model, top_k=3):
    question_embedding = embedding_model.encode([question])

    similarities = cosine_similarity(question_embedding, chunk_embeddings)
    scores = similarities[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved_chunks = [chunks[idx] for idx in top_indices]
    retrieved_scores = scores[top_indices]

    context_parts = []

    for chunk, score in zip(retrieved_chunks, retrieved_scores):
        context_parts.append(
            f"Source: {chunk['source']}\n"
            f"Title: {chunk.get('title', 'Unknown')}\n"
            f"Content:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    return context, retrieved_chunks, retrieved_scores