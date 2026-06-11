# src/retriever.py

import re
import numpy as np
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity


STOPWORDS = {
    "a", "an", "the", "of", "to", "do", "does", "i", "you", "in", "on",
    "for", "with", "and", "or", "is", "are", "can", "how", "what",
    "where", "when", "why", "make", "craft", "build", "create", "find",
    "get", "use", "using", "crafting", "making", "building", "built",
    "minecraft", "game"
}


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_important_words(text):
    text = normalize_text(text)
    words = text.split()

    important_words = [
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    ]

    return important_words


def title_word_coverage(question_words, title_words, word_threshold=80):
    """
    Measures how many important title words are matched by the question.

    Example:
    question: "how to build nether portal"
    title: "Nether Portal"
    coverage = 2/2

    question: "how to build nether portal"
    title: "Nether Wastes"
    coverage = 1/2
    """

    if not title_words:
        return 0

    matched = 0

    for title_word in title_words:
        best_word_score = 0

        for question_word in question_words:
            score = fuzz.ratio(question_word, title_word)

            if score > best_word_score:
                best_word_score = score

        if best_word_score >= word_threshold:
            matched += 1

    return matched / len(title_words)


def fuzzy_title_match(question, chunks, threshold=80, coverage_threshold=0.75):
    """
    Try to find a chunk whose title closely matches the user's question.

    This avoids bad matches like:
    "how to build nether portal" -> "Nether Wastes"

    because "Nether Wastes" only matches one important title word.
    """

    question_normalized = normalize_text(question)
    question_words = get_important_words(question)

    best_chunk = None
    best_score = 0

    for chunk in chunks:
        title = chunk.get("title", "")

        if not title:
            continue

        title_normalized = normalize_text(title)
        title_words = get_important_words(title)

        coverage = title_word_coverage(
            question_words=question_words,
            title_words=title_words,
            word_threshold=80
        )

        # Reject titles where not enough important title words match
        if coverage < coverage_threshold:
            continue

        score_1 = fuzz.partial_ratio(question_normalized, title_normalized)
        score_2 = fuzz.token_set_ratio(question_normalized, title_normalized)
        score_3 = fuzz.WRatio(question_normalized, title_normalized)

        fuzzy_score = max(score_1, score_2, score_3)

        final_score = (0.7 * fuzzy_score) + (0.3 * coverage * 100)

        if final_score > best_score:
            best_score = final_score
            best_chunk = chunk

    if best_chunk is None:
        return None, None

    if best_score < threshold:
        return None, None

    return best_chunk, best_score


def build_context_from_chunks(retrieved_chunks):
    """
    Convert retrieved chunk dictionaries into one context string
    that will be sent to the LLM.
    """

    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"Source: {chunk['source']}\n"
            f"Title: {chunk.get('title', 'Unknown')}\n"
            f"Content:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    return context


def retrieve_chunks(question, chunks, chunk_embeddings, embedding_model, top_k=3):
    """
    Normal embedding-based retrieval.

    This is used when fuzzy title matching does not find
    a confident title match.
    """

    question_embedding = embedding_model.encode([question])

    similarities = cosine_similarity(question_embedding, chunk_embeddings)
    scores = similarities[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved_chunks = [chunks[idx] for idx in top_indices]
    retrieved_scores = scores[top_indices]

    context = build_context_from_chunks(retrieved_chunks)

    return context, retrieved_chunks, retrieved_scores