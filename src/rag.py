from src.retriever import (
    fuzzy_title_match,
    retrieve_chunks,
    build_context_from_chunks
)

from src.generator import generate_answer

def should_use_fuzzy_title_match(question):
    question_lower = question.lower()

    broad_words = [
        "summary",
        "summarize",
        "history",
        "overview",
        "explain",
        "tell me about",
        "what are",
        "what is minecraft",
        "introduction"
    ]

    if any(word in question_lower for word in broad_words):
        return False

    return True


def ask_rag(question, chunks, chunk_embeddings, embedding_model, tokenizer, model):
    use_fuzzy = should_use_fuzzy_title_match(question)

    matched_chunk = None
    title_score = None

    if use_fuzzy:
        matched_chunk, title_score = fuzzy_title_match(
            question=question,
            chunks=chunks,
            threshold=80
        )

    if matched_chunk is not None:
        retrieved_chunks = [matched_chunk]
        scores = [title_score / 100]
        context = build_context_from_chunks(retrieved_chunks)
        retrieval_method = "fuzzy_title_match"

    else:
        context, retrieved_chunks, scores = retrieve_chunks(
            question=question,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            embedding_model=embedding_model,
            top_k=3
        )
        retrieval_method = "embedding_similarity"

    answer = generate_answer(
        question=question,
        context=context,
        tokenizer=tokenizer,
        model=model
    )

    return answer, retrieved_chunks, scores, retrieval_method