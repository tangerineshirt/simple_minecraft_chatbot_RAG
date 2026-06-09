from src.retriever import retrieve_chunks
from src.generator import generate_answer


def ask_rag(question, chunks, chunk_embeddings, embedding_model, tokenizer, model):
    top_k = 1 if is_recipe_question(question) else 3

    context, retrieved_chunks, scores = retrieve_chunks(
        question=question,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        embedding_model=embedding_model,
        top_k=top_k
    )

    answer = generate_answer(
        question=question,
        context=context,
        tokenizer=tokenizer,
        model=model
    )

    return answer, retrieved_chunks, scores


def is_recipe_question(question):
    recipe_keywords = [
        "craft",
        "recipe",
        "make",
        "create",
        "how do i make",
        "how to make",
        "how do i craft",
        "how to craft"
    ]

    question_lower = question.lower()

    return any(keyword in question_lower for keyword in recipe_keywords)