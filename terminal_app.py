from src.rag import ask_rag
from src.storage import load_index
from src.generator import load_llm
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunks, chunk_embeddings = load_index()

tokenizer, model = load_llm()

while True:
    question = input("User: ")

    if question.lower().strip() in ["exit", "quit", "q"]:
        print("Bot: Goodbye!")
        break

    answer, retrieved_chunks, scores = ask_rag(
        question=question,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        model=model
    )

    print("\nBot:", answer)
    print("\nSources:")
    for chunk, score in zip(retrieved_chunks, scores):
        print(f"- {chunk['source']} / {chunk.get('title', '')} | score={score:.3f}")
    print()