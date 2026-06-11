from src.storage import load_index
from src.retriever import fuzzy_title_match

chunks, chunk_embeddings = load_index()

test_questions = [
    "how to build nether portal",
    "how to buid nether protal",
    "how to craft furnce",
    "what does creper drop",
    "how to make strenght potion",
    "what does sharpnes do",
    "where can i find diamon ore",
]

for question in test_questions:
    chunk, score = fuzzy_title_match(question, chunks, threshold=80)

    print("\nQuestion:", question)

    if chunk is None:
        print("No fuzzy title match")
    else:
        print("Matched title:", chunk.get("title"))
        print("Source:", chunk.get("source"))
        print("Score:", score)