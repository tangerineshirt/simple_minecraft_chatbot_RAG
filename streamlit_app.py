import streamlit as st
from sentence_transformers import SentenceTransformer

from src.storage import load_index
from src.generator import load_llm
from src.rag import ask_rag


st.set_page_config(
    page_title="Minecraft RAG Chatbot",
    page_icon="⛏️",
    layout="wide"
)


st.title("⛏️ Minecraft RAG Chatbot")
st.write("Ask Minecraft questions using your local knowledge base.")


@st.cache_resource
def load_resources():
    """
    Load heavy resources only once.

    Without this, Streamlit may reload the embedding model,
    chunks, embeddings, tokenizer, and LLM repeatedly.
    """

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    chunks, chunk_embeddings = load_index()

    tokenizer, model = load_llm()

    return embedding_model, chunks, chunk_embeddings, tokenizer, model


with st.spinner("Loading Minecraft RAG system..."):
    embedding_model, chunks, chunk_embeddings, tokenizer, model = load_resources()


st.success(f"Knowledge base loaded: {len(chunks)} chunks")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


question = st.chat_input("Ask about Minecraft...")


if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, retrieved_chunks, scores, retrieval_method = ask_rag(
                question=question,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                embedding_model=embedding_model,
                tokenizer=tokenizer,
                model=model
            )

        st.write(answer)

        st.markdown(f"**Retrieval method:** `{retrieval_method}`")

        with st.expander("Retrieved sources"):
            for chunk, score in zip(retrieved_chunks, scores):
                st.markdown(
                    f"**Source:** `{chunk['source']}`  \n"
                    f"**Title:** `{chunk.get('title', 'Unknown')}`  \n"
                    f"**Score:** `{score:.3f}`"
                )

                st.text(chunk["text"])

                st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })