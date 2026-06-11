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
├── app.py
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
```

---

## How It Works

The system follows a basic RAG pipeline:

```text
Local Minecraft text files
↓
Load documents
↓
Split documents into chunks
↓
Create embeddings
↓
Save chunks and embeddings
↓
Retrieve relevant chunks based on user question
↓
Send retrieved context to the language model
↓
Generate answer
```

The model does not search the internet.

It answers based on the local files inside the `mc_knowledge/` folder.

---

## Knowledge File Format

For best results, knowledge files should use `##` headings.

Example:

```text
# Minecraft Crafting Recipes Knowledge Base

## Crafting Table
A crafting table is crafted using 4 wooden planks.
Place 4 wooden planks in a 2x2 crafting grid.

## Furnace
A furnace is crafted using 8 cobblestone.
Place cobblestone around the crafting grid and leave the center empty.
```

Each `##` heading becomes one retrievable knowledge chunk.

This format is useful for structured Minecraft data such as:

```text
crafting recipes
mobs
biomes
commands
items
enchantments
redstone components
```

Files without `##` headings can still be processed using fallback size-based chunking, but heading-based files usually give better retrieval results.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/minecraft-rag-chatbot.git
cd minecraft-rag-chatbot
```

Replace `YOUR_USERNAME` with your actual GitHub username.

---

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
streamlit
sentence-transformers
transformers
torch
scikit-learn
numpy
```

---

## Build the Vector Index

Before running the chatbot, build the index first:

```bash
python build_index.py
```

This will:

```text
load files from mc_knowledge/
split them into chunks
create embeddings
save chunks.pkl and embeddings.npy into vector_store/
```

After running this command, the project should contain:

```text
vector_store/
├── chunks.pkl
└── embeddings.npy
```

You only need to run `build_index.py` again when you:

```text
add new knowledge files
edit existing knowledge files
change the chunking logic
change the embedding model
```

---

## Run the Terminal Chatbot

After building the index, run:

```bash
python app.py
```

Example usage:

```text
User: how do I craft a shield?
Bot: A shield is crafted using 6 wooden planks and 1 iron ingot...
```

To exit the chatbot, type:

```text
exit
```

or:

```text
quit
```

---

## Run the Streamlit Web App

To run the web interface:

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal.

Usually it will be:

```text
http://localhost:8501
```

The Streamlit app provides:

```text
chat interface
answer display
retrieved source display
similarity score display
```

---

## Normal Usage Flow

### First Time Setup

```bash
pip install -r requirements.txt
python build_index.py
streamlit run streamlit_app.py
```

### After Editing Knowledge Files

```bash
python build_index.py
streamlit run streamlit_app.py
```

### If the Index Is Already Built

```bash
streamlit run streamlit_app.py
```

or for terminal mode:

```bash
python app.py
```

---

## Main Files Explanation

### `build_index.py`

Builds the RAG index.

It loads text files, chunks them, creates embeddings, and saves the result into the `vector_store/` folder.

---

### `app.py`

Runs the terminal chatbot.

Use this for simple testing and debugging.

---

### `streamlit_app.py`

Runs the web-based chatbot interface using Streamlit.

Use this when you want a more user-friendly app.

---

### `src/loader.py`

Loads `.txt` files from the `mc_knowledge/` folder.

---

### `src/chunker.py`

Splits documents into chunks.

It supports:

```text
heading-based chunking using ##
fallback size-based chunking
```

---

### `src/storage.py`

Loads saved chunks and embeddings from the `vector_store/` folder.

---

### `src/retriever.py`

Retrieves the most relevant chunks based on the user's question using cosine similarity.

---

### `src/generator.py`

Loads the Hugging Face language model and generates answers using the retrieved context.

---

### `src/rag.py`

Connects the retriever and generator into one RAG pipeline.

---

## Model Used

This project uses:

```text
Embedding model:
sentence-transformers/all-MiniLM-L6-v2

Language model:
Qwen/Qwen2.5-0.5B-Instruct
```

The embedding model is used for semantic search.

The language model is used to generate the final answer.

---

## Notes About Accuracy

The chatbot only knows what is written in the local knowledge files.

If the answer is not inside the retrieved context, the chatbot should respond with:

```text
I do not know based on the provided context.
```

If the chatbot gives an incorrect answer, check the retrieved sources first.

Possible causes:

```text
wrong chunk retrieved
knowledge file is missing information
chunking format is unclear
top_k value is too high
model is mixing multiple chunks
prompt needs to be stricter
```

---

## Recommended Knowledge Base Format

Use separate files for different Minecraft topics:

```text
mc_knowledge/
├── crafting_recipes.txt
├── mobs.txt
├── biomes.txt
├── commands.txt
├── enchantments.txt
├── redstone.txt
└── dimensions.txt
```

Use `##` headings inside each file:

```text
## Creeper
Type: Hostile mob.
Behavior: Creepers approach players and explode.
Drops: Gunpowder.

## Zombie
Type: Hostile undead mob.
Behavior: Zombies attack players in melee range.
Drops: Rotten flesh.
```

---

## Limitations

- The chatbot does not browse the internet.
- The chatbot only answers from local `.txt` files.
- The answer quality depends heavily on the quality of the knowledge base.
- Small local models may still hallucinate if the prompt or retrieval is weak.
- Large knowledge bases may require a proper vector database such as FAISS or Chroma.

---

## Future Improvements

Possible future upgrades:

```text
add PDF support
add file upload in Streamlit
add Chroma or FAISS vector database
add LangChain integration
add chat memory
add source filtering by category
add better Minecraft knowledge files
add image input using a vision-language model
add deployment to Hugging Face Spaces or Streamlit Community Cloud
```

---

## Example Questions

```text
How do I craft a shield?
How do I craft a furnace?
What does a creeper drop?
Where do zombies spawn?
How do I make a bow?
What is a Nether portal?
```

---

## License

This project is for learning and educational purposes.
