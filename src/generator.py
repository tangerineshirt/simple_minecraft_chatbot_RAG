import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_llm():
    """
    Load the tokenizer and language model.

    This function is separated so app.py does not need to know
    the details of how the model is loaded.
    """

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    return tokenizer, model


def generate_answer(question, context, tokenizer, model):
    """
    Generate an answer using the retrieved RAG context.

    Parameters:
    - question: the user's question
    - context: retrieved chunks from retriever.py
    - tokenizer: loaded tokenizer
    - model: loaded language model

    Returns:
    - answer: generated text answer
    """

    system_prompt = """
You are a Minecraft assistant.

Answer the user's question using only the provided context.
Do not use outside knowledge.
Do not guess missing information.
Do not combine unrelated chunks.

If the answer is not found in the context, say:
"I do not know based on the provided context."

For crafting questions, only mention the item name, materials, and placement instructions that are written in the context.
"""

    user_prompt = f"""
Context:
{context}

Question:
{question}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=200,
        do_sample=False
    )

    input_length = model_inputs["input_ids"].shape[1]

    new_output_ids = output_ids[:, input_length:]

    answer = tokenizer.decode(
        new_output_ids[0],
        skip_special_tokens=True
    )

    return answer.strip()