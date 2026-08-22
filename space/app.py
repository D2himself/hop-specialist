"""Hop-1 decomposition demo — one question in, one Hop-1 YAML decomposition out.

Loads Omokemi/hop-1-gemma-270m from the Hub (not a local path) and reuses the
exact prompt format documented in model_card.md: a fixed system turn plus the
question as the user turn.
"""

import gradio as gr
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Omokemi/hop-1-gemma-270m"

SYSTEM_PROMPT = (
    "You are a reasoning agent. Given a multi-hop question, output only the "
    "first reasoning step as YAML with three fields: thought, action, "
    "target_entity."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()  # skip this and dropout stays on, which collapses generation on a model this small

@spaces.GPU(duration=30)
def decompose(question: str) -> str:
    model.to("cuda")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output[0][inputs['input_ids'].shape[1] :], skip_special_tokens=True)


demo = gr.Interface(
    fn=decompose,
    title="Hop 1 Specialist",
    description=(
        "This is a fine-tuned small language model trained to do a narrow task. "
        "It takes in a multi-hop question and outputs a structured YAML "
        "decomposition (thought, action, target_entity), the first step toward "
        "answering the question."
    ),
    inputs=gr.Textbox(label="Multi-hop question"),
    outputs=gr.Textbox(label="Hop-1 decomposition"),
    examples=[
        ["The author of The Hobbit was born in which country?"],
        ["Are the Eiffel Tower and the Statue of Liberty located in the same country?"],
    ]
)

if __name__ == "__main__":
    demo.launch()
