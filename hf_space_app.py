"""
Simple Gradio app for HuggingFace Space
Demonstrates the SOCRATES environment with the trained model
"""

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the trained model
MODEL_NAME = "shivam250124/socrates-tutor-qwen-1.5b"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Model loaded!")

def generate_socratic_question(student_misconception):
    """Generate a Socratic question for the student's misconception."""
    
    prompt = f"""System: You are a Socratic tutor. Ask questions, never give answers.

User: {student_misconception}

Tutor:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    return response

# Example misconceptions
examples = [
    ["I think 0.1 + 0.2 should equal 0.3 exactly in Python."],
    ["I think default arguments are created fresh each time a function is called."],
    ["I think 5/2 in Python should always give 2.5 as the result."],
    ["I think recursion always terminates eventually."],
    ["I think Python is pass-by-value for everything."],
]

# Create Gradio interface
demo = gr.Interface(
    fn=generate_socratic_question,
    inputs=gr.Textbox(
        label="Student's Misconception",
        placeholder="Enter a programming misconception...",
        lines=3
    ),
    outputs=gr.Textbox(
        label="Socratic Question from Trained Model",
        lines=5
    ),
    title="🎓 SOCRATES: Socratic Teaching Agent",
    description="""
    This model was trained to teach using the **Socratic method** - guiding students through questions instead of giving answers.
    
    **Try it**: Enter a programming misconception and see how the model responds with a question (not an explanation).
    
    **Links**:
    - [GitHub Repository](https://github.com/Shivam250124/socrates-env)
    - [Trained Model](https://huggingface.co/shivam250124/socrates-tutor-qwen-1.5b)
    - [Blog Post](https://github.com/Shivam250124/socrates-env/blob/main/BLOG_POST.md)
    """,
    examples=examples,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
