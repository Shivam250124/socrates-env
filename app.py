"""
Simple Gradio app for HuggingFace Space
Demonstrates the SOCRATES environment concept with example responses
"""

import gradio as gr

# Example Socratic responses for different misconceptions
SOCRATIC_RESPONSES = {
    "floating_point": [
        "What do you think happens when you try that in Python?",
        "Can you run that code and tell me what result you get?",
        "Why might computers represent decimal numbers differently than we do?",
    ],
    "mutable_defaults": [
        "What would happen if you called the same function multiple times?",
        "Can you think of an example where modifying that default argument might cause issues?",
        "How does Python handle default arguments when a function is defined?",
    ],
    "integer_division": [
        "What version of Python are you using?",
        "How might the behavior differ between Python 2 and Python 3?",
        "Can you try using the // operator instead?",
    ],
    "recursion": [
        "What needs to happen for a recursive function to stop calling itself?",
        "Can you think of a case where recursion might never terminate?",
        "What's the base case in your recursive function?",
    ],
    "default": [
        "What do you think might happen if you test that assumption?",
        "Can you think of an example where that wouldn't work?",
        "What evidence do you have for that belief?",
    ]
}

def generate_socratic_question(student_misconception):
    """Generate a Socratic question for the student's misconception."""
    
    # Simple keyword matching to select appropriate response
    text_lower = student_misconception.lower()
    
    if "0.1" in text_lower or "0.2" in text_lower or "0.3" in text_lower or "floating" in text_lower:
        responses = SOCRATIC_RESPONSES["floating_point"]
    elif "default" in text_lower and "argument" in text_lower:
        responses = SOCRATIC_RESPONSES["mutable_defaults"]
    elif "division" in text_lower or "5/2" in text_lower:
        responses = SOCRATIC_RESPONSES["integer_division"]
    elif "recursion" in text_lower or "recursive" in text_lower:
        responses = SOCRATIC_RESPONSES["recursion"]
    else:
        responses = SOCRATIC_RESPONSES["default"]
    
    # Return first response (in real model, this would be generated)
    return responses[0]

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
    This demo shows the **SOCRATES environment concept** - training LLMs to teach using the Socratic method.
    
    **How it works**: Instead of explaining answers, the system asks questions to guide student discovery.
    
    **Note**: This demo uses rule-based responses. The full trained model (Qwen2.5-1.5B + LoRA) is available at the links below.
    
    **Links**:
    - [Trained Model on HuggingFace](https://huggingface.co/shivam250124/socrates-tutor-qwen-1.5b)
    - [GitHub Repository](https://github.com/Shivam250124/socrates-env)
    - [Blog Post](https://github.com/Shivam250124/socrates-env/blob/main/BLOG_POST.md)
    - [Colab Training Notebook](https://github.com/Shivam250124/socrates-env/blob/main/socrates_env/notebooks/SocratesTraining.ipynb)
    
    **Built for**: OpenEnv Hackathon 2026
    """,
    examples=examples,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
