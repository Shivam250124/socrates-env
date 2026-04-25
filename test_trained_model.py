#!/usr/bin/env python3
"""
Quick test script to demonstrate the trained model.
Run this to generate output for a screenshot.
"""

print("=" * 70)
print("SOCRATES Trained Model Demo")
print("=" * 70)
print()

# Test prompts showing different misconceptions
test_cases = [
    {
        "concept": "Floating Point Precision",
        "student": "I think 0.1 + 0.2 should equal 0.3 exactly in Python.",
    },
    {
        "concept": "Mutable Default Arguments",
        "student": "I think default arguments are created fresh each time a function is called.",
    },
    {
        "concept": "Integer Division",
        "student": "I think 5/2 in Python should always give 2.5 as the result.",
    },
]

print("Testing trained model: shivam250124/socrates-tutor-qwen-1.5b")
print()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "shivam250124/socrates-tutor-qwen-1.5b",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("shivam250124/socrates-tutor-qwen-1.5b")
    print("✓ Model loaded\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"{'─' * 70}")
        print(f"Example {i}: {test['concept']}")
        print(f"{'─' * 70}")
        print()
        
        prompt = f"""System: You are a Socratic tutor. Ask questions, never give answers.

User: {test['student']}

Tutor:"""
        
        print(f"Student: \"{test['student']}\"")
        print()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        print(f"Trained Model: \"{response}\"")
        print()
        
        # Check if it's asking a question
        if "?" in response:
            print("✅ Model asked a question (Socratic method)")
        else:
            print("⚠️  Model didn't ask a question")
        
        print()
    
    print("=" * 70)
    print("Demo complete! Take a screenshot of this output.")
    print("=" * 70)
    
except ImportError:
    print("❌ Error: transformers library not installed")
    print("Install with: pip install transformers torch")
    print()
    print("Or test on HuggingFace directly:")
    print("https://huggingface.co/shivam250124/socrates-tutor-qwen-1.5b")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Alternative: Test on HuggingFace directly:")
    print("https://huggingface.co/shivam250124/socrates-tutor-qwen-1.5b")
