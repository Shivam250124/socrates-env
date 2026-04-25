"""
Simple training script that actually works on T4 GPU.
Uses basic supervised fine-tuning instead of GRPO.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def train():
    """Simple training that works."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from datasets import Dataset
    from training.config import CONFIG
    from client import SocratesEnv
    from models import SocratesAction
    import torch

    print("=" * 60)
    print("SIMPLE TRAINING (Works on T4 GPU)")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Model loaded")
    
    # Connect to environment
    print("\n2. Connecting to environment...")
    env = SocratesEnv(base_url=CONFIG["env_url"])
    print("✓ Connected")
    
    # Collect training data
    print("\n3. Collecting training examples...")
    examples = []
    
    for i in range(50):  # Collect 50 examples
        obs = env.reset()
        
        # Create training example
        prompt = f"System: You are a Socratic tutor. Ask questions, never give answers.\n\nUser: {obs.student_response}\n\nTutor:"
        
        # Good Socratic question examples
        good_questions = [
            "What do you think might cause that?",
            "Can you think of an example where that wouldn't work?",
            "What assumptions are you making?",
            "How would you test that idea?",
            "What evidence supports that view?",
        ]
        
        response = good_questions[i % len(good_questions)]
        
        examples.append({
            "text": prompt + " " + response
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Collected {i + 1}/50 examples...")
    
    env.close()
    print("✓ Training data ready")
    
    # Create dataset
    dataset = Dataset.from_list(examples)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    print("\n4. Configuring training...")
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=100,
        logging_steps=10,
        fp16=True,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train
    print("\n5. Training...")
    print("=" * 60)
    trainer.train()
    print("=" * 60)
    
    # Save
    print("\n6. Saving model...")
    output_dir = Path(CONFIG["output_dir"]) / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Model saved to {output_dir}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
