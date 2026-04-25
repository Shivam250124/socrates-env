"""
Simple training script that actually works on T4 GPU.
Uses 8-bit quantization with LoRA for memory efficiency.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def train():
    """Simple training that works."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    from training.config import CONFIG
    from client import SocratesEnv
    from models import SocratesAction
    import torch

    print("=" * 60)
    print("SIMPLE TRAINING (Works on T4 GPU)")
    print("=" * 60)
    
    # Load model with 8-bit quantization
    print("\n1. Loading model with 8-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("✓ Model loaded with LoRA (only training 1-2% of parameters)")
    
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
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        # Add labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    print("\n4. Configuring training...")
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Can use larger batch with 8-bit
        gradient_accumulation_steps=1,
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,  # Higher LR for LoRA
        warmup_steps=10,
        optim="adamw_8bit",  # 8-bit optimizer
        report_to="none",
        max_grad_norm=1.0,
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
    
    # Save LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Also merge and save full model
    print("\n7. Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_dir = Path(CONFIG["output_dir"]) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    
    print(f"✓ LoRA adapters saved to {output_dir}")
    print(f"✓ Merged model saved to {merged_dir}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
