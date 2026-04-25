"""
GRPO Training Loop — TRL + Unsloth + SOCRATES environment.

This script connects the SOCRATES environment to TRL's GRPO trainer
for reinforcement learning with verifiable rewards.

Requirements (GPU needed):
    pip install trl unsloth torch transformers wandb
"""

import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def train():
    """Main training entry point."""
    import os
    import torch
    
    # Force fp16, disable bf16 for T4 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cuda.matmul.allow_tf32 = True
    
    from trl import GRPOTrainer, GRPOConfig
    from unsloth import FastLanguageModel
    from training.config import CONFIG
    from training.rollout import build_tutor_prompt
    from client import SocratesEnv
    from models import SocratesAction

    logger.info("Loading model with Unsloth optimization...")
    
    # Force fp16 for T4 GPU
    import torch
    torch_dtype = torch.float16  # Explicitly use fp16, not bf16
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=2048,
        dtype=torch_dtype,  # Explicitly set to fp16
        load_in_4bit=CONFIG["load_in_4bit"],
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Use unsloth's optimized checkpointing
        use_rslora=False,
        use_loftq=False,
    )

    logger.info("Connecting to SOCRATES environment...")
    env = SocratesEnv(base_url=CONFIG["env_url"])

    def socrates_reward_fn(completions, prompts, **kwargs) -> list[float]:
        """
        Reward function that connects TRL to the SOCRATES environment.
        
        For each completion (a Socratic question), sends it to the environment
        and returns the reward.
        """
        rewards = []
        for completion in completions:
            question = completion.strip()
            try:
                action = SocratesAction(question=question)
                obs, reward, done, info = env.step(action)
                rewards.append(reward)
            except Exception as e:
                logger.error(f"Environment error: {e}")
                rewards.append(-1.0)
        return rewards

    def build_prompt_dataset(env_client: SocratesEnv, num_prompts: int = 200):
        """Build a dataset of initial prompts from the environment."""
        from datasets import Dataset

        prompts = []
        for i in range(num_prompts):
            obs = env_client.reset()
            obs_dict = obs.model_dump()
            prompt = build_tutor_prompt(obs_dict)
            prompts.append({"prompt": prompt})

        return Dataset.from_list(prompts)

    logger.info("Building prompt dataset...")
    dataset = build_prompt_dataset(env, num_prompts=200)

    logger.info("Configuring GRPO trainer...")
    grpo_config = GRPOConfig(
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["batch_size"],
        num_generations=CONFIG["num_rollouts_per_prompt"],
        max_prompt_length=1024,
        max_completion_length=256,
        output_dir=CONFIG["output_dir"],
        logging_steps=CONFIG["log_every_n_steps"],
        save_steps=CONFIG["save_every_n_episodes"],
        report_to="wandb" if CONFIG["use_wandb"] else "none",
        bf16=False,  # T4 GPU doesn't support bf16
        fp16=True,   # Use fp16 instead
    )

    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        reward_funcs=[socrates_reward_fn],
        train_dataset=dataset,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    # Save model correctly (Unsloth merged save)
    logger.info("Saving trained model...")
    output_dir = Path(CONFIG["output_dir"]) / "final"
    model.save_pretrained_merged(
        str(output_dir),
        tokenizer=tokenizer,
        save_method="merged_16bit",
    )
    logger.info(f"Model saved to {output_dir}")

    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
