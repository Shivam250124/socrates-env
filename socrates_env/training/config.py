"""
Training configuration — hyperparameters for GRPO + Unsloth + TRL.
"""

CONFIG = {
    # Model: Start small for fast curves, scale for demo
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    # "model_name": "Qwen/Qwen2.5-7B-Instruct",  # Scale up after env is stable

    # Training
    "algorithm": "GRPO",
    "num_rollouts_per_prompt": 8,
    "learning_rate": 2e-5,
    "batch_size": 16,
    "gradient_accumulation": 4,
    "max_episodes": 500,

    # Unsloth settings
    "load_in_4bit": True,
    "use_lora": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,

    # Environment
    "env_url": "ws://localhost:7860/ws",
    "max_episode_steps": 12,
    "curriculum_enabled": True,

    # Logging
    "use_wandb": True,
    "log_every_n_steps": 10,
    "save_every_n_episodes": 100,
    "output_dir": "./checkpoints",
}
