"""
Rollout function — runs complete episodes for evaluation and dataset building.
"""

import torch
from typing import Optional

from models import SocratesAction
from client import SocratesEnv


def build_tutor_prompt(obs_dict: dict) -> str:
    """Build the system + user prompt for the Socratic tutor agent."""
    system = (
        "You are a Socratic tutor. Your ONLY role is to ask questions.\n"
        "You must NEVER state, explain, or hint at the answer directly.\n"
        "Guide your student to the answer through probing questions.\n"
        "Ask exactly ONE question per turn. Make it open-ended.\n"
    )

    # Format conversation history
    history_text = ""
    history = obs_dict.get("history", [])
    for turn in history:
        q = turn.get("agent_question", turn.get("question", ""))
        r = turn.get("student_response", turn.get("response", ""))
        history_text += f"Tutor: {q}\nStudent: {r}\n"

    user = (
        f"=== TUTORING SESSION ===\n"
        f"Concept your student is learning: {obs_dict.get('concept_description', '')}\n"
        f"Student's initial belief: {obs_dict.get('student_current_belief', '')}\n"
        f"Student confidence: {obs_dict.get('student_confidence', 'confused')}\n"
        f"Steps remaining: {obs_dict.get('steps_remaining', 0)}\n\n"
        f"=== CONVERSATION HISTORY ===\n"
        f"{history_text if history_text else '(No history yet — this is the first turn.)'}\n\n"
        f"=== STUDENT'S LATEST RESPONSE ===\n"
        f"{obs_dict.get('student_response', '')}\n\n"
        f"What is your next Socratic question? (One question only. Never state the answer.)"
    )

    return f"System: {system}\n\nUser: {user}"


def run_episode(
    env_client: SocratesEnv,
    model=None,
    tokenizer=None,
    task: str = "foundation",
    max_steps: int = 12,
    default_question: str = "What do you think about that?",
) -> dict:
    """
    Run one complete episode and return trajectory.

    If model/tokenizer are None, uses default_question for each step.
    This is useful for testing the environment without a model.

    Args:
        env_client: Connected SocratesEnv client.
        model: Optional HuggingFace model for generation.
        tokenizer: Optional tokenizer.
        task: Task difficulty level.
        max_steps: Maximum steps per episode.
        default_question: Fallback question if no model provided.

    Returns:
        Trajectory dict with steps, rewards, and outcomes.
    """
    obs = env_client.reset(task=task)
    obs_dict = obs.model_dump()

    trajectory = {
        "concept": obs_dict.get("concept_description", "")[:80],
        "steps": [],
        "total_reward": 0.0,
        "success": False,
        "final_confidence": "confused",
    }

    for step_num in range(max_steps):
        # Build prompt
        prompt = build_tutor_prompt(obs_dict)

        # Generate question
        if model is not None and tokenizer is not None:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            question = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
        else:
            question = default_question

        # Step environment
        action = SocratesAction(question=question)
        obs, reward, done, info = env_client.step(action)
        obs_dict = obs.model_dump()

        trajectory["steps"].append({
            "step": step_num + 1,
            "prompt": prompt[:200] + "...",
            "question": question,
            "student_response": obs_dict.get("student_response", ""),
            "reward": reward,
            "confidence": obs_dict.get("student_confidence", ""),
        })
        trajectory["total_reward"] += reward

        if done:
            trajectory["success"] = obs_dict.get("success", False)
            trajectory["final_confidence"] = obs_dict.get("student_confidence", "")
            break

    return trajectory
