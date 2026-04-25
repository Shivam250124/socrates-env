"""
Baseline evaluation — measure zero-shot performance before training.

Runs the environment with default/naive questions to establish reference scores.
Can also evaluate with a language model via API.
"""

import json
import logging
import sys
from pathlib import Path

from models import SocratesAction
from client import SocratesEnv
from training.rollout import run_episode

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Representative baseline questions that a naive model would ask
NAIVE_QUESTIONS = [
    "Can you tell me more about that?",
    "Why do you think that is?",
    "What happens when you try it?",
    "Have you considered other possibilities?",
    "What would you expect to happen?",
    "Can you think of a simpler case?",
    "What does the documentation say?",
    "Have you seen this behavior before?",
]

# Direct-answer questions (should be penalized)
CHEATING_QUESTIONS = [
    "Is it because of IEEE 754 binary fraction representation?",
    "Don't you think the base case is missing?",
    "Isn't it because mutable defaults are evaluated once at definition time?",
    "So the array index is really a memory offset, right?",
]


def run_baseline(
    env_url: str = "ws://localhost:7860/ws",
    tasks: list[str] = None,
    episodes_per_task: int = 3,
) -> dict:
    """
    Run baseline evaluation across all difficulty levels.

    Args:
        env_url: WebSocket URL of the environment server.
        tasks: List of task difficulties to evaluate.
        episodes_per_task: Number of episodes per task.

    Returns:
        Results dict with scores per task.
    """
    if tasks is None:
        tasks = ["foundation", "intermediate", "advanced"]

    results = {}

    with SocratesEnv(base_url=env_url) as client:
        for task in tasks:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running baseline: {task.upper()}")
            logger.info(f"{'='*60}")

            scores = []
            for ep in range(episodes_per_task):
                # Use naive questions in rotation
                question_idx = 0

                def get_question():
                    nonlocal question_idx
                    q = NAIVE_QUESTIONS[question_idx % len(NAIVE_QUESTIONS)]
                    question_idx += 1
                    return q

                trajectory = run_episode(
                    env_client=client,
                    task=task,
                    default_question=get_question(),
                )

                scores.append(trajectory["total_reward"])
                logger.info(
                    f"  Episode {ep + 1}: reward={trajectory['total_reward']:.3f}, "
                    f"success={trajectory['success']}, "
                    f"steps={len(trajectory['steps'])}, "
                    f"confidence={trajectory['final_confidence']}"
                )

            mean_score = sum(scores) / len(scores) if scores else 0.0
            results[task] = {
                "mean_reward": mean_score,
                "episodes": len(scores),
                "scores": scores,
            }
            logger.info(f"  Mean reward: {mean_score:.4f}")

    return results


def main():
    """Run baseline evaluation and print results."""
    results = run_baseline()

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))

    # Log expected vs actual
    expected = {"foundation": 0.50, "intermediate": 0.35, "advanced": 0.20}
    print("\nComparison with expected baseline:")
    for task, data in results.items():
        exp = expected.get(task, 0.0)
        diff = data["mean_reward"] - exp
        print(f"  {task}: {data['mean_reward']:.4f} (expected: {exp:.4f}, diff: {diff:+.4f})")


if __name__ == "__main__":
    main()
