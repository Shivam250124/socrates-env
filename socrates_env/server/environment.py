"""
SOCRATES Environment — main environment class.

Implements the OpenEnv Environment interface: reset(), step(), state().
Orchestrates student simulator, reward calculator, concept bank, and curriculum.
"""

import os
import logging
from pathlib import Path
from typing import Optional

try:
    from openenv.core.env_server import Environment
except ImportError:
    # Graceful fallback when openenv-core is not installed (local dev)
    class Environment:
        """Stub base class for local development without openenv-core."""
        def __init__(self):
            pass

from models import SocratesAction, SocratesObservation, SocratesState, ConversationTurn
from server.student import StudentSimulator
from server.concepts import ConceptBank
from server.rewards import SocratesRewardCalculator
from server.curriculum import SocratesCurriculum

logger = logging.getLogger(__name__)


class SocratesEnvironment(Environment):
    """
    RL environment for Socratic teaching.
    Trains LLMs to guide students to understanding using questions alone.
    """

    def __init__(self, concepts_dir: str = None):
        super().__init__()

        # Resolve concepts directory relative to this file
        if concepts_dir is None:
            base = Path(__file__).resolve().parent.parent
            concepts_dir = str(base / "concepts")

        self.concept_bank = ConceptBank.load(concepts_dir)
        self.reward_calculator = SocratesRewardCalculator()
        self.curriculum = SocratesCurriculum()

        self.student: Optional[StudentSimulator] = None
        self.current_concept = None
        self.episode_num = 0
        self.episode_history: list[dict] = []
        self.cumulative_compliance: float = 0.0

        logger.info(f"SocratesEnvironment initialized with {len(self.concept_bank.concepts)} concepts")

    def reset(self, task: str = "foundation") -> SocratesObservation:
        """
        Start a new tutoring episode.

        Args:
            task: Curriculum phase / difficulty. One of "foundation", "intermediate", "advanced".
                  Also accepts "easy", "medium", "hard" as aliases.

        Returns:
            Initial SocratesObservation.
        """
        # Map task aliases
        task_map = {"easy": "foundation", "medium": "intermediate", "hard": "advanced"}
        phase_name = task_map.get(task, task)

        # Select concept via curriculum
        self.episode_num += 1
        concept_id = self.curriculum.get_concept_for_episode(self.episode_num)
        self.current_concept = self.concept_bank.get(concept_id)
        max_steps = self.curriculum.max_steps_for(self.episode_num)
        self.student = StudentSimulator(self.current_concept, max_steps=max_steps)
        self.episode_history = []
        self.cumulative_compliance = 0.0

        logger.info(
            f"Episode {self.episode_num}: concept={concept_id}, "
            f"phase={self.curriculum.get_phase_name(self.episode_num)}, "
            f"max_steps={max_steps}"
        )

        return SocratesObservation(
            concept_description=self.current_concept.description,
            student_current_belief=self.current_concept.initial_misconception,
            student_response=self.current_concept.initial_student_statement,
            student_confidence="confused",
            steps_remaining=max_steps,
            history=[],
            done=False,
            success=False,
        )

    def step(self, action: SocratesAction) -> SocratesObservation:
        """
        Process one agent question and advance the episode.

        Args:
            action: SocratesAction containing the question.

        Returns:
            SocratesObservation with updated state.

        Raises:
            RuntimeError: If called before reset().
        """
        if self.student is None:
            raise RuntimeError("Call reset() before step()")

        prev_state = self.student.get_state()

        # Compute template similarity using embeddings (Fix 2)
        template_sim = self.concept_bank.template_similarity(
            action.question, self.current_concept.concept_id
        )

        # Student responds to question
        student_response, understanding_delta = self.student.respond_to_question(
            action.question, template_similarity=template_sim
        )
        new_state = self.student.get_state()

        # Check for repeated questions (Fix 7)
        repeat_penalty = 0.0
        if len(self.episode_history) > 0:
            repeat_penalty = self._check_repeat(action.question)

        # Compute reward
        episode_done = self.student.is_done
        reward, reward_breakdown = self.reward_calculator.compute_reward(
            action=action,
            prev_state=prev_state,
            new_state=new_state,
            concept=self.current_concept,
            episode_done=episode_done,
            template_similarity=template_sim,
            cumulative_compliance=self.cumulative_compliance,
        )
        reward += repeat_penalty

        # Track cumulative compliance for efficiency gating (Fix 5)
        self.cumulative_compliance += reward_breakdown.get("socratic_compliance", 0.0)

        # Log history
        turn = {
            "step": new_state.step_count,
            "agent_question": action.question,
            "student_response": student_response,
            "understanding_before": prev_state.understanding_level,
            "understanding_after": new_state.understanding_level,
            "reward": reward,
            "reward_breakdown": reward_breakdown,
            "template_similarity": template_sim,
        }
        self.episode_history.append(turn)

        # Record for adaptive curriculum
        if episode_done:
            self.curriculum.record_episode_result(
                self.current_concept.concept_id, new_state.success
            )

        max_steps = self.curriculum.max_steps_for(self.episode_num)

        obs = SocratesObservation(
            concept_description=self.current_concept.description,
            student_current_belief=self.current_concept.initial_misconception,
            student_response=student_response,
            student_confidence=new_state.confidence_label,
            steps_remaining=max_steps - new_state.step_count,
            history=[
                ConversationTurn(
                    step=h["step"],
                    agent_question=h["agent_question"],
                    student_response=h["student_response"],
                    student_confidence=self._confidence_for_level(h["understanding_after"]),
                )
                for h in self.episode_history
            ],
            done=episode_done,
            success=new_state.success,
        )

        # Attach reward and info to observation metadata for OpenEnv protocol
        obs._reward = reward
        obs._done = episode_done
        obs._info = {
            "reward_breakdown": reward_breakdown,
            "understanding_level": new_state.understanding_level,
            "repeat_penalty": repeat_penalty,
            "concept_id": self.current_concept.concept_id,
            "template_similarity": template_sim,
            "episode_num": self.episode_num,
        }

        return obs

    def state(self) -> SocratesState:
        """Return full environment state for debugging."""
        if self.student is None:
            return SocratesState(empty=True)

        student_state = self.student.get_state()
        return SocratesState(
            concept_id=self.current_concept.concept_id,
            understanding_level=student_state.understanding_level,
            active_misconceptions=student_state.active_misconceptions,
            episode_history=self.episode_history,
            step_count=student_state.step_count,
            done=self.student.is_done,
            success=self.student.success,
            episode_num=self.episode_num,
            cumulative_compliance=self.cumulative_compliance,
        )

    def _check_repeat(self, question: str) -> float:
        """
        Check for repeated/rephrased questions (Fix 7).
        
        Uses embedding cosine similarity when sentence-transformers is available,
        falls back to word-overlap when it's not.
        """
        embeddings = self.concept_bank.embeddings

        for entry in self.episode_history:
            prev_q = entry.get("agent_question", "")
            if not prev_q:
                continue

            # Tier 1: Embedding-based similarity (sentence-transformers)
            if embeddings._use_st and embeddings._model is not None:
                try:
                    import numpy as np
                    q_emb = embeddings._model.encode([question], normalize_embeddings=True)
                    p_emb = embeddings._model.encode([prev_q], normalize_embeddings=True)
                    sim = float(np.dot(q_emb[0], p_emb[0]))
                    if sim > 0.85:
                        return -0.3
                    continue  # Checked via embeddings, skip word-overlap
                except Exception:
                    pass  # Fall through to word-overlap

            # Tier 2: Word-overlap fallback
            from server.concepts import _word_overlap_similarity
            overlap = _word_overlap_similarity(question, prev_q)
            if overlap > 0.75:
                return -0.3

        return 0.0

    @staticmethod
    def _confidence_for_level(level: float) -> str:
        """Map understanding level to confidence label.
        
        Must match concept JSON response keys:
        confused, uncertain, starting_to_see, almost_there, understood.
        """
        if level < 0.15:
            return "confused"
        elif level < 0.35:
            return "uncertain"
        elif level < 0.60:
            return "starting_to_see"
        elif level < 0.80:
            return "almost_there"
        else:
            return "understood"
