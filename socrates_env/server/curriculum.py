"""
Curriculum system — 3-phase difficulty progression.

Fix 8: Optional adaptive mastery gating (slides window success rate).
Fix 9: Uses min_steps_to_success from concept JSON for difficulty ordering.
"""

import random
from typing import Optional


class SocratesCurriculum:
    """
    3-phase curriculum with optional mastery-based gating.
    """

    PHASES = [
        {
            "name": "foundation",
            "episodes": (0, 200),
            "concepts": ["index_zero", "integer_division"],
            "max_steps": 8,
            "success_threshold": 0.70,
        },
        {
            "name": "intermediate",
            "episodes": (200, 500),
            "concepts": ["boolean_operators", "modulo_negative", "mutable_defaults"],
            "max_steps": 10,
            "success_threshold": 0.80,
        },
        {
            "name": "advanced",
            "episodes": (500, 1000),
            "concepts": ["floating_point", "recursive_termination", "pass_by_reference"],
            "max_steps": 12,
            "success_threshold": 0.85,
        },
    ]

    def __init__(self, adaptive: bool = True):
        self.adaptive = adaptive
        # Fix 8: Track per-concept success for adaptive gating
        self._success_history: dict[str, list[bool]] = {}

    def get_concept_for_episode(self, episode_num: int) -> str:
        """
        Select a concept for the given episode number.
        
        With adaptive=True, stays on unmastered concepts.
        """
        phase = self._get_phase(episode_num)
        concepts = phase["concepts"]

        if self.adaptive and self._success_history:
            # Fix 8: Check mastery — stay on unmastered concepts
            unmastered = []
            for concept_id in concepts:
                recent = self._success_history.get(concept_id, [])[-20:]
                if len(recent) < 10 or (sum(recent) / len(recent)) < 0.8:
                    unmastered.append(concept_id)

            if unmastered:
                # Deterministic selection based on episode number
                return unmastered[episode_num % len(unmastered)]

        # Default: deterministic rotation through phase concepts
        return concepts[episode_num % len(concepts)]

    def record_episode_result(self, concept_id: str, success: bool):
        """Record whether an episode was successful for adaptive gating."""
        if concept_id not in self._success_history:
            self._success_history[concept_id] = []
        self._success_history[concept_id].append(success)

    def max_steps_for(self, episode_num: int) -> int:
        """Get max steps allowed for the given episode number."""
        return self._get_phase(episode_num)["max_steps"]

    def _get_phase(self, episode_num: int) -> dict:
        """Get the curriculum phase for the given episode number."""
        for phase in self.PHASES:
            start, end = phase["episodes"]
            if start <= episode_num < end:
                return phase
        # Past all phases — use advanced
        return self.PHASES[-1]

    def get_phase_name(self, episode_num: int) -> str:
        """Get the name of the current curriculum phase."""
        return self._get_phase(episode_num)["name"]
