"""
Reward calculator — 5 independent signals + anti-hacking measures.

Implements all fixes:
- Fix 1: Classifier ordering bug (handled in student.py, rewards uses its output)
- Fix 4: R3/R4 are decoupled from student classification (computed purely from text)
- Fix 5: Efficiency × compliance coupling
"""

import re
from typing import Optional

from models import Concept, SocratesAction
from server.student import StudentState


class SocratesRewardCalculator:
    """
    5 independent reward components. All programmatic. No LLM-as-judge.
    """

    WEIGHTS = {
        "teaching_progress":      0.40,
        "socratic_compliance":    0.25,
        "question_quality":       0.15,
        "efficiency":             0.10,
        "misconception_targeting": 0.10,
    }

    def compute_reward(
        self,
        action: SocratesAction,
        prev_state: StudentState,
        new_state: StudentState,
        concept: Concept,
        episode_done: bool,
        template_similarity: float = 0.0,
        cumulative_compliance: float = 0.0,
    ) -> tuple[float, dict]:
        """
        Compute the total reward and per-signal breakdown.

        Returns:
            (total_reward, reward_breakdown_dict)
        """
        rewards = {}

        # ─── R1: Teaching Progress ────────────────────────────────────────
        delta = new_state.understanding_level - prev_state.understanding_level
        rewards["teaching_progress"] = delta * 2.0

        # ─── R2: Socratic Compliance ──────────────────────────────────────
        rewards["socratic_compliance"] = self._check_socratic_compliance(
            action.question, concept.answer_keywords
        )

        # ─── R3: Question Quality (Fix 4: purely from text, not classification) ──
        rewards["question_quality"] = self._score_question_quality(action.question)

        # ─── R4: Misconception Targeting (Fix 4: uses embeddings, not classification) ──
        rewards["misconception_targeting"] = self._check_targeting(
            action.question, prev_state.active_misconceptions, concept, template_similarity
        )

        # ─── R5: Efficiency (Fix 5: gated by cumulative compliance) ───────
        if episode_done and new_state.success:
            steps_used = new_state.step_count
            max_steps = 12
            raw_efficiency = (max_steps - steps_used) / max_steps * 0.5

            # Gate by average compliance — cheaters get no efficiency bonus
            avg_steps = max(1, new_state.step_count)
            avg_compliance = cumulative_compliance / avg_steps
            # compliance is typically in [-1.5, 0.0]; normalize to [0, 1]
            compliance_gate = max(0.0, (avg_compliance + 1.5) / 1.5)
            rewards["efficiency"] = raw_efficiency * compliance_gate
        elif episode_done and not new_state.success:
            rewards["efficiency"] = new_state.understanding_level * 0.2 - 0.3
        else:
            rewards["efficiency"] = 0.0

        # Hard rule penalties
        hard_penalty = self._apply_hard_rules(action, prev_state)
        rewards["hard_penalties"] = hard_penalty

        # Weighted total
        total = sum(
            self.WEIGHTS.get(k, 0.0) * v for k, v in rewards.items()
            if k != "hard_penalties"
        )
        total += hard_penalty  # Hard penalties are unweighted — they apply directly

        return total, rewards

    def _check_socratic_compliance(self, question: str, answer_keywords: list[str]) -> float:
        """
        Check if the question reveals the answer.

        -1.5 = directly stated answer (2+ keywords)
        -0.8 = partial reveal (1 keyword)
        -0.4 = leading/rhetorical pattern
        0.0  = clean question
        """
        q_lower = question.lower()

        # Strict: answer keywords directly stated
        direct_hits = sum(1 for kw in answer_keywords if kw.lower() in q_lower)
        if direct_hits >= 2:
            return -1.5
        if direct_hits == 1:
            return -0.8

        # Rhetorical confirms (Fix 1 — these pass keyword checks but are clearly leading)
        rhetorical_patterns = [
            r"wouldn't (it|that) be",
            r"so (it|that) must be",
            r"isn't that because",
            r"don't you think.*(because|that)",
            r"isn't it (true|obvious) that",
            r"so basically",
            r"so the (answer|reason) is",
        ]
        for pattern in rhetorical_patterns:
            if re.search(pattern, q_lower):
                return -0.4

        # Soft: leading question structure
        leading_patterns = [
            r"isn't it because",
            r"could it be that",
            r"is it possible that",
            r"would you say that",
            r"could it be related to",
        ]
        for pattern in leading_patterns:
            if re.search(pattern, q_lower):
                return -0.4

        return 0.0

    def _score_question_quality(self, question: str) -> float:
        """
        Score the intrinsic quality of the question structure.
        Fix 4: Computed purely from question text, independent of student state.

        +0.3 = open-ended
        0.0  = neutral
        -0.2 = yes/no
        -0.3 = multi-part
        -0.5 = not even a question
        """
        q_stripped = question.strip()

        # Must end with ?
        if not q_stripped.endswith("?"):
            return -0.5

        q_lower = q_stripped.lower()

        # Multiple questions → confusing
        if q_stripped.count("?") > 1:
            return -0.3

        # Yes/no question detector
        yes_no_starters = [
            "is ", "are ", "do ", "does ", "can ", "did ",
            "was ", "were ", "has ", "have ", "would ", "could ", "should ",
        ]
        if any(q_lower.startswith(s) for s in yes_no_starters):
            return -0.2

        # Good open question starters
        open_starters = [
            "what", "how", "why", "when", "where", "who",
            "which", "in what way", "to what extent",
        ]
        if any(q_lower.startswith(s) for s in open_starters):
            return 0.3

        return 0.0

    def _check_targeting(
        self,
        question: str,
        active_misconceptions: list[str],
        concept: Concept,
        template_similarity: float,
    ) -> float:
        """
        Does the question target an active misconception?
        Fix 4: Uses embedding similarity, not student classification.
        """
        if not active_misconceptions:
            return 0.1  # No misconceptions left

        q_lower = question.lower()

        # Check relevance to concept (must have at least one concept keyword)
        concept_kw_hits = sum(
            1 for kw in concept.concept_keywords if kw.lower() in q_lower
        )
        if concept_kw_hits == 0:
            return -0.1  # Off-topic

        # Use template similarity (from embeddings, not Jaccard)
        if template_similarity > 0.6:
            return 0.3
        elif template_similarity > 0.35:
            return 0.15
        else:
            return 0.0

    def _apply_hard_rules(self, action: SocratesAction, prev_state: StudentState) -> float:
        """
        Hard penalties for blatant violations.
        These apply directly (not weighted).
        """
        penalty = 0.0
        q = action.question.strip()

        # Rule 1: Minimum question length
        if len(q) < 10:
            penalty -= 0.5

        # Rule 2: Must contain at least one "?"
        if "?" not in q:
            penalty -= 0.4

        # Rule 3: Max length (no essay dumps)
        if len(q) > 200:
            penalty -= 0.2

        return penalty

    def check_repeated_question(
        self, question: str, history: list[dict],
        similarity_fn=None,
    ) -> float:
        """
        Check for repeated questions using embedding similarity (Fix 7).
        Returns penalty (0.0 or negative).
        """
        if not history or similarity_fn is None:
            return 0.0

        for entry in history:
            prev_q = entry.get("agent_question", "")
            if prev_q:
                sim = similarity_fn(question, prev_q)
                if sim > 0.85:
                    return -0.3
        return 0.0
