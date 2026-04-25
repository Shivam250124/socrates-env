"""
Student Simulator — deterministic state machine.

The student responds to the agent's questions with rule-based transitions.
Understanding increases based on question quality, with non-linear
diminishing returns (Fix 3).
"""

import random
import re
from typing import Optional

from models import Concept


class StudentState:
    """Snapshot of the student's internal state."""

    def __init__(self, understanding_level: float, active_misconceptions: list[str],
                 step_count: int, last_response: str):
        self.understanding_level = understanding_level
        self.active_misconceptions = active_misconceptions.copy()
        self.step_count = step_count
        self.last_response = last_response

    @property
    def success(self) -> bool:
        return self.understanding_level >= 0.85

    @property
    def confidence_label(self) -> str:
        """Map understanding_level to a human-readable confidence label.
        
        IMPORTANT: These labels MUST match the keys in concept JSON 'responses' dicts.
        The 5 levels are: confused, uncertain, starting_to_see, almost_there, understood.
        """
        if self.understanding_level < 0.15:
            return "confused"
        elif self.understanding_level < 0.35:
            return "uncertain"
        elif self.understanding_level < 0.60:
            return "starting_to_see"
        elif self.understanding_level < 0.80:
            return "almost_there"
        else:
            return "understood"


class StudentSimulator:
    """
    Deterministic student state machine.
    
    - No randomness in responses (deterministic for RL training)
    - Non-linear understanding deltas (Fix 3: diminishing returns)
    - Contextual responses per confidence level
    - Question classification via rules (embedding similarity done externally)
    """

    def __init__(self, concept: Concept, max_steps: int = 12):
        self.concept = concept
        self.understanding_level: float = 0.0
        self.active_misconceptions: list[str] = list(concept.misconception_phrases)
        self.last_response: str = concept.initial_student_statement
        self.step_count: int = 0
        self.max_steps: int = max_steps

    def get_state(self) -> StudentState:
        """Return a snapshot of the current student state."""
        return StudentState(
            understanding_level=self.understanding_level,
            active_misconceptions=self.active_misconceptions,
            step_count=self.step_count,
            last_response=self.last_response,
        )

    def respond_to_question(
        self, question: str, template_similarity: float = 0.0
    ) -> tuple[str, float]:
        """
        Process a question and return (student_response, understanding_delta).

        Args:
            question: The agent's question text.
            template_similarity: Pre-computed embedding similarity to good templates.

        Returns:
            (response_text, understanding_delta)
        """
        self.step_count += 1
        q_type = self._classify_question(question, template_similarity)

        # Compute targeting score
        targeting_score = self._compute_targeting_score(question)

        # Compute non-linear delta (Fix 3)
        delta = self._compute_delta(q_type, self.understanding_level, targeting_score)

        # Apply delta
        prev_level = self.understanding_level
        self.understanding_level = min(1.0, self.understanding_level + delta)

        # Weaken misconceptions if good question
        if q_type in ("good_socratic", "counterexample") and delta > 0.05:
            self._weaken_misconceptions(question)

        # Generate response based on new understanding level
        response = self._generate_response()
        self.last_response = response

        return response, delta

    def _classify_question(self, question: str, template_similarity: float) -> str:
        """
        Classify the question type using rules + external embedding similarity.
        
        Fix 1: Compute ALL classifications, return the worst (most penalizing).
        Does NOT short-circuit.
        """
        q_lower = question.lower().strip()

        # --- Compute all signals ---
        # Signal 1: Direct answer (contains answer keywords)
        answer_hits = sum(1 for kw in self.concept.answer_keywords if kw.lower() in q_lower)
        is_direct_answer = answer_hits >= 1

        # Signal 2: Rhetorical confirm patterns (Fix 1 addition)
        rhetorical_patterns = [
            r"wouldn't (it|that) be",
            r"so (it|that) must be",
            r"isn't that because",
            r"don't you think.*(because|that)",
            r"isn't it (true|obvious) that",
            r"so basically",
            r"so the (answer|reason) is",
        ]
        is_rhetorical = any(re.search(p, q_lower) for p in rhetorical_patterns)

        # Signal 3: Leading question (hedged hint)
        leading_patterns = [
            r"isn't it because",
            r"could it be that",
            r"is it possible that",
            r"would you say that",
            r"could it be related to",
        ]
        is_leading = any(re.search(p, q_lower) for p in leading_patterns)

        # Signal 4: Yes/no structure
        yes_no_starters = [
            "is ", "are ", "do ", "does ", "can ", "did ",
            "was ", "were ", "has ", "have ", "should ", "would ", "could ",
        ]
        is_yes_no = any(q_lower.startswith(s) for s in yes_no_starters)

        # --- Return the worst classification (Fix 1: no short-circuit) ---
        # Direct answer with rhetorical = extremely bad
        if is_direct_answer and (is_rhetorical or answer_hits >= 2):
            return "direct_answer"

        # Direct answer alone
        if is_direct_answer:
            return "direct_answer"

        # Rhetorical confirm (even without keywords, it's leading)
        if is_rhetorical:
            return "leading"

        # Leading question
        if is_leading:
            return "leading"

        # Good Socratic (uses embedding similarity, not Jaccard — Fix 2)
        if template_similarity > 0.55:
            return "good_socratic"

        # Counterexample patterns
        counterexample_markers = [
            "what if", "imagine", "consider", "what about",
            "what happens when", "suppose", "let's say",
        ]
        if any(m in q_lower for m in counterexample_markers):
            return "counterexample"

        # Yes/no (only if nothing worse was found)
        if is_yes_no:
            return "yes_no"

        # Off-topic: no relevant concept keywords
        concept_kw_hits = sum(1 for kw in self.concept.concept_keywords if kw.lower() in q_lower)
        if concept_kw_hits == 0:
            return "off_topic"

        # Check if it's still a decent question via template similarity
        if template_similarity > 0.35:
            return "good_socratic"

        return "generic"

    def _compute_delta(self, q_type: str, understanding: float,
                       targeting_score: float) -> float:
        """
        Compute understanding delta with non-linear diminishing returns (Fix 3).
        
        Early targeted questions give big jumps.
        Late-stage questions require more precision for smaller gains.
        """
        base_deltas = {
            "good_socratic": 0.25,
            "counterexample": 0.18,
            "yes_no": 0.05,
            "generic": 0.02,
            "direct_answer": 0.0,  # Understanding via cheating — no Socratic learning
            "leading": 0.0,
            "off_topic": 0.0,
        }
        base = base_deltas.get(q_type, 0.0)

        if base <= 0:
            return 0.0

        # Diminishing returns: harder to advance at higher understanding
        difficulty_factor = 1.0 - (understanding ** 1.5)
        difficulty_factor = max(0.1, difficulty_factor)  # Always some small gain possible

        # Targeting multiplier: hitting misconceptions matters more late-game
        targeting_multiplier = 1.0 + (targeting_score * understanding * 0.5)

        return base * difficulty_factor * targeting_multiplier

    def _compute_targeting_score(self, question: str) -> float:
        """How well does the question target active misconceptions?"""
        if not self.active_misconceptions:
            return 0.1  # No misconceptions left

        q_lower = question.lower()
        hits = 0
        for phrase in self.active_misconceptions:
            # Check if any word from the misconception phrase appears
            words = phrase.lower().split()
            word_hits = sum(1 for w in words if w in q_lower)
            if word_hits >= len(words) * 0.5:
                hits += 1

        return min(1.0, hits / max(1, len(self.active_misconceptions)))

    def _weaken_misconceptions(self, question: str):
        """Remove misconceptions that were targeted by the question."""
        q_lower = question.lower()
        remaining = []
        for phrase in self.active_misconceptions:
            words = phrase.lower().split()
            word_hits = sum(1 for w in words if w in q_lower)
            # Keep the misconception if less than half its words were targeted
            if word_hits < len(words) * 0.5:
                remaining.append(phrase)
        self.active_misconceptions = remaining

    def _generate_response(self) -> str:
        """
        Generate a deterministic, contextual response based on understanding level.
        
        Fix 10: Supports variable interpolation in response templates.
        Templates can use: {target_question}, {misconception}, {step_count},
        {remaining_misconceptions}, {concept_description}.
        """
        state = self.get_state()
        confidence = state.confidence_label
        responses = self.concept.responses.get(confidence, [])

        if not responses:
            return f"I'm at the '{confidence}' stage but I'm not sure what to say."

        # Deterministic: use step_count to select response (not random)
        idx = (self.step_count - 1) % len(responses)
        template = responses[idx]

        # Variable interpolation (Fix 10)
        try:
            remaining = len(self.active_misconceptions)
            template = template.format(
                target_question=self.concept.target_question,
                misconception=self.concept.initial_misconception,
                step_count=self.step_count,
                remaining_misconceptions=remaining,
                concept_description=self.concept.description,
            )
        except (KeyError, IndexError):
            pass  # Template doesn't use variables — that's fine

        return template

    @property
    def is_done(self) -> bool:
        return self.understanding_level >= 0.85 or self.step_count >= self.max_steps

    @property
    def success(self) -> bool:
        return self.understanding_level >= 0.85
