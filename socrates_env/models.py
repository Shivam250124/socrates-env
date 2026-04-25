"""
Data models for the SOCRATES environment.

Shared between client and server — no server imports here.
Defines Action, Observation, and State using Pydantic BaseModel.
"""

from pydantic import BaseModel, Field
from typing import Optional


class SocratesAction(BaseModel):
    """
    Agent's action: a Socratic question to ask the student.
    
    The agent should ask exactly ONE open-ended question per turn
    that guides the student toward understanding without revealing the answer.
    """
    
    question: str = Field(
        description=(
            "The Socratic question to ask the student. Must be a single question "
            "that guides without revealing the answer."
        )
    )
    
    question_type: str = Field(
        default="socratic",
        description="Hint for question intent: 'socratic', 'counterexample', or 'meta'"
    )


class ConversationTurn(BaseModel):
    """A single turn in the tutoring conversation."""
    step: int
    agent_question: str
    student_response: str
    student_confidence: str


class SocratesObservation(BaseModel):
    """
    What the agent sees at each step.
    
    Note: understanding_level is intentionally hidden from the agent.
    The agent only sees the student's stated confidence level, not the
    internal scalar — just like a real tutor.
    """
    
    concept_description: str = Field(
        description="What the student is trying to understand"
    )
    
    student_current_belief: str = Field(
        description="The student's initial misconception, stated as their perspective"
    )
    
    student_response: str = Field(
        description="Student's response to the agent's last question"
    )
    
    student_confidence: str = Field(
        default="confused",
        description="Student's apparent confidence: confused, uncertain, starting_to_see, almost_there, understood"
    )
    
    steps_remaining: int = Field(
        description="Steps left in the episode"
    )
    
    history: list[ConversationTurn] = Field(
        default_factory=list,
        description="Full conversation history"
    )
    
    done: bool = Field(
        default=False,
        description="Whether the episode has ended"
    )
    
    success: bool = Field(
        default=False,
        description="Whether the student reached understanding"
    )


class SocratesState(BaseModel):
    """
    Full environment state for debugging (not shown to agent during training).
    Includes the internal understanding_level scalar.
    """
    
    concept_id: str = Field(default="")
    understanding_level: float = Field(default=0.0)
    active_misconceptions: list[str] = Field(default_factory=list)
    episode_history: list[dict] = Field(default_factory=list)
    step_count: int = Field(default=0)
    done: bool = Field(default=False)
    success: bool = Field(default=False)
    episode_num: int = Field(default=0)
    cumulative_compliance: float = Field(
        default=0.0,
        description="Sum of Socratic compliance scores across the episode"
    )
    empty: bool = Field(default=False)


class Concept(BaseModel):
    """A single concept from the concept bank."""
    
    concept_id: str
    difficulty: str
    target_question: str
    description: str
    initial_misconception: str
    initial_student_statement: str
    correct_understanding: str
    answer_keywords: list[str]
    concept_keywords: list[str]
    good_question_templates: list[str]
    
    responses: dict[str, list[str]] = Field(
        description="Keyed by confidence level, each maps to a list of response templates"
    )
    
    misconception_phrases: list[str] = Field(
        default_factory=list,
        description="Key phrases from the misconception for targeting detection"
    )
    
    min_steps_to_success: int = Field(
        default=4,
        description="Estimated minimum steps from hand-authored traces"
    )
