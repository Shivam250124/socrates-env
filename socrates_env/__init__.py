"""
SOCRATES: Socratic Teaching Agent RL Environment

Train LLMs to teach like Socrates — through questions, never answers.
"""

from models import SocratesAction, SocratesObservation, SocratesState
from client import SocratesEnv

__all__ = [
    "SocratesAction",
    "SocratesObservation",
    "SocratesState",
    "SocratesEnv",
]
