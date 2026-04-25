"""
Concept bank loader.

Loads concept JSON files from the concepts/ directory and provides
lookup by concept_id. Optionally pre-computes embeddings for 
question templates using sentence-transformers.
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from models import Concept

logger = logging.getLogger(__name__)


def _word_overlap_similarity(text_a: str, text_b: str) -> float:
    """Pure-Python word overlap similarity (Jaccard-style but improved)."""
    stop_words = {"a", "an", "the", "is", "are", "do", "does", "in", "on",
                  "of", "to", "and", "or", "it", "that", "this", "for", "you",
                  "your", "what", "how", "why", "when", "where", "can", "could",
                  "would", "should", "has", "have", "had", "be", "been", "was"}
    words_a = set(text_a.lower().split()) - stop_words
    words_b = set(text_b.lower().split()) - stop_words
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


class ConceptEmbeddings:
    """
    Pre-computed embeddings for concept question templates.
    
    Fallback chain:
    1. sentence-transformers (best — semantic similarity)
    2. sklearn TF-IDF (good — term frequency cosine)
    3. Pure-Python word overlap (always works — no dependencies)
    """

    def __init__(self):
        self._model = None
        self._cache = {}
        self._use_st = False
        self._use_tfidf = False
        self._tfidf = None
        self._tfidf_matrices = {}
        self._template_texts: dict[str, list[str]] = {}
        self._load_model()

    def _load_model(self):
        """Try sentence-transformers → TF-IDF → word overlap."""
        # Tier 1: sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_st = True
            logger.info("ConceptEmbeddings: Using sentence-transformers (all-MiniLM-L6-v2)")
            return
        except (ImportError, Exception) as e:
            logger.info(f"sentence-transformers not available: {e}")

        # Tier 2: sklearn TF-IDF
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(stop_words="english")
            self._use_tfidf = True
            logger.info("ConceptEmbeddings: Using sklearn TF-IDF")
            return
        except ImportError:
            pass

        # Tier 3: Pure Python word overlap
        logger.info("ConceptEmbeddings: Using pure-Python word overlap (no ML dependencies)")

    def embed_concept(self, concept: Concept):
        """Pre-compute embeddings for a concept's good question templates."""
        templates = concept.good_question_templates
        if not templates:
            return

        # Always store raw templates for word overlap fallback
        self._template_texts[concept.concept_id] = list(templates)

        if self._use_st and _HAS_NUMPY:
            self._cache[concept.concept_id] = self._model.encode(
                templates, normalize_embeddings=True
            )
        elif self._use_tfidf:
            try:
                matrix = self._tfidf.fit_transform(templates)
                self._tfidf_matrices[concept.concept_id] = matrix
            except Exception:
                pass  # Will fall back to word overlap

    def similarity(self, question: str, concept_id: str) -> float:
        """
        Compute max similarity between question and concept's templates.
        Uses best available method.
        """
        # Tier 1: sentence-transformers
        if self._use_st and _HAS_NUMPY and concept_id in self._cache:
            q_emb = self._model.encode([question], normalize_embeddings=True)
            templates = self._cache[concept_id]
            sims = np.dot(q_emb, templates.T).flatten()
            return float(np.max(sims)) if len(sims) > 0 else 0.0

        # Tier 2: TF-IDF
        if self._use_tfidf and concept_id in self._tfidf_matrices:
            try:
                q_vec = self._tfidf.transform([question])
                template_matrix = self._tfidf_matrices[concept_id]
                from sklearn.metrics.pairwise import cosine_similarity
                sims = cosine_similarity(q_vec, template_matrix).flatten()
                return float(max(sims)) if len(sims) > 0 else 0.0
            except Exception:
                pass  # Fall through to word overlap

        # Tier 3: Word overlap
        templates = self._template_texts.get(concept_id, [])
        if not templates:
            return 0.0
        scores = [_word_overlap_similarity(question, t) for t in templates]
        return max(scores) if scores else 0.0


class ConceptBank:
    """
    Bank of concepts loaded from JSON files.
    Pre-computes embeddings for question templates.
    """

    def __init__(self, concepts: dict[str, Concept], embeddings: ConceptEmbeddings):
        self.concepts = concepts
        self.embeddings = embeddings

    @classmethod
    def load(cls, concepts_dir: str = "./concepts") -> "ConceptBank":
        """
        Load all concept JSON files from the given directory.

        Args:
            concepts_dir: Path to directory containing concept JSON files.

        Returns:
            ConceptBank with all concepts loaded and embeddings pre-computed.
        """
        concepts_path = Path(concepts_dir)
        if not concepts_path.exists():
            raise FileNotFoundError(f"Concepts directory not found: {concepts_dir}")

        concepts: dict[str, Concept] = {}
        embeddings = ConceptEmbeddings()

        for json_file in sorted(concepts_path.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                concept = Concept(**data)
                concepts[concept.concept_id] = concept
                embeddings.embed_concept(concept)
                logger.info(f"Loaded concept: {concept.concept_id} ({concept.difficulty})")
            except Exception as e:
                logger.error(f"Failed to load concept from {json_file}: {e}")
                raise

        logger.info(f"ConceptBank loaded: {len(concepts)} concepts")
        return cls(concepts=concepts, embeddings=embeddings)

    def get(self, concept_id: str) -> Concept:
        """Get a concept by ID. Raises KeyError if not found."""
        if concept_id not in self.concepts:
            raise KeyError(f"Unknown concept: {concept_id}. Available: {list(self.concepts.keys())}")
        return self.concepts[concept_id]

    def get_by_difficulty(self, difficulty: str) -> list[Concept]:
        """Get all concepts matching a difficulty level."""
        return [c for c in self.concepts.values() if c.difficulty == difficulty]

    def template_similarity(self, question: str, concept_id: str) -> float:
        """Compute semantic similarity between a question and a concept's templates."""
        return self.embeddings.similarity(question, concept_id)
