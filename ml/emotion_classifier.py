"""
MANORA ML Emotion Classifier Abstraction.
Performs raw emotion detection and probability estimation from student text.
Pluggable architecture supporting local lexicon/heuristics and HuggingFace transformers.
"""

import abc
import logging
import re
from typing import Dict, Optional

from config.settings import get_settings

logger = logging.getLogger("manora.ml")


class BaseEmotionClassifier(abc.ABC):
    """Abstract base class for emotion classifiers."""

    @abc.abstractmethod
    def predict(self, text: str) -> Dict[str, float]:
        """
        Analyzes input text and returns a dictionary of emotion probabilities.
        Each value is bounded between 0.0 and 1.0.
        """
        pass


class RuleAndLexiconEmotionClassifier(BaseEmotionClassifier):
    """
    Lightweight rule-and-lexicon classifier tailored for student mental health contexts.
    Detects signals for: frustration, guilt, stress, sadness, anxiety, fatigue, motivation, happiness.
    """

    LEXICON = {
        "frustration": [
            "frustrated", "annoyed", "irritated", "stuck", "angry", "hate this", "giving up",
            "waste of time", "keeps happening", "again", "failing", "cannot focus", "distracted"
        ],
        "guilt": [
            "guilt", "guilty", "should have", "could have", "my fault", "wasted time",
            "ashamed", "regret", "procrastinated", "netflix", "series", "avoided", "skipped"
        ],
        "stress": [
            "stress", "stressed", "overwhelmed", "deadline", "pressure", "burden",
            "placements", "exams", "finals", "too much", "running out of time"
        ],
        "anxiety": [
            "anxious", "anxiety", "worried", "scared", "nervous", "panic", "fear",
            "what if", "uncertain", "dread", "terrified", "insecure"
        ],
        "sadness": [
            "sad", "depressed", "unhappy", "down", "hopeless", "lonely", "miserable",
            "crying", "lost", "empty", "gloomy", "heartbroken"
        ],
        "fatigue": [
            "tired", "sleepy", "exhausted", "drained", "no energy", "burnout", "burnt out",
            "sleep", "lazy", "sluggish", "weary"
        ],
        "motivation": [
            "motivated", "ready", "excited", "start", "focus", "achieve", "goal",
            "progress", "determined", "inspired", "confident", "productive"
        ],
        "happiness": [
            "happy", "glad", "great", "relieved", "good", "wonderful", "proud",
            "enjoyed", "calm", "content", "peaceful", "better"
        ]
    }

    def predict(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))
        results: Dict[str, float] = {}

        for emotion, keywords in self.LEXICON.items():
            score = 0.0
            for kw in keywords:
                if " " in kw:
                    if kw in text_lower:
                        score += 0.4
                elif kw in words:
                    score += 0.3

            # Add baseline small probability
            base_prob = 0.05
            calculated = min(1.0, base_prob + score)
            results[emotion] = round(calculated, 3)

        # Normalize or ensure primary signals stand out
        return results


class TransformerEmotionClassifier(BaseEmotionClassifier):
    """
    HuggingFace transformer pipeline classifier (e.g. distilbert / roberta emotion models).
    Loaded when transformers and torch are installed in the environment.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-emotion"):
        self.model_name = model_name
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
            )
            logger.info(f"Loaded HuggingFace model {self.model_name} successfully.")
        except Exception as e:
            logger.warning(f"Could not load HuggingFace pipeline ({e}). Will use fallback.")
            self.pipeline = None

    def predict(self, text: str) -> Dict[str, float]:
        if not self.pipeline:
            return RuleAndLexiconEmotionClassifier().predict(text)

        try:
            raw_results = self.pipeline(text[:512])
            # raw_results format: [[{'label': 'sadness', 'score': 0.8}, ...]]
            probs = {}
            if raw_results and isinstance(raw_results, list):
                items = raw_results[0] if isinstance(raw_results[0], list) else raw_results
                for item in items:
                    label = item.get("label", "").lower()
                    score = float(item.get("score", 0.0))
                    probs[label] = round(score, 3)
            return probs
        except Exception as e:
            logger.error(f"Transformer inference error: {e}. Falling back to rule classifier.")
            return RuleAndLexiconEmotionClassifier().predict(text)


class EmotionClassifier(BaseEmotionClassifier):
    """
    Main EmotionClassifier entrypoint.
    Delegates to transformer model if available and configured, otherwise uses the
    optimized rule and lexicon classifier.
    """

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMOTION_MODEL_NAME
        self.fallback = RuleAndLexiconEmotionClassifier()
        self._underlying: BaseEmotionClassifier

        # Attempt to load transformer if requested and libraries available
        try:
            import transformers
            import torch
            self._underlying = TransformerEmotionClassifier(self.model_name)
        except ImportError:
            logger.info("Transformers/Torch not installed. Using RuleAndLexiconEmotionClassifier.")
            self._underlying = self.fallback

    def predict(self, text: str) -> Dict[str, float]:
        """Runs emotion prediction on text."""
        if not text or not text.strip():
            return {k: 0.05 for k in RuleAndLexiconEmotionClassifier.LEXICON.keys()}
        return self._underlying.predict(text)


# Global singleton instance
emotion_classifier = EmotionClassifier()
