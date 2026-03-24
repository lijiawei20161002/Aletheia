"""
CoTShield Evaluations - Adversarial tasks for testing model honesty.
"""

from .tasks import (
    AdversarialTaskSuite,
    TaskEvaluator,
    EvalTask,
    EvalResult,
    TaskCategory
)

__all__ = [
    "AdversarialTaskSuite",
    "TaskEvaluator",
    "EvalTask",
    "EvalResult",
    "TaskCategory"
]
