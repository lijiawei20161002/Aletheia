"""
Curriculum learning strategies for neural generator.
Gradually increases difficulty from simple to complex conjectures.
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..logic.expressions import Expression


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    # Complexity stages
    initial_complexity: int = 2
    final_complexity: int = 15
    complexity_increment: int = 2

    # Progression strategy
    progression_metric: str = "success_rate"  # "success_rate" or "steps"
    success_threshold: float = 0.3  # Move to next stage when success rate > threshold
    min_samples_per_stage: int = 100  # Minimum samples before advancing

    # Temperature scheduling
    initial_temperature: float = 0.5  # Lower temperature for undertrained models
    final_temperature: float = 0.3   # Even lower for focused generation


class CurriculumScheduler:
    """
    Manages curriculum learning progression.

    Gradually increases the complexity of generated conjectures
    based on the model's success rate at proving them.
    """

    def __init__(self, config: CurriculumConfig):
        """
        Args:
            config: Curriculum configuration
        """
        self.config = config

        # Current curriculum state
        self.current_complexity = config.initial_complexity
        self.current_stage = 0

        # Statistics for current stage
        self.stage_samples = 0
        self.stage_successes = 0
        self.stage_failures = 0

        # History
        self.stage_history = []

    def get_current_complexity_range(self) -> Tuple[int, int]:
        """
        Get the current complexity range for generation.

        Returns:
            (min_complexity, max_complexity)
        """
        min_complexity = max(1, self.current_complexity - 1)
        max_complexity = self.current_complexity + 2
        return min_complexity, max_complexity

    def get_current_temperature(self) -> float:
        """
        Get the current sampling temperature.

        Temperature decreases as complexity increases (exploration -> exploitation).

        Returns:
            Current temperature
        """
        total_stages = (
            (self.config.final_complexity - self.config.initial_complexity) //
            self.config.complexity_increment
        )

        if total_stages <= 0:
            return self.config.initial_temperature

        # Linear interpolation
        progress = min(1.0, self.current_stage / total_stages)
        temperature = (
            self.config.initial_temperature * (1 - progress) +
            self.config.final_temperature * progress
        )

        return temperature

    def record_result(self, success: bool, complexity: int):
        """
        Record the result of a proof attempt.

        Args:
            success: Whether the proof succeeded
            complexity: Complexity of the conjecture
        """
        self.stage_samples += 1

        if success:
            self.stage_successes += 1
        else:
            self.stage_failures += 1

    def should_advance_stage(self) -> bool:
        """
        Determine if curriculum should advance to next stage.

        Returns:
            True if should advance, False otherwise
        """
        # Need minimum samples
        if self.stage_samples < self.config.min_samples_per_stage:
            return False

        # Check success rate
        success_rate = self.stage_successes / max(self.stage_samples, 1)

        if success_rate >= self.config.success_threshold:
            return True

        return False

    def advance_stage(self):
        """Advance to next curriculum stage."""
        # Record current stage statistics
        self.stage_history.append({
            'stage': self.current_stage,
            'complexity': self.current_complexity,
            'samples': self.stage_samples,
            'successes': self.stage_successes,
            'success_rate': self.stage_successes / max(self.stage_samples, 1)
        })

        # Advance
        self.current_stage += 1
        self.current_complexity = min(
            self.config.final_complexity,
            self.current_complexity + self.config.complexity_increment
        )

        # Reset stage statistics
        self.stage_samples = 0
        self.stage_successes = 0
        self.stage_failures = 0

        print(f"\nCurriculum advanced to stage {self.current_stage}")
        print(f"  Complexity range: {self.get_current_complexity_range()}")
        print(f"  Temperature: {self.get_current_temperature():.2f}")

    def is_complete(self) -> bool:
        """Check if curriculum has reached final stage."""
        return self.current_complexity >= self.config.final_complexity

    def get_statistics(self) -> Dict:
        """Get curriculum statistics."""
        current_success_rate = self.stage_successes / max(self.stage_samples, 1)

        return {
            'current_stage': self.current_stage,
            'current_complexity': self.current_complexity,
            'complexity_range': self.get_current_complexity_range(),
            'current_temperature': self.get_current_temperature(),
            'stage_samples': self.stage_samples,
            'stage_successes': self.stage_successes,
            'stage_success_rate': current_success_rate,
            'is_complete': self.is_complete(),
            'history': self.stage_history
        }

    def filter_by_complexity(
        self,
        expressions: List[Expression],
        allow_margin: bool = True
    ) -> List[Expression]:
        """
        Filter expressions to match current curriculum stage.

        Args:
            expressions: Expressions to filter
            allow_margin: Allow expressions slightly outside range

        Returns:
            Filtered expressions
        """
        min_complexity, max_complexity = self.get_current_complexity_range()

        if allow_margin:
            # Allow some margin for diversity
            min_complexity = max(1, min_complexity - 1)
            max_complexity = max_complexity + 1

        filtered = [
            expr for expr in expressions
            if min_complexity <= expr.complexity() <= max_complexity
        ]

        return filtered

    def sample_by_stage(
        self,
        expressions: List[Expression],
        n_samples: int
    ) -> List[Expression]:
        """
        Sample expressions appropriate for current stage.

        Args:
            expressions: Pool of expressions
            n_samples: Number of samples to draw

        Returns:
            Sampled expressions
        """
        # Filter by complexity
        filtered = self.filter_by_complexity(expressions)

        if not filtered:
            return []

        # Sample
        if len(filtered) <= n_samples:
            return filtered

        indices = np.random.choice(len(filtered), size=n_samples, replace=False)
        return [filtered[i] for i in indices]

    def reset(self):
        """Reset curriculum to initial stage."""
        self.current_complexity = self.config.initial_complexity
        self.current_stage = 0
        self.stage_samples = 0
        self.stage_successes = 0
        self.stage_failures = 0
        self.stage_history = []


class AdaptiveCurriculum(CurriculumScheduler):
    """
    Adaptive curriculum that adjusts based on performance.

    Can increase or decrease difficulty based on success rate.
    """

    def __init__(self, config: CurriculumConfig):
        super().__init__(config)
        self.decrease_threshold = 0.1  # Decrease complexity if success rate < threshold

    def update(self):
        """Update curriculum based on current performance."""
        if self.stage_samples < self.config.min_samples_per_stage:
            return

        success_rate = self.stage_successes / max(self.stage_samples, 1)

        # Too hard - decrease complexity
        if success_rate < self.decrease_threshold:
            if self.current_complexity > self.config.initial_complexity:
                self.current_complexity = max(
                    self.config.initial_complexity,
                    self.current_complexity - self.config.complexity_increment
                )
                print(f"Curriculum difficulty decreased to {self.current_complexity}")
                self._reset_stage_stats()

        # Just right - advance
        elif success_rate >= self.config.success_threshold:
            self.advance_stage()

    def _reset_stage_stats(self):
        """Reset stage statistics without recording to history."""
        self.stage_samples = 0
        self.stage_successes = 0
        self.stage_failures = 0
