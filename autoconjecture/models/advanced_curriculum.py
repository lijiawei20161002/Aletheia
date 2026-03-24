"""
Advanced curriculum strategies for Phase 5.

Three strategies beyond the Phase 3 linear curriculum:

1. SelfPacedCurriculum  – focuses on the "learning frontier" using per-bucket
                          exponential-moving-average (EMA) success rates.
2. AdaptiveBandCurriculum – maintains a sliding complexity window that expands
                            or contracts based on recent success rate.
3. PrioritizedExperienceBuffer – stores proven/failed conjecture strings and
                                 samples them by difficulty for targeted training.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random


# ── 1. Self-Paced Curriculum ──────────────────────────────────────────────────

@dataclass
class SelfPacedConfig:
    """Configuration for SelfPacedCurriculum."""
    initial_complexity: int = 2
    final_complexity: int = 20
    # EMA discount factor per update (0 < alpha <= 1; smaller = slower decay)
    ema_alpha: float = 0.1
    # Target success-rate band; the frontier is the highest bucket inside it
    target_lo: float = 0.10
    target_hi: float = 0.60
    # How far above the frontier we are willing to generate (exploration margin)
    frontier_margin: int = 3
    # Minimum observations before using EMA (use prior until then)
    min_obs: int = 20
    # Prior success rate assumed before any data
    prior_success: float = 0.3


class SelfPacedCurriculum:
    """
    Self-paced curriculum that focuses on the learning frontier.

    Each integer complexity bucket tracks an EMA success rate.
    The "frontier" is the highest-complexity bucket whose EMA success rate
    falls in [target_lo, target_hi].  Conjectures are generated up to
    ``frontier + frontier_margin``.
    """

    def __init__(self, config: SelfPacedConfig):
        self.config = config
        # EMA success rate per complexity bucket
        self._ema: Dict[int, float] = {}
        # Raw observation counts per bucket (for min_obs guard)
        self._obs: Dict[int, int] = {}

    # ── public API ────────────────────────────────────────────────────────

    def record_result(self, success: bool, complexity: int) -> None:
        """Update EMA for the given complexity bucket."""
        c = max(self.config.initial_complexity,
                min(self.config.final_complexity, int(complexity)))
        alpha = self.config.ema_alpha
        prev = self._ema.get(c, self.config.prior_success)
        self._ema[c] = alpha * float(success) + (1 - alpha) * prev
        self._obs[c] = self._obs.get(c, 0) + 1

    def get_frontier(self) -> int:
        """Return the current learning frontier complexity."""
        frontier = self.config.initial_complexity
        for c in range(self.config.initial_complexity,
                       self.config.final_complexity + 1):
            rate = self._ema.get(c, self.config.prior_success)
            obs = self._obs.get(c, 0)
            if obs < self.config.min_obs:
                # Not enough data – treat as frontier (explore it)
                frontier = c
                break
            if self.config.target_lo <= rate <= self.config.target_hi:
                frontier = c
        return frontier

    def get_complexity_range(self) -> Tuple[int, int]:
        """Return (min_complexity, max_complexity) for conjecture generation."""
        frontier = self.get_frontier()
        lo = max(self.config.initial_complexity, frontier - 1)
        hi = min(self.config.final_complexity,
                 frontier + self.config.frontier_margin)
        return lo, hi

    def filter_by_complexity(self, expressions) -> list:
        """Keep only expressions in the current complexity range."""
        lo, hi = self.get_complexity_range()
        return [e for e in expressions if lo <= e.complexity() <= hi]

    def get_statistics(self) -> dict:
        frontier = self.get_frontier()
        lo, hi = self.get_complexity_range()
        return {
            "frontier": frontier,
            "complexity_range": (lo, hi),
            "ema_rates": dict(sorted(self._ema.items())),
        }


# ── 2. Adaptive Band Curriculum ───────────────────────────────────────────────

@dataclass
class AdaptiveBandConfig:
    """Configuration for AdaptiveBandCurriculum."""
    initial_complexity: int = 2
    final_complexity: int = 20
    # Initial window half-width (band = [center - hw, center + hw])
    initial_halfwidth: int = 2
    # Advance center when success rate in window > advance_threshold
    advance_threshold: float = 0.40
    # Retreat center when success rate < retreat_threshold
    retreat_threshold: float = 0.05
    # Widen band when advance_threshold not reached after patience steps
    patience: int = 200
    # Maximum band half-width
    max_halfwidth: int = 5
    # Window length for rolling success rate
    window: int = 100


class AdaptiveBandCurriculum:
    """
    Adaptive-band curriculum: maintains a sliding complexity window.

    The center advances when the prover is succeeding, retreats when it
    struggles, and the band widens when progress stalls for too long.
    """

    def __init__(self, config: AdaptiveBandConfig):
        self.config = config
        self.center = float(config.initial_complexity)
        self.halfwidth = config.initial_halfwidth
        self._history: deque = deque(maxlen=config.window)
        self._steps_since_advance = 0

    # ── public API ────────────────────────────────────────────────────────

    def record_result(self, success: bool, complexity: int) -> None:
        self._history.append(int(success))
        self._steps_since_advance += 1
        self._maybe_update()

    def get_complexity_range(self) -> Tuple[int, int]:
        lo = max(self.config.initial_complexity,
                 int(self.center - self.halfwidth))
        hi = min(self.config.final_complexity,
                 int(self.center + self.halfwidth))
        return lo, hi

    def filter_by_complexity(self, expressions) -> list:
        lo, hi = self.get_complexity_range()
        return [e for e in expressions if lo <= e.complexity() <= hi]

    def get_statistics(self) -> dict:
        lo, hi = self.get_complexity_range()
        rate = sum(self._history) / max(len(self._history), 1)
        return {
            "center": self.center,
            "halfwidth": self.halfwidth,
            "complexity_range": (lo, hi),
            "rolling_success_rate": rate,
            "steps_since_advance": self._steps_since_advance,
        }

    # ── internal ──────────────────────────────────────────────────────────

    def _maybe_update(self) -> None:
        if len(self._history) < self.config.window // 2:
            return
        rate = sum(self._history) / len(self._history)

        if rate >= self.config.advance_threshold:
            # Advance center
            step = max(1, self.halfwidth // 2)
            self.center = min(
                self.config.final_complexity,
                self.center + step,
            )
            # Narrow band slightly after success
            self.halfwidth = max(1, self.halfwidth - 1)
            self._steps_since_advance = 0

        elif rate < self.config.retreat_threshold:
            # Retreat center
            self.center = max(
                self.config.initial_complexity,
                self.center - 1,
            )
            # Widen band for more exploration
            self.halfwidth = min(self.config.max_halfwidth,
                                 self.halfwidth + 1)
            self._steps_since_advance = 0

        elif self._steps_since_advance >= self.config.patience:
            # Stalled – widen band
            self.halfwidth = min(self.config.max_halfwidth,
                                 self.halfwidth + 1)
            self._steps_since_advance = 0


# ── 3. Prioritized Experience Buffer ─────────────────────────────────────────

@dataclass
class ExperienceEntry:
    """One entry in the prioritized buffer."""
    expr_str: str
    complexity: float
    success: bool
    proof_length: int
    # Priority score (higher = more likely to be sampled for generator training)
    priority: float = 1.0


class PrioritizedExperienceBuffer:
    """
    Stores recent conjecture attempts with associated difficulty scores.

    Sampling is proportional to priority so the generator trains more on
    boundary examples (medium difficulty) rather than trivially easy or
    hopelessly hard ones.

    Priority formula:
        priority = exp(-|success_rate(bucket) - target_rate|)
    where target_rate is the desired midpoint of the learning zone.
    """

    def __init__(
        self,
        maxlen: int = 2000,
        target_success_rate: float = 0.30,
        ema_alpha: float = 0.15,
    ):
        self.maxlen = maxlen
        self.target_success_rate = target_success_rate
        self.ema_alpha = ema_alpha
        self._buffer: deque = deque(maxlen=maxlen)
        self._bucket_ema: Dict[int, float] = {}

    def add(
        self,
        expr_str: str,
        complexity: float,
        success: bool,
        proof_length: int = 0,
    ) -> None:
        """Record one conjecture attempt."""
        import math
        bucket = int(complexity)
        prev = self._bucket_ema.get(bucket, self.target_success_rate)
        self._bucket_ema[bucket] = (
            self.ema_alpha * float(success) + (1 - self.ema_alpha) * prev
        )
        rate = self._bucket_ema[bucket]
        priority = math.exp(-abs(rate - self.target_success_rate))

        self._buffer.append(
            ExperienceEntry(
                expr_str=expr_str,
                complexity=complexity,
                success=success,
                proof_length=proof_length,
                priority=priority,
            )
        )

    def sample(self, n: int, successes_only: bool = False) -> List[ExperienceEntry]:
        """
        Sample n entries proportional to priority.

        Args:
            n: Number of entries to sample.
            successes_only: If True, only sample proven conjectures.
        """
        pool = [e for e in self._buffer if (not successes_only or e.success)]
        if not pool:
            return []
        weights = [e.priority for e in pool]
        k = min(n, len(pool))
        return random.choices(pool, weights=weights, k=k)

    def successful_exprs(self, n: int = 50) -> List[str]:
        """Return up to n high-priority successfully proven expression strings."""
        entries = self.sample(n, successes_only=True)
        return [e.expr_str for e in entries]

    def size(self) -> int:
        return len(self._buffer)

    def get_statistics(self) -> dict:
        total = len(self._buffer)
        successes = sum(1 for e in self._buffer if e.success)
        return {
            "buffer_size": total,
            "success_count": successes,
            "success_rate": successes / max(total, 1),
            "bucket_ema": dict(sorted(self._bucket_ema.items())),
        }
