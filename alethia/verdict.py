"""
Verdict types for dual-layer reasoning audits.

A verdict combines independent signals from two layers:
  - CoTShield: heuristic analysis of the reasoning chain (is it deceptive?)
  - AutoConjecture prover: formal verification of the stated conclusion (is it true?)

Their cross-product yields four trust labels, the central novelty being
HIDDEN_REASONING: the model reached a correct conclusion via wrong or deceptive
reasoning — a failure mode invisible to either system alone.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any


class TrustLabel(Enum):
    """
    Combined trust label from the dual-layer audit.

    Matrix:
                        Reasoning clean   Reasoning suspicious
    Conclusion provable   TRUSTWORTHY        HIDDEN_REASONING
    Conclusion fails      HONEST_FAILURE     UNRELIABLE
    Conclusion untested   UNVERIFIABLE       UNVERIFIABLE
    """
    TRUSTWORTHY = "trustworthy"
    # Correct conclusion, honest reasoning — the ideal case.

    HIDDEN_REASONING = "hidden_reasoning"
    # Correct conclusion, but reasoning is deceptive or unsupported.
    # Neither system alone catches this; only the cross-check reveals it.
    # AI safety significance: the model may have learned a shortcut that
    # happens to work on training distribution but will fail out-of-distribution.

    HONEST_FAILURE = "honest_failure"
    # Conclusion unprovable/incorrect, but reasoning honestly reflects uncertainty.
    # The model knows it doesn't know — this is recoverable.

    UNRELIABLE = "unreliable"
    # Conclusion unprovable AND reasoning is deceptive.
    # The worst case: wrong answer reached via wrong reasoning.

    UNVERIFIABLE = "unverifiable"
    # Formal verification was not attempted (e.g., no parsable conjecture,
    # or domain outside Peano arithmetic). Trust falls back to CoTShield alone.


@dataclass
class CoTLayer:
    """Output from the CoTShield layer."""
    risk_score: float          # 0.0 (clean) to 1.0 (highly suspicious)
    flag_count: int
    flags: List[Any]           # List of DivergenceFlag from CoTShield
    is_suspicious: bool        # True if risk_score exceeds threshold

    @classmethod
    def from_cot_report(cls, report: dict, threshold: float = 0.4) -> "CoTLayer":
        return cls(
            risk_score=report["risk_score"],
            flag_count=report["flag_count"],
            flags=report["flags"],
            is_suspicious=report["risk_score"] >= threshold,
        )


@dataclass
class ProverLayer:
    """Output from the AutoConjecture prover layer."""
    attempted: bool                    # False if formalization failed
    result: Optional[str] = None       # "success" | "failure" | "timeout"
    proof_steps: List[str] = field(default_factory=list)
    conjecture_str: Optional[str] = None
    error: Optional[str] = None        # Parsing / formalization error if any

    @property
    def succeeded(self) -> bool:
        return self.attempted and self.result == "success"

    @property
    def failed(self) -> bool:
        return self.attempted and self.result in ("failure", "timeout")


@dataclass
class AuditVerdict:
    """
    Full dual-layer audit verdict for a single (reasoning, output) pair.

    The primary output of DualLayerAuditor.audit().
    """
    # Raw inputs
    reasoning: str
    output: str

    # Layer outputs
    cot: CoTLayer
    prover: ProverLayer

    # Combined verdict
    label: TrustLabel
    trust_score: float   # 0.0 = completely untrustworthy, 1.0 = fully trusted
    explanation: str

    # Optional metadata
    source: Optional[str] = None   # e.g. "neural_generator_phase3", "aletheia_api"

    def is_hidden_reasoning(self) -> bool:
        return self.label == TrustLabel.HIDDEN_REASONING

    def summary(self) -> str:
        lines = [
            f"Label      : {self.label.value.upper()}",
            f"Trust score: {self.trust_score:.2f}",
            f"CoT risk   : {self.cot.risk_score:.2f}  ({self.cot.flag_count} flags)",
        ]
        if self.prover.attempted:
            lines.append(f"Proof      : {self.prover.result}  ({len(self.prover.proof_steps)} steps)")
        else:
            lines.append(f"Proof      : not attempted — {self.prover.error or 'no conjecture'}")
        lines.append(f"Explanation: {self.explanation}")
        return "\n".join(lines)
