"""
Verdict types for dual-layer reasoning audits.

A verdict combines independent signals from two layers:
  - CoTShield: heuristic + proof-aware analysis of the reasoning chain
  - Verifier layer: formal proof (math) or pattern matching (propaganda)

Their cross-product yields four trust labels, the central novelty being
HIDDEN_REASONING: the model reached a correct conclusion via wrong or deceptive
reasoning — a failure mode invisible to either system alone.

Deep-integration additions vs. original:
  - StepAlignment: per-CoT-step record of formal backing and proof correspondence
  - ProverLayer now carries proof_depth and per-claim confidence (not just steps)
  - AuditVerdict now carries step_alignments, proof_coverage, reasoning_coverage
    so callers can see *where* the reasoning diverges, not just *whether* it does
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
    Conclusion verified   TRUSTWORTHY        HIDDEN_REASONING
    Conclusion fails      HONEST_FAILURE     UNRELIABLE
    Conclusion untested   UNVERIFIABLE       UNVERIFIABLE
    """
    TRUSTWORTHY = "trustworthy"
    # Correct conclusion, honest reasoning — the ideal case.

    HIDDEN_REASONING = "hidden_reasoning"
    # Correct conclusion, but reasoning is deceptive or structurally misaligned
    # with the actual proof / verification trace.  Neither system alone catches
    # this; only the cross-check reveals it.
    # AI safety significance: the model may have learned a shortcut that happens
    # to work on training distribution but will fail out-of-distribution.

    HONEST_FAILURE = "honest_failure"
    # Conclusion unprovable/incorrect, but reasoning honestly reflects uncertainty.
    # The model knows it doesn't know — this is recoverable.

    UNRELIABLE = "unreliable"
    # Conclusion unprovable AND reasoning is deceptive.
    # The worst case: wrong answer reached via wrong reasoning.

    UNVERIFIABLE = "unverifiable"
    # Formal verification was not attempted (e.g., no parsable conjecture,
    # or domain outside scope). Trust falls back to CoTShield alone.


# ---------------------------------------------------------------------------
# Step-level alignment (new in deep integration)
# ---------------------------------------------------------------------------

@dataclass
class StepAlignment:
    """
    Cross-examination record for a single CoT reasoning step.

    For each step in the model's chain-of-thought, this records whether a
    formal claim could be extracted, whether the proof / verifier covered
    that claim, and which verification step (if any) corresponds to it.

    A low alignment_score on a step that CoTShield also flagged is a
    strong signal of HIDDEN_REASONING at that specific point.
    """
    step_index: int
    step_text: str
    has_formal_backing: bool            # True if a proof/pattern step aligns here
    formal_claim: Optional[str]         # Extracted formalizable claim, or None
    cot_flags: List[Any]                # DivergenceFlags whose line_number == this step
    corresponding_proof_step: Optional[str]  # Best-aligned verification step
    alignment_score: float              # Jaccard overlap: 0.0 = no match, 1.0 = perfect


# ---------------------------------------------------------------------------
# Layer outputs
# ---------------------------------------------------------------------------

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
    """
    Output from the verification layer (FormalProofVerifier or SemanticPatternVerifier).

    Now carries the full VerificationResult trace so the combinator and summary
    can use proof depth, confidence, and per-claim breakdowns.
    """
    attempted: bool
    result: Optional[str] = None            # "success" | "failure" | "timeout"
    proof_steps: List[str] = field(default_factory=list)
    conjecture_str: Optional[str] = None
    error: Optional[str] = None
    proof_depth: int = 0                    # Deepest proof depth reached (0 if N/A)
    confidence: float = 0.0                 # Verifier confidence in its result
    claim_results: List[Any] = field(default_factory=list)  # ClaimVerificationResult list

    @property
    def succeeded(self) -> bool:
        return self.attempted and self.result == "success"

    @property
    def failed(self) -> bool:
        return self.attempted and self.result in ("failure", "timeout")

    @classmethod
    def from_verification_result(
        cls,
        vr,                             # VerificationResult
        conjecture_str: Optional[str] = None,
    ) -> "ProverLayer":
        """Build a ProverLayer directly from a VerificationResult."""
        return cls(
            attempted=True,
            result=vr.result_str,
            proof_steps=vr.proof_steps,
            conjecture_str=conjecture_str or vr.main_claim,
            error=vr.error,
            proof_depth=vr.proof_depth,
            confidence=vr.confidence,
            claim_results=vr.claim_results,
        )


# ---------------------------------------------------------------------------
# Combined verdict
# ---------------------------------------------------------------------------

@dataclass
class AuditVerdict:
    """
    Full dual-layer audit verdict for a single (reasoning, output) pair.

    The primary output of DualLayerAuditor.audit().

    Deep-integration fields:
      step_alignments   — per-step cross-examination records
      proof_coverage    — fraction of proof/verification steps that appear
                          in the model's CoT (did the model show its work?)
      reasoning_coverage — fraction of CoT steps with formal backing
                          (what fraction of what the model *claimed* is verified?)
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

    # Step-level alignment (empty list when prover not attempted)
    step_alignments: List[StepAlignment] = field(default_factory=list)
    proof_coverage: float = 0.0      # ↑ good: model explains actual proof steps
    reasoning_coverage: float = 0.0  # ↑ good: model's claims are formally backed

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
            lines.append(
                f"Proof      : {self.prover.result}  "
                f"({len(self.prover.proof_steps)} steps, depth={self.prover.proof_depth}, "
                f"confidence={self.prover.confidence:.2f})"
            )
            lines.append(
                f"Coverage   : proof→CoT {self.proof_coverage:.0%}  |  "
                f"CoT→proof {self.reasoning_coverage:.0%}"
            )
        else:
            lines.append(
                f"Proof      : not attempted — {self.prover.error or 'no claim'}"
            )
        lines.append(f"Explanation: {self.explanation}")
        if self.step_alignments:
            weak = [
                s for s in self.step_alignments
                if not s.has_formal_backing and s.cot_flags
            ]
            if weak:
                lines.append(
                    f"Weak steps : {len(weak)} CoT step(s) are both flagged and "
                    "have no formal backing"
                )
        return "\n".join(lines)
