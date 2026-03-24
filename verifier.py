"""
Abstract Verifier interface — the spine of Aletheia's dual-layer design.

Both AutoConjecture's ProofEngine (formal Peano arithmetic proofs) and
SemanticTechniqueVerifier (propaganda pattern matching) now implement this
single interface.  The DualLayerAuditor calls verify() without knowing which
domain it's operating in, eliminating the pipeline's override hack and
making both use cases first-class citizens.

Key design decision: every Verifier returns a VerificationResult that carries
not just a binary pass/fail but a structured trace of *what* was verified and
*how* — so CoTShield's proof-aware detector can cross-examine CoT steps against
actual verification steps instead of treating the prover as a black box.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ClaimVerificationResult:
    """Verification result for a single extracted claim or step."""
    claim_text: str
    verified: bool
    confidence: float        # 0.0 – 1.0
    evidence: List[str]      # Matched patterns, proof step strings, etc.
    error: Optional[str] = None


@dataclass
class VerificationResult:
    """
    Unified output from any Verifier implementation.

    Carries the full verification trace so CoTShield's proof-aware detector
    can compare it step-by-step against the model's stated reasoning — not
    just ask "did it succeed?"
    """
    overall_verified: bool
    confidence: float                                      # 0.0 – 1.0
    main_claim: Optional[str] = None                      # The claim verified
    claim_results: List[ClaimVerificationResult] = field(default_factory=list)
    proof_steps: List[str] = field(default_factory=list)  # Tactic/match steps as strings
    proof_depth: int = 0                                   # Search depth (formal) or 0
    error: Optional[str] = None

    @property
    def result_str(self) -> str:
        return "success" if self.overall_verified else "failure"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Verifier(ABC):
    """
    Abstract verification layer for Aletheia.

    Subclasses:
        FormalProofVerifier    — wraps AutoConjecture ProofEngine
        SemanticPatternVerifier — wraps SemanticTechniqueVerifier
    """

    @abstractmethod
    def verify(
        self,
        claim_text: str,
        context_text: str = "",
    ) -> VerificationResult:
        """
        Verify a claim against available evidence or axioms.

        Parameters
        ----------
        claim_text   : The main claim to verify.  For the math domain this
                       is a natural-language statement of the conjecture; for
                       the propaganda domain it is a comma-separated list of
                       claimed rhetorical technique names.
        context_text : Supporting context.  For propaganda this is the source
                       article text; for math this is the surrounding reasoning.

        Returns
        -------
        VerificationResult with full trace.
        """


# ---------------------------------------------------------------------------
# Formal proof verifier (AutoConjecture)
# ---------------------------------------------------------------------------

class FormalProofVerifier(Verifier):
    """
    Wraps AutoConjecture's ProofEngine as a Verifier.

    Parses claim_text into a formal Expression, runs the best-first proof
    search, and returns the full tactic trace so CoTShield can check whether
    the model's stated proof steps match the actual ones.
    """

    def __init__(self, proof_engine, knowledge_base: Optional[List] = None):
        self.proof_engine = proof_engine
        self.knowledge_base = knowledge_base or []

    def verify(self, claim_text: str, context_text: str = "") -> VerificationResult:
        from bridge import nl_to_expression

        # Try to parse the claim; fall back to full context if the isolated
        # claim string isn't parseable
        expr = nl_to_expression(claim_text) or nl_to_expression(context_text)
        if expr is None:
            return VerificationResult(
                overall_verified=False,
                confidence=0.0,
                error="Could not parse claim into a formal expression",
            )

        try:
            proof = self.proof_engine.prove(expr, hypotheses=self.knowledge_base)
            steps = [str(s) for s in proof.steps]

            # Extract maximum search depth from step metadata when available
            depth = len(steps)
            for s in proof.steps:
                if hasattr(s, "state_before") and s.state_before is not None:
                    if hasattr(s.state_before, "depth"):
                        depth = max(depth, s.state_before.depth)

            result_val = (
                proof.result.value
                if hasattr(proof.result, "value")
                else str(proof.result)
            )
            succeeded = result_val == "success"

            return VerificationResult(
                overall_verified=succeeded,
                confidence=0.9 if succeeded else 0.1,
                main_claim=str(expr),
                proof_steps=steps,
                proof_depth=depth,
                error=None if succeeded else f"Proof {result_val}",
            )
        except Exception as exc:
            return VerificationResult(
                overall_verified=False,
                confidence=0.0,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Semantic pattern verifier (propaganda domain)
# ---------------------------------------------------------------------------

class SemanticPatternVerifier(Verifier):
    """
    Wraps SemanticTechniqueVerifier as a Verifier.

    Verifies that claimed rhetorical techniques actually appear in source text
    using cosine similarity over sentence embeddings (regex fallback when the
    sentence-transformers library is absent).

    claim_text  : comma-separated list of technique names
    context_text: source article / document to verify against
    """

    def __init__(self, semantic_verifier=None):
        if semantic_verifier is None:
            from semantic_verifier import SemanticTechniqueVerifier as _STV
            semantic_verifier = _STV()
        self._verifier = semantic_verifier

    def verify(self, claim_text: str, context_text: str = "") -> VerificationResult:
        techniques = [t.strip() for t in claim_text.split(",") if t.strip()]
        if not techniques:
            return VerificationResult(
                overall_verified=False,
                confidence=0.0,
                error="No techniques specified in claim_text",
            )

        result = self._verifier.verify_techniques(context_text, techniques)
        verified = result.get("verified", False)
        matched = result.get("matched_patterns", [])
        scores = result.get("scores", {})
        avg_confidence = (
            sum(scores.values()) / len(scores)
            if scores
            else (0.75 if verified else 0.2)
        )

        claim_results = []
        for tech in techniques:
            tech_verified = any(tech.lower() in m.lower() for m in matched)
            score = scores.get(tech, 0.0)
            claim_results.append(
                ClaimVerificationResult(
                    claim_text=tech,
                    verified=tech_verified,
                    confidence=score,
                    evidence=[m for m in matched if tech.lower() in m.lower()],
                )
            )

        return VerificationResult(
            overall_verified=verified,
            confidence=avg_confidence,
            main_claim=claim_text,
            claim_results=claim_results,
            proof_steps=matched,   # Matched patterns serve as "proof steps" for alignment
            proof_depth=0,
            error=None,
        )
