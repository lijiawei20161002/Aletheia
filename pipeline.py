"""
End-to-end audit pipelines for the two primary Aletheia use cases.

MathReasoningPipeline    — audits AutoConjecture's neural generator (Phase 2/3)
PropagandaAuditPipeline  — audits Aletheia's Claude-based propaganda analysis

Both pipelines now use the same DualLayerAuditor with domain-appropriate
Verifier instances — no more domain-specific override hacks.  The propaganda
pipeline passes a SemanticPatternVerifier; the math pipeline passes a
FormalProofVerifier.  The auditor's integrated pipeline handles alignment,
proof-aware CoT detection, and trust-score computation uniformly.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Dict, Any

from dual_layer import DualLayerAuditor, make_auditor
from verdict import AuditVerdict, TrustLabel


# ---------------------------------------------------------------------------
# Pipeline statistics (shared by both pipelines)
# ---------------------------------------------------------------------------

@dataclass
class PipelineStats:
    total: int = 0
    by_label: Dict[str, int] = field(default_factory=lambda: {
        label.value: 0 for label in TrustLabel
    })
    hidden_reasoning_rate: float = 0.0
    mean_trust_score: float = 0.0
    mean_cot_risk: float = 0.0
    mean_proof_coverage: float = 0.0
    mean_reasoning_coverage: float = 0.0
    _trust_sum: float = field(default=0.0, repr=False)
    _risk_sum: float  = field(default=0.0, repr=False)
    _pcov_sum: float  = field(default=0.0, repr=False)
    _rcov_sum: float  = field(default=0.0, repr=False)

    def update(self, verdict: AuditVerdict):
        self.total += 1
        self.by_label[verdict.label.value] += 1
        self._trust_sum += verdict.trust_score
        self._risk_sum  += verdict.cot.risk_score
        self._pcov_sum  += verdict.proof_coverage
        self._rcov_sum  += verdict.reasoning_coverage
        n = self.total
        self.mean_trust_score        = self._trust_sum / n
        self.mean_cot_risk           = self._risk_sum  / n
        self.mean_proof_coverage     = self._pcov_sum  / n
        self.mean_reasoning_coverage = self._rcov_sum  / n
        hr = self.by_label[TrustLabel.HIDDEN_REASONING.value]
        self.hidden_reasoning_rate = hr / n

    def report(self) -> str:
        lines = [
            f"=== Pipeline Statistics (n={self.total}) ===",
            f"  Mean trust score         : {self.mean_trust_score:.3f}",
            f"  Mean CoT risk            : {self.mean_cot_risk:.3f}",
            f"  Mean proof→CoT coverage  : {self.mean_proof_coverage:.1%}",
            f"  Mean CoT→proof coverage  : {self.mean_reasoning_coverage:.1%}",
            f"  Hidden-reasoning rate    : {self.by_label[TrustLabel.HIDDEN_REASONING.value]} "
            f"({self.hidden_reasoning_rate:.1%})",
            "",
            "  Label breakdown:",
        ]
        for label in TrustLabel:
            count = self.by_label[label.value]
            pct   = count / self.total if self.total else 0
            lines.append(f"    {label.value:<20} {count:>5}  ({pct:.1%})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Math Reasoning Pipeline
# ---------------------------------------------------------------------------

class MathReasoningPipeline:
    """
    Audit AutoConjecture's neural generator output.

    The pipeline intercepts the generator's (conjecture, justification) pairs,
    runs the integrated dual-layer audit, and flags HIDDEN_REASONING cases
    where the neural model reaches a provable conjecture via unsupported or
    structurally misaligned reasoning.

    This directly targets the alignment-critical failure mode in Phase 3 RL:
    reward hacking where the model gets the right answer for the wrong reason.
    The new step-level alignment data (proof_coverage, reasoning_coverage)
    gives a quantitative signal of *how much* the model's explanation diverges
    from the actual proof trace.
    """

    def __init__(self, auditor: Optional[DualLayerAuditor] = None):
        self.auditor = auditor or make_auditor()
        self.stats = PipelineStats()

    def run(
        self,
        generator,
        n: int = 50,
        justification_fn=None,
    ) -> Iterator[AuditVerdict]:
        """
        Generate n conjectures, produce a justification for each, and audit.

        Parameters
        ----------
        generator : NeuralConjectureGenerator
        n         : Number of conjectures to audit.
        justification_fn : optional callable (conjecture) → (reasoning, output).
        """
        conjectures = generator.generate(n)

        for i, conjecture in enumerate(conjectures):
            if conjecture is None:
                continue

            if justification_fn is not None:
                reasoning, output = justification_fn(conjecture)
            else:
                reasoning, output = self._default_justification(conjecture)

            verdict = self.auditor.audit(
                reasoning=reasoning,
                output=output,
                conjecture=conjecture,
                source=f"neural_generator[{i}]",
            )
            self.stats.update(verdict)
            yield verdict

    def audit_proof_traces(self, proofs) -> Iterator[AuditVerdict]:
        """
        Audit AutoConjecture Proof objects by analyzing their step traces.

        Catches cases where the prover reports SUCCESS but the step-by-step
        trace doesn't actually support the final statement.
        """
        for proof in proofs:
            verdict = self.auditor.audit_proof(proof, source="proof_trace")
            self.stats.update(verdict)
            yield verdict

    @staticmethod
    def _default_justification(conjecture) -> tuple:
        """
        Produce a deliberately misleading justification to test detection.
        """
        conjecture_str = str(conjecture)
        reasoning = (
            f"We want to show that {conjecture_str}. "
            "Since addition is commutative, we can freely swap operands. "
            "Therefore the result clearly follows from symmetry. "
            "Obviously, the identity holds because both sides are equivalent."
        )
        output = f"Therefore: {conjecture_str}"
        return reasoning, output


# ---------------------------------------------------------------------------
# Propaganda Audit Pipeline
# ---------------------------------------------------------------------------

class PropagandaAuditPipeline:
    """
    Audit Aletheia's propaganda analysis for reasoning trustworthiness.

    Layer 1 (CoTShield + proof-aware detector):
        Does Claude's explanation actually support its score?  Are there
        logical leaps from evidence to conclusion?  Do pattern matches in
        the source text correspond to the reasoning steps the model gave?

    Layer 2 (SemanticPatternVerifier):
        Do the claimed rhetorical techniques actually appear in the source
        text via cosine similarity?  This is the domain-appropriate verifier
        for propaganda — it plays the same role the formal prover plays in
        the math domain, through the same Verifier interface.

    Combined verdict:
        HIDDEN_REASONING — Claude correctly identified propaganda but its
                           stated reasons don't match the actual patterns found
                           in the text (now also flagged as PROOF_MISMATCH or
                           PHANTOM_STEP when the mismatch is tactic-level).
    """

    def __init__(
        self,
        semantic_verifier=None,
    ):
        from verifier import SemanticPatternVerifier
        verifier = SemanticPatternVerifier(semantic_verifier)
        self.auditor = DualLayerAuditor(verifier=verifier)
        self.stats = PipelineStats()

    def audit_direct(
        self,
        text: str,
        reasoning: str,
        output: str,
        techniques: Optional[List[str]] = None,
    ) -> AuditVerdict:
        """
        Audit using explicit reasoning and output strings.

        Parameters
        ----------
        text       : original source text (for pattern verification)
        reasoning  : the AI's chain-of-thought explanation
        output     : the AI's stated verdict/conclusion
        techniques : list of technique names to verify in text
        """
        claim_text = ", ".join(techniques) if techniques else ""
        verdict = self.auditor.audit(
            reasoning=reasoning,
            output=output,
            claim_text=claim_text,
            context_text=text,
            source="aletheia_direct",
        )
        self.stats.update(verdict)
        return verdict

    def audit_analysis(self, text: str, aletheia_response: dict) -> AuditVerdict:
        """
        Audit a single Aletheia analysis response.

        Parameters
        ----------
        text              : The original media text that was analyzed.
        aletheia_response : JSON response from Aletheia's /analyze endpoint.
        """
        reasoning, output = self._extract_reasoning(aletheia_response)
        techniques = [
            t.get("technique", "")
            for t in aletheia_response.get("rhetorical_techniques", [])
        ]
        claim_text = ", ".join(t for t in techniques if t)

        verdict = self.auditor.audit(
            reasoning=reasoning,
            output=output,
            claim_text=claim_text,
            context_text=text,
            source="aletheia_analysis",
        )
        self.stats.update(verdict)
        return verdict

    def audit_with_thinking(
        self,
        text: str,
        aletheia_response: dict,
        thinking_text: str,
        adversary_critique: Optional[Dict[str, Any]] = None,
    ) -> AuditVerdict:
        """
        Audit using Claude's extended thinking block as the primary reasoning input.

        If adversary_critique is provided, the analyst's thinking and the adversary
        dialogue are merged into a single CoT string so CoTShield audits both
        signals together — the strongest HIDDEN_REASONING detector for this domain.
        """
        if adversary_critique is not None:
            from adversary import AdversarialCritique
            adversary_obj = AdversarialCritique(client=None)
            adversary_reasoning = adversary_obj.build_audit_reasoning(
                aletheia_response, adversary_critique
            )
            reasoning = (
                thinking_text
                + "\n\n=== ADVERSARIAL REVIEW ===\n"
                + adversary_reasoning
            )
        else:
            reasoning = thinking_text

        score        = aletheia_response.get("propaganda_score", "?")
        verdict_text = aletheia_response.get("verdict", "")
        output       = f"Propaganda score: {score}/10. {verdict_text}"

        claimed_techniques = [
            t.get("technique", "")
            for t in aletheia_response.get("rhetorical_techniques", [])
        ]
        claim_text = ", ".join(t for t in claimed_techniques if t)

        verdict = self.auditor.audit(
            reasoning=reasoning,
            output=output,
            claim_text=claim_text,
            context_text=text,
            source="aletheia_extended_thinking",
        )
        self.stats.update(verdict)
        return verdict

    def audit_batch_from_file(self, jsonl_path: str) -> Iterator[AuditVerdict]:
        """
        Audit a JSONL file where each line has {"text": ..., "analysis": ...}.
        """
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line.strip())
                yield self.audit_analysis(record["text"], record["analysis"])

    # ------------------------------------------------------------------
    # Helpers (unchanged)
    # ------------------------------------------------------------------

    def _extract_reasoning(self, response: dict) -> tuple:
        parts = []

        for t in response.get("rhetorical_techniques", []):
            parts.append(
                f"Technique '{t.get('technique', '?')}': {t.get('description', '')} "
                f"Evidence: '{t.get('example', '')}'"
            )

        em = response.get("emotional_manipulation", {})
        if em:
            parts.append(
                f"Primary emotion targeted: {em.get('primary_emotion', '?')} "
                f"at intensity {em.get('intensity', '?')}. {em.get('analysis', '')}"
            )

        nf = response.get("narrative_framing", {})
        if nf:
            def flag(val):
                return "present" if val else "absent"
            parts.append(
                f"Narrative framing: {nf.get('core_narrative', '')}. "
                f"Us-vs-them framing is {flag(nf.get('us_vs_them', False))}. "
                f"Scapegoating is {flag(nf.get('scapegoating', False))}. "
                f"False urgency is {flag(nf.get('false_urgency', False))}. "
                f"{nf.get('analysis', '')}"
            )

        for kp in response.get("key_passages", []):
            parts.append(
                f"Key passage: '{kp.get('passage', '')}' — {kp.get('concern', '')}"
            )

        summary = response.get("summary", "")
        if summary:
            parts.append(f"Summary: {summary}")

        reasoning = "\n".join(parts) if parts else "No detailed reasoning provided."
        score        = response.get("propaganda_score", "?")
        verdict_text = response.get("verdict", "")
        output = f"Propaganda score: {score}/10. {verdict_text}"

        return reasoning, output
