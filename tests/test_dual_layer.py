"""
Unit tests for DualLayerAuditor and verdict logic.

Run:
    cd /home/ubuntu/Aletheia
    python -m pytest tests/ -v
"""

import sys
import os
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from verdict import TrustLabel, CoTLayer, ProverLayer, StepAlignment, AuditVerdict
from dual_layer import DualLayerAuditor, make_auditor
from cotshield.monitor.detector import DivergenceType, DivergenceFlag, ProofAwareCoTDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def auditor_no_prover():
    return make_auditor(with_prover=False, cot_sensitivity=0.6, suspicious_threshold=0.4)


@pytest.fixture
def auditor_with_prover():
    try:
        return make_auditor(with_prover=True, cot_sensitivity=0.6, suspicious_threshold=0.4)
    except Exception:
        pytest.skip("AutoConjecture prover not available")


# ---------------------------------------------------------------------------
# Combinator logic (unit tests — no prover needed)
# ---------------------------------------------------------------------------

class TestCombinator:
    """Test _combine() directly with mock CoT/prover layers."""

    def _make_cot(self, risk: float, suspicious: bool) -> CoTLayer:
        return CoTLayer(risk_score=risk, flag_count=int(risk * 5), flags=[], is_suspicious=suspicious)

    def _make_prover(self, succeeded: bool, attempted: bool = True) -> ProverLayer:
        return ProverLayer(
            attempted=attempted,
            result="success" if succeeded else "failure",
            proof_steps=["step1", "step2"] if succeeded else [],
        )

    def test_trustworthy_when_clean_reasoning_and_proof_succeeds(self, auditor_no_prover):
        cot = self._make_cot(0.1, False)
        prover = self._make_prover(True)
        label, score, explanation = auditor_no_prover._combine(cot, prover)
        assert label == TrustLabel.TRUSTWORTHY
        assert score > 0.8
        assert "trustworthy" in explanation.lower()

    def test_hidden_reasoning_when_suspicious_cot_but_proof_succeeds(self, auditor_no_prover):
        cot = self._make_cot(0.8, True)
        prover = self._make_prover(True)
        label, score, explanation = auditor_no_prover._combine(cot, prover)
        assert label == TrustLabel.HIDDEN_REASONING
        assert score < 0.5
        assert "hidden" in explanation.lower() or "deceptive" in explanation.lower()

    def test_honest_failure_when_clean_reasoning_and_proof_fails(self, auditor_no_prover):
        cot = self._make_cot(0.2, False)
        prover = self._make_prover(False)
        label, score, explanation = auditor_no_prover._combine(cot, prover)
        assert label == TrustLabel.HONEST_FAILURE
        assert 0.2 < score < 0.6

    def test_unreliable_when_both_fail(self, auditor_no_prover):
        cot = self._make_cot(0.9, True)
        prover = self._make_prover(False)
        label, score, explanation = auditor_no_prover._combine(cot, prover)
        assert label == TrustLabel.UNRELIABLE
        assert score < 0.2

    def test_unverifiable_when_not_attempted(self, auditor_no_prover):
        cot = self._make_cot(0.1, False)
        prover = ProverLayer(attempted=False, error="no conjecture")
        label, score, _ = auditor_no_prover._combine(cot, prover)
        assert label == TrustLabel.UNVERIFIABLE

    def test_unverifiable_suspicious_still_lowers_score(self, auditor_no_prover):
        cot_clean = self._make_cot(0.1, False)
        cot_susp  = self._make_cot(0.9, True)
        prover = ProverLayer(attempted=False, error="no conjecture")
        _, score_clean, _ = auditor_no_prover._combine(cot_clean, prover)
        _, score_susp, _  = auditor_no_prover._combine(cot_susp,  prover)
        assert score_clean > score_susp


# ---------------------------------------------------------------------------
# Full audit() with CoTShield-only (no prover)
# ---------------------------------------------------------------------------

class TestAuditNoProver:

    CLEAN_REASONING = (
        "By the definition of Peano addition, x + 0 is the base case and equals x. "
        "Therefore x + 0 = x for all x."
    )
    DECEPTIVE_REASONING = (
        "Clearly and obviously this holds without any doubt. "
        "It definitely works because of symmetry. "
        "This is certainly true in all cases immediately."
    )

    def test_clean_reasoning_gives_unverifiable(self, auditor_no_prover):
        v = auditor_no_prover.audit(
            reasoning=self.CLEAN_REASONING,
            output="Proved: forall x. x + 0 = x",
        )
        assert v.label == TrustLabel.UNVERIFIABLE
        assert not v.prover.attempted

    def test_deceptive_reasoning_still_unverifiable_no_prover(self, auditor_no_prover):
        v = auditor_no_prover.audit(
            reasoning=self.DECEPTIVE_REASONING,
            output="Proved: forall x. x + 0 = x",
        )
        # Without a prover, suspicious reasoning cannot yield HIDDEN_REASONING.
        # Note: CoTShield's heuristics may NOT flag purely over-confident reasoning
        # that lacks an explicit contradiction or uncertainty→confidence transition.
        # This is precisely why the prover layer is necessary — some forms of
        # deceptive reasoning are invisible to rule-based CoT monitoring alone.
        assert v.label == TrustLabel.UNVERIFIABLE
        assert not v.prover.attempted


# ---------------------------------------------------------------------------
# Bridge layer
# ---------------------------------------------------------------------------

class TestBridge:

    def test_extract_formal_claims_finds_equations(self):
        from bridge import extract_formal_claims
        text = "We claim that x + 0 = x for all x."
        claims = extract_formal_claims(text)
        assert len(claims) > 0
        # At least one should contain an equals sign
        assert any("=" in c for c in claims)

    def test_nl_to_expression_returns_none_on_gibberish(self):
        from bridge import nl_to_expression
        result = nl_to_expression("the cat sat on the mat and was very comfortable")
        assert result is None

    def test_proof_to_cot_empty_proof(self):
        from bridge import proof_to_cot
        reasoning, output = proof_to_cot(None)
        assert reasoning == ""
        assert "No proof" in output


# ---------------------------------------------------------------------------
# AuditVerdict dataclass
# ---------------------------------------------------------------------------

class TestAuditVerdict:

    def _make_verdict(self, label: TrustLabel) -> AuditVerdict:
        cot = CoTLayer(risk_score=0.5, flag_count=2, flags=[], is_suspicious=True)
        prover = ProverLayer(attempted=True, result="success", proof_steps=["step"])
        return AuditVerdict(
            reasoning="test",
            output="test",
            cot=cot,
            prover=prover,
            label=label,
            trust_score=0.5,
            explanation="test explanation",
        )

    def test_is_hidden_reasoning_true(self):
        v = self._make_verdict(TrustLabel.HIDDEN_REASONING)
        assert v.is_hidden_reasoning()

    def test_is_hidden_reasoning_false(self):
        v = self._make_verdict(TrustLabel.TRUSTWORTHY)
        assert not v.is_hidden_reasoning()

    def test_summary_contains_label(self):
        v = self._make_verdict(TrustLabel.HIDDEN_REASONING)
        s = v.summary()
        assert "HIDDEN_REASONING" in s
        assert "0.50" in s

    def test_step_alignments_default_empty(self):
        v = self._make_verdict(TrustLabel.TRUSTWORTHY)
        assert v.step_alignments == []
        assert v.proof_coverage == 0.0
        assert v.reasoning_coverage == 0.0


# ---------------------------------------------------------------------------
# ProofAwareCoTDetector — deep integration unit tests
# ---------------------------------------------------------------------------

class TestProofAwareDetector:

    def test_detects_phantom_step_when_cot_claims_induction_but_proof_does_not(self):
        detector = ProofAwareCoTDetector(sensitivity=0.6)
        cot = "We proceed by induction on n. Base case holds trivially."
        output = "Proved by induction."
        proof_steps = ["Apply rewrite to goal 'x + 0 = x' → QED"]
        flags = detector.detect_with_proof(cot, output, proof_steps)
        phantom_flags = [f for f in flags if f.type == DivergenceType.PHANTOM_STEP]
        assert len(phantom_flags) >= 1

    def test_detects_proof_mismatch_when_proof_uses_induction_but_cot_does_not(self):
        detector = ProofAwareCoTDetector(sensitivity=0.6)
        cot = "Since x + 0 = x follows from the definition, the result is obvious."
        output = "Proved: x + 0 = x."
        proof_steps = [
            "Apply induction to goal 'forall x. x + 0 = x'",
            "Base case: 0 + 0 = 0 → QED",
        ]
        flags = detector.detect_with_proof(cot, output, proof_steps)
        mismatch_flags = [f for f in flags if f.type == DivergenceType.PROOF_MISMATCH]
        assert len(mismatch_flags) >= 1

    def test_no_structural_flags_when_cot_and_proof_aligned(self):
        detector = ProofAwareCoTDetector(sensitivity=0.6)
        cot = "We apply induction. The base case holds. The inductive step follows."
        output = "Proved by induction."
        proof_steps = [
            "Apply induction to goal 'forall n. P(n)'",
            "Base case: P(0) → QED",
            "Inductive step: P(S(n)) → QED",
        ]
        flags = detector.detect_with_proof(cot, output, proof_steps)
        structural = [
            f for f in flags
            if f.type in (DivergenceType.PROOF_MISMATCH, DivergenceType.PHANTOM_STEP)
        ]
        assert len(structural) == 0

    def test_analyze_cot_trace_uses_proof_aware_detector_when_steps_provided(self):
        from cotshield.monitor.detector import analyze_cot_trace
        cot = "By induction on n, the base case and inductive step both hold."
        output = "Proved."
        proof_steps = ["Apply rewrite → QED"]
        report_with    = analyze_cot_trace(cot, output, proof_steps=proof_steps)
        report_without = analyze_cot_trace(cot, output)
        # With proof steps, PHANTOM_STEP should be detected; without, it won't be
        with_types    = set(report_with["divergence_types"].keys())
        assert "phantom_step"   in with_types
        assert "proof_mismatch" in with_types


# ---------------------------------------------------------------------------
# Bridge — deep integration helpers
# ---------------------------------------------------------------------------

class TestBridgeDeepIntegration:

    def test_extract_step_claims_splits_steps(self):
        from bridge import extract_step_claims
        reasoning = "Step 1: x + 0 = x by definition.\nStep 2: Therefore the claim holds."
        steps = extract_step_claims(reasoning)
        assert len(steps) >= 2
        for idx, text, claim in steps:
            assert isinstance(idx, int)
            assert isinstance(text, str)
            # claim is either str or None — both are valid

    def test_align_proof_to_cot_finds_matching_step(self):
        from bridge import align_proof_to_cot
        proof_steps = ["Apply induction to goal 'x + 0 = x'"]
        cot_steps   = ["We use induction on x.", "The base case is trivial."]
        alignments = align_proof_to_cot(proof_steps, cot_steps)
        assert len(alignments) == 1
        pi, ci, score = alignments[0]
        # "induction" appears in both → should align to cot_steps[0]
        assert ci == 0
        assert score > 0.0

    def test_align_proof_to_cot_empty_inputs(self):
        from bridge import align_proof_to_cot
        assert align_proof_to_cot([], []) == []
        assert align_proof_to_cot(["step"], []) == [(0, -1, 0.0)]
