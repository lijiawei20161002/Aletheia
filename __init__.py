"""
Aletheia — Dual-Layer Reasoning Auditor

Deeply integrates AutoConjecture (formal verification) with CoTShield
(CoT monitoring) to detect the "hidden reasoning" failure mode: AI models
reaching correct conclusions via deceptive, unsupported, or structurally
misaligned reasoning chains.

Both use cases — mathematical reasoning audits and propaganda analysis —
go through the same DualLayerAuditor with domain-appropriate Verifier
implementations.  No domain-specific hacks in the pipeline layer.

Quick start (math domain):
    from dual_layer import make_auditor

    auditor = make_auditor()
    verdict = auditor.audit(
        reasoning="x + 0 = 0 because addition is symmetric",
        output="Therefore: forall x. x + 0 = x",
    )
    print(verdict.summary())

Quick start (propaganda domain):
    from pipeline import PropagandaAuditPipeline

    pipeline = PropagandaAuditPipeline()
    verdict = pipeline.audit_analysis(article_text, aletheia_response)
    print(verdict.summary())
"""

from dual_layer import DualLayerAuditor, make_auditor
from verdict import TrustLabel, AuditVerdict, CoTLayer, ProverLayer, StepAlignment
from verifier import Verifier, FormalProofVerifier, SemanticPatternVerifier, VerificationResult
from pipeline import MathReasoningPipeline, PropagandaAuditPipeline, PipelineStats
from bridge import (
    nl_to_expression,
    proof_to_cot,
    extract_formal_claims,
    extract_step_claims,
    align_proof_to_cot,
)
from semantic_verifier import SemanticTechniqueVerifier
from adversary import AdversarialCritique
from campaign import CampaignDetector, CampaignResult, CampaignMatch

__version__ = "0.3.0"
__all__ = [
    # Core auditor
    "DualLayerAuditor",
    "make_auditor",
    # Verifier abstraction (new)
    "Verifier",
    "FormalProofVerifier",
    "SemanticPatternVerifier",
    "VerificationResult",
    # Verdict types
    "TrustLabel",
    "AuditVerdict",
    "CoTLayer",
    "ProverLayer",
    "StepAlignment",          # new
    # Pipelines
    "MathReasoningPipeline",
    "PropagandaAuditPipeline",
    "PipelineStats",
    # Bridge
    "nl_to_expression",
    "proof_to_cot",
    "extract_formal_claims",
    "extract_step_claims",    # new
    "align_proof_to_cot",     # new
    # Other modules
    "SemanticTechniqueVerifier",
    "AdversarialCritique",
    "CampaignDetector",
    "CampaignResult",
    "CampaignMatch",
]
