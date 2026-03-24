"""
Aletheia — Dual-Layer Reasoning Auditor

Combines AutoConjecture (formal verification) with CoTShield (CoT monitoring)
to detect the "hidden reasoning" failure mode: AI models reaching correct
conclusions via deceptive or unsupported reasoning chains.

Quick start:
    from dual_layer import make_auditor

    auditor = make_auditor()
    verdict = auditor.audit(
        reasoning="x + 0 = 0 because addition is symmetric",
        output="Therefore: forall x. x + 0 = x",
        conjecture=None,  # auto-extracted from text
    )
    print(verdict.summary())
"""

from dual_layer import DualLayerAuditor, make_auditor
from verdict import TrustLabel, AuditVerdict, CoTLayer, ProverLayer
from pipeline import MathReasoningPipeline, PropagandaAuditPipeline, PipelineStats
from bridge import nl_to_expression, proof_to_cot, extract_formal_claims
from semantic_verifier import SemanticTechniqueVerifier
from adversary import AdversarialCritique
from campaign import CampaignDetector, CampaignResult, CampaignMatch

__version__ = "0.2.0"
__all__ = [
    "DualLayerAuditor",
    "make_auditor",
    "TrustLabel",
    "AuditVerdict",
    "CoTLayer",
    "ProverLayer",
    "MathReasoningPipeline",
    "PropagandaAuditPipeline",
    "PipelineStats",
    "nl_to_expression",
    "proof_to_cot",
    "extract_formal_claims",
    "SemanticTechniqueVerifier",
    "AdversarialCritique",
    "CampaignDetector",
    "CampaignResult",
    "CampaignMatch",
]
