"""Theorem proving system with tactics and proof search."""
from .tactics import Tactic, RewriteTactic, SubstituteTactic, SimplifyTactic
from .proof_engine import ProofEngine, ProofState, ProofResult

__all__ = [
    "Tactic", "RewriteTactic", "SubstituteTactic", "SimplifyTactic",
    "ProofEngine", "ProofState", "ProofResult",
]
