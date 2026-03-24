"""Logic system for formal reasoning with Peano arithmetic."""

from .terms import Term, Var, Zero, Succ, Add, Mul
from .expressions import Expression, Equation, Forall, Exists, Implies, And, Or, Not
from .axioms import PEANO_AXIOMS

__all__ = [
    "Term", "Var", "Zero", "Succ", "Add", "Mul",
    "Expression", "Equation", "Forall", "Exists", "Implies", "And", "Or", "Not",
    "PEANO_AXIOMS",
]
