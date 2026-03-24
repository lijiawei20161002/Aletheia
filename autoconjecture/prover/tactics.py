"""
Tactics for theorem proving.
A tactic transforms a proof state by applying logical rules.

Basic tactics:
- Rewrite: Replace subterm using an axiom or proven theorem
- Substitute: Substitute a variable with a term
- Simplify: Apply basic simplification rules
- Induction: Apply induction principle
- Split: Split conjunctions/disjunctions
"""
from __future__ import annotations
from typing import List, Optional, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..logic.terms import Term, Var, Zero, Succ, Add, Mul
from ..logic.expressions import Expression, Equation, Forall, And, Or, Implies, Not


@dataclass
class ProofState:
    """
    Current state in a proof.
    Contains the goal to prove and available hypotheses.
    """
    goal: Expression  # What we're trying to prove
    hypotheses: List[Expression]  # Known facts we can use
    depth: int = 0  # Proof depth (for limiting search)

    def __str__(self) -> str:
        hyps = "\n  ".join(str(h) for h in self.hypotheses) if self.hypotheses else "(none)"
        return f"Goal: {self.goal}\nHypotheses:\n  {hyps}"


class Tactic(ABC):
    """
    Base class for tactics.
    A tactic transforms one proof state into zero or more new proof states.
    """

    @abstractmethod
    def apply(self, state: ProofState, knowledge_base: List[Expression]) -> List[ProofState]:
        """
        Apply the tactic to a proof state.
        Returns list of new proof states (empty list means proof complete).
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return human-readable tactic name."""
        pass


class RewriteTactic(Tactic):
    """
    Rewrite using an equation from hypotheses or knowledge base.
    If goal is "A = B" and we have "A = C" in hypotheses,
    we can rewrite to prove "C = B".
    """

    def __init__(self, equation_index: Optional[int] = None):
        """
        Args:
            equation_index: Which equation to use (from hypotheses/KB).
                           If None, try all equations.
        """
        self.equation_index = equation_index

    def apply(self, state: ProofState, knowledge_base: List[Expression]) -> List[ProofState]:
        """Apply rewrite tactic."""
        if not isinstance(state.goal, Equation):
            return [state]  # Can't rewrite non-equations

        # Collect available equations from hypotheses and knowledge base
        available_eqs = []
        for hyp in state.hypotheses:
            if isinstance(hyp, Equation):
                available_eqs.append(hyp)
            elif isinstance(hyp, Forall) and isinstance(hyp.body, Equation):
                # Instantiate universal quantifier (simplified)
                available_eqs.append(hyp.body)
        for stmt in knowledge_base:
            if isinstance(stmt, Equation):
                available_eqs.append(stmt)
            elif isinstance(stmt, Forall) and isinstance(stmt.body, Equation):
                available_eqs.append(stmt.body)

        # Try to match goal with available equations
        new_states = []
        goal_left, goal_right = state.goal.left, state.goal.right

        for eq in available_eqs:
            # Try rewriting left side of goal
            if self._terms_match(goal_left, eq.left):
                new_goal = Equation(eq.right, goal_right)
                new_states.append(ProofState(
                    goal=new_goal,
                    hypotheses=state.hypotheses,
                    depth=state.depth + 1
                ))

            # Try rewriting right side of goal
            if self._terms_match(goal_right, eq.left):
                new_goal = Equation(goal_left, eq.right)
                new_states.append(ProofState(
                    goal=new_goal,
                    hypotheses=state.hypotheses,
                    depth=state.depth + 1
                ))

            # Also try symmetric rewrites (A=B implies B=A)
            if self._terms_match(goal_left, eq.right):
                new_goal = Equation(eq.left, goal_right)
                new_states.append(ProofState(
                    goal=new_goal,
                    hypotheses=state.hypotheses,
                    depth=state.depth + 1
                ))

            if self._terms_match(goal_right, eq.right):
                new_goal = Equation(goal_left, eq.left)
                new_states.append(ProofState(
                    goal=new_goal,
                    hypotheses=state.hypotheses,
                    depth=state.depth + 1
                ))

        return new_states if new_states else [state]

    def _terms_match(self, t1: Term, t2: Term) -> bool:
        """Check if two terms are syntactically equal (simplified)."""
        return str(t1) == str(t2)

    def name(self) -> str:
        return "rewrite"


class SubstituteTactic(Tactic):
    """
    Substitute a variable with a term in the goal.
    Useful for instantiating universal quantifiers.
    """

    def __init__(self, var: Var, term: Term):
        self.var = var
        self.term = term

    def apply(self, state: ProofState, knowledge_base: List[Expression]) -> List[ProofState]:
        """Apply substitution."""
        # If goal is forall x. P(x), instantiate with specific term
        if isinstance(state.goal, Forall):
            new_body = self._substitute_expr(state.goal.body, state.goal.var, self.term)
            new_state = ProofState(
                goal=new_body,
                hypotheses=state.hypotheses,
                depth=state.depth + 1
            )
            return [new_state]

        return [state]

    def _substitute_expr(self, expr: Expression, var: Var, term: Term) -> Expression:
        """Substitute variable in expression."""
        if isinstance(expr, Equation):
            return Equation(
                expr.left.substitute(var, term),
                expr.right.substitute(var, term)
            )
        # Add more cases as needed
        return expr

    def name(self) -> str:
        return f"substitute({self.var.name} := {self.term})"


class SimplifyTactic(Tactic):
    """
    Apply basic simplification rules:
    - 0 + x = x
    - x + 0 = x
    - 0 * x = 0
    - 1 * x = x
    - S(x) = x + 1
    """

    def apply(self, state: ProofState, knowledge_base: List[Expression]) -> List[ProofState]:
        """Apply simplification."""
        if not isinstance(state.goal, Equation):
            return [state]

        left_simp = self._simplify_term(state.goal.left)
        right_simp = self._simplify_term(state.goal.right)

        # Check if goal is now trivially true
        if self._terms_equal(left_simp, right_simp):
            return []  # Proof complete!

        # Return simplified goal
        if left_simp != state.goal.left or right_simp != state.goal.right:
            new_state = ProofState(
                goal=Equation(left_simp, right_simp),
                hypotheses=state.hypotheses,
                depth=state.depth + 1
            )
            return [new_state]

        return [state]

    def _simplify_term(self, term: Term) -> Term:
        """Recursively simplify a term."""
        if isinstance(term, Add):
            left = self._simplify_term(term.left)
            right = self._simplify_term(term.right)

            # 0 + x = x
            if isinstance(left, Zero):
                return right
            # x + 0 = x
            if isinstance(right, Zero):
                return left

            return Add(left, right)

        elif isinstance(term, Mul):
            left = self._simplify_term(term.left)
            right = self._simplify_term(term.right)

            # 0 * x = 0
            if isinstance(left, Zero) or isinstance(right, Zero):
                return Zero()

            return Mul(left, right)

        elif isinstance(term, Succ):
            inner_term = self._simplify_term(term.term)
            return Succ(inner_term)

        return term

    def _terms_equal(self, t1: Term, t2: Term) -> bool:
        """Check if two terms are equal."""
        return str(t1) == str(t2)

    def name(self) -> str:
        return "simplify"


class ReflexivityTactic(Tactic):
    """
    Proves equations of the form t = t (reflexivity).
    """

    def apply(self, state: ProofState, knowledge_base: List[Expression]) -> List[ProofState]:
        """Check if goal is t = t."""
        if isinstance(state.goal, Equation):
            if str(state.goal.left) == str(state.goal.right):
                return []  # Proof complete!
        return [state]

    def name(self) -> str:
        return "reflexivity"


class AssumptionTactic(Tactic):
    """
    Proves goal if it matches one of the hypotheses.
    """

    def apply(self, state: ProofState, knowledge_base: List[Expression]) -> List[ProofState]:
        """Check if goal is in hypotheses."""
        goal_str = str(state.goal)
        for hyp in state.hypotheses:
            if str(hyp) == goal_str:
                return []  # Proof complete!
        return [state]

    def name(self) -> str:
        return "assumption"


# Default tactics available
DEFAULT_TACTICS = [
    ReflexivityTactic(),
    AssumptionTactic(),
    SimplifyTactic(),
    RewriteTactic(),
]
