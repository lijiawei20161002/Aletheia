"""
Logical expressions for first-order logic over Peano arithmetic.

Expressions include:
- Equations: t1 = t2
- Quantifiers: ∀x.φ, ∃x.φ
- Logical connectives: φ ∧ ψ, φ ∨ ψ, ¬φ, φ → ψ
"""

from __future__ import annotations
from typing import Set, Any
from abc import ABC, abstractmethod
from .terms import Term, Var


class Expression(ABC):
    """Abstract base class for logical expressions."""

    @abstractmethod
    def __str__(self) -> str:
        """String representation."""
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        pass

    @abstractmethod
    def free_vars(self) -> Set[str]:
        """Return the set of free variables."""
        pass

    @abstractmethod
    def substitute(self, var: str, replacement: Term) -> Expression:
        """Substitute term for variable (only if not bound)."""
        pass

    @abstractmethod
    def depth(self) -> int:
        """Return the depth of the expression tree."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the number of nodes in the expression tree."""
        pass

    @abstractmethod
    def complexity(self) -> int:
        """Return a complexity score (depth + size + quantifier count)."""
        pass

    def __repr__(self) -> str:
        return str(self)


class Equation(Expression):
    """Equation: left = right."""

    def __init__(self, left: Term, right: Term):
        if not isinstance(left, Term) or not isinstance(right, Term):
            raise TypeError("Equation sides must be Terms")
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left} = {self.right})"

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Equation) and
                self.left == other.left and
                self.right == other.right)

    def __hash__(self) -> int:
        return hash(("Equation", self.left, self.right))

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute(self, var: str, replacement: Term) -> Expression:
        return Equation(
            self.left.substitute(var, replacement),
            self.right.substitute(var, replacement)
        )

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def complexity(self) -> int:
        return self.depth() + self.size()


class Forall(Expression):
    """Universal quantification: ∀var.body."""

    def __init__(self, var: Var, body: Expression):
        if not isinstance(var, Var):
            raise TypeError("Forall variable must be a Var")
        if not isinstance(body, Expression):
            raise TypeError("Forall body must be an Expression")
        self.var = var
        self.body = body

    def __str__(self) -> str:
        return f"∀{self.var.name}.{self.body}"

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Forall) and
                self.var == other.var and
                self.body == other.body)

    def __hash__(self) -> int:
        return hash(("Forall", self.var, self.body))

    def free_vars(self) -> Set[str]:
        return self.body.free_vars() - {self.var.name}

    def substitute(self, var: str, replacement: Term) -> Expression:
        # Don't substitute if variable is bound by this quantifier
        if var == self.var.name:
            return self
        # TODO: Handle variable capture (alpha-conversion)
        if self.var.name in replacement.free_vars():
            # For now, just don't substitute to avoid capture
            return self
        return Forall(self.var, self.body.substitute(var, replacement))

    def depth(self) -> int:
        return 1 + self.body.depth()

    def size(self) -> int:
        return 1 + self.body.size()

    def complexity(self) -> int:
        return self.depth() + self.size() + 3  # +3 penalty for quantifier


class Exists(Expression):
    """Existential quantification: ∃var.body."""

    def __init__(self, var: Var, body: Expression):
        if not isinstance(var, Var):
            raise TypeError("Exists variable must be a Var")
        if not isinstance(body, Expression):
            raise TypeError("Exists body must be an Expression")
        self.var = var
        self.body = body

    def __str__(self) -> str:
        return f"∃{self.var.name}.{self.body}"

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Exists) and
                self.var == other.var and
                self.body == other.body)

    def __hash__(self) -> int:
        return hash(("Exists", self.var, self.body))

    def free_vars(self) -> Set[str]:
        return self.body.free_vars() - {self.var.name}

    def substitute(self, var: str, replacement: Term) -> Expression:
        # Don't substitute if variable is bound by this quantifier
        if var == self.var.name:
            return self
        # TODO: Handle variable capture (alpha-conversion)
        if self.var.name in replacement.free_vars():
            return self
        return Exists(self.var, self.body.substitute(var, replacement))

    def depth(self) -> int:
        return 1 + self.body.depth()

    def size(self) -> int:
        return 1 + self.body.size()

    def complexity(self) -> int:
        return self.depth() + self.size() + 3  # +3 penalty for quantifier


class And(Expression):
    """Conjunction: left ∧ right."""

    def __init__(self, left: Expression, right: Expression):
        if not isinstance(left, Expression) or not isinstance(right, Expression):
            raise TypeError("And arguments must be Expressions")
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, And) and
                self.left == other.left and
                self.right == other.right)

    def __hash__(self) -> int:
        return hash(("And", self.left, self.right))

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute(self, var: str, replacement: Term) -> Expression:
        return And(
            self.left.substitute(var, replacement),
            self.right.substitute(var, replacement)
        )

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def complexity(self) -> int:
        return self.depth() + self.size()


class Or(Expression):
    """Disjunction: left ∨ right."""

    def __init__(self, left: Expression, right: Expression):
        if not isinstance(left, Expression) or not isinstance(right, Expression):
            raise TypeError("Or arguments must be Expressions")
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Or) and
                self.left == other.left and
                self.right == other.right)

    def __hash__(self) -> int:
        return hash(("Or", self.left, self.right))

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute(self, var: str, replacement: Term) -> Expression:
        return Or(
            self.left.substitute(var, replacement),
            self.right.substitute(var, replacement)
        )

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def complexity(self) -> int:
        return self.depth() + self.size()


class Not(Expression):
    """Negation: ¬body."""

    def __init__(self, body: Expression):
        if not isinstance(body, Expression):
            raise TypeError("Not argument must be an Expression")
        self.body = body

    def __str__(self) -> str:
        return f"¬{self.body}"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Not) and self.body == other.body

    def __hash__(self) -> int:
        return hash(("Not", self.body))

    def free_vars(self) -> Set[str]:
        return self.body.free_vars()

    def substitute(self, var: str, replacement: Term) -> Expression:
        return Not(self.body.substitute(var, replacement))

    def depth(self) -> int:
        return 1 + self.body.depth()

    def size(self) -> int:
        return 1 + self.body.size()

    def complexity(self) -> int:
        return self.depth() + self.size()


class Implies(Expression):
    """Implication: hypothesis → conclusion."""

    def __init__(self, hypothesis: Expression, conclusion: Expression):
        if not isinstance(hypothesis, Expression) or not isinstance(conclusion, Expression):
            raise TypeError("Implies arguments must be Expressions")
        self.hypothesis = hypothesis
        self.conclusion = conclusion

    def __str__(self) -> str:
        return f"({self.hypothesis} → {self.conclusion})"

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Implies) and
                self.hypothesis == other.hypothesis and
                self.conclusion == other.conclusion)

    def __hash__(self) -> int:
        return hash(("Implies", self.hypothesis, self.conclusion))

    def free_vars(self) -> Set[str]:
        return self.hypothesis.free_vars() | self.conclusion.free_vars()

    def substitute(self, var: str, replacement: Term) -> Expression:
        return Implies(
            self.hypothesis.substitute(var, replacement),
            self.conclusion.substitute(var, replacement)
        )

    def depth(self) -> int:
        return 1 + max(self.hypothesis.depth(), self.conclusion.depth())

    def size(self) -> int:
        return 1 + self.hypothesis.size() + self.conclusion.size()

    def complexity(self) -> int:
        return self.depth() + self.size()


# Convenience functions
def eq(left: Term, right: Term) -> Equation:
    """Create an equation."""
    return Equation(left, right)


def forall(var: str, body: Expression) -> Forall:
    """Create universal quantification."""
    from .terms import Var
    return Forall(Var(var), body)


def exists(var: str, body: Expression) -> Exists:
    """Create existential quantification."""
    from .terms import Var
    return Exists(Var(var), body)


def and_(left: Expression, right: Expression) -> And:
    """Create conjunction."""
    return And(left, right)


def or_(left: Expression, right: Expression) -> Or:
    """Create disjunction."""
    return Or(left, right)


def not_(body: Expression) -> Not:
    """Create negation."""
    return Not(body)


def implies(hypothesis: Expression, conclusion: Expression) -> Implies:
    """Create implication."""
    return Implies(hypothesis, conclusion)


# Example usage
if __name__ == "__main__":
    from .terms import var, zero, succ, add, mul, nat

    x = var("x")
    y = var("y")

    # ∀x. (x + 0 = x)
    expr1 = forall("x", eq(add(x, zero()), x))
    print(f"Identity axiom: {expr1}")
    print(f"Complexity: {expr1.complexity()}")

    # ∀x. ∀y. (x + S(y) = S(x + y))
    expr2 = forall("x", forall("y", eq(add(x, succ(y)), succ(add(x, y)))))
    print(f"Addition axiom: {expr2}")
    print(f"Complexity: {expr2.complexity()}")

    # (2 + 3 = 5)
    expr3 = eq(add(nat(2), nat(3)), nat(5))
    print(f"Simple equation: {expr3}")
    print(f"Complexity: {expr3.complexity()}")
