"""
Term representation for Peano arithmetic.

Terms are the basic building blocks of our formal system:
- Variables: x, y, z, ...
- Zero: 0
- Successor: S(t) represents t + 1
- Addition: t1 + t2
- Multiplication: t1 * t2
"""

from __future__ import annotations
from typing import Set, Dict, Any
from abc import ABC, abstractmethod


class Term(ABC):
    """Abstract base class for all terms."""

    @abstractmethod
    def __str__(self) -> str:
        """String representation of the term."""
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
        """Return the set of free variables in this term."""
        pass

    @abstractmethod
    def substitute(self, var: str, replacement: Term) -> Term:
        """Substitute all occurrences of var with replacement."""
        pass

    @abstractmethod
    def depth(self) -> int:
        """Return the depth of the expression tree."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the number of nodes in the expression tree."""
        pass

    def __repr__(self) -> str:
        return str(self)


class Var(Term):
    """Variable term."""

    def __init__(self, name: str):
        if not name or not name.isidentifier():
            raise ValueError(f"Invalid variable name: {name}")
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Var) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("Var", self.name))

    def free_vars(self) -> Set[str]:
        return {self.name}

    def substitute(self, var: str, replacement: Term) -> Term:
        if self.name == var:
            return replacement
        return self

    def depth(self) -> int:
        return 1

    def size(self) -> int:
        return 1


class Zero(Term):
    """Zero constant (0)."""

    def __str__(self) -> str:
        return "0"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Zero)

    def __hash__(self) -> int:
        return hash("Zero")

    def free_vars(self) -> Set[str]:
        return set()

    def substitute(self, var: str, replacement: Term) -> Term:
        return self

    def depth(self) -> int:
        return 1

    def size(self) -> int:
        return 1


class Succ(Term):
    """Successor function S(t) = t + 1."""

    def __init__(self, term: Term):
        if not isinstance(term, Term):
            raise TypeError(f"Successor argument must be a Term, got {type(term)}")
        self.term = term

    def __str__(self) -> str:
        return f"S({self.term})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Succ) and self.term == other.term

    def __hash__(self) -> int:
        return hash(("Succ", self.term))

    def free_vars(self) -> Set[str]:
        return self.term.free_vars()

    def substitute(self, var: str, replacement: Term) -> Term:
        return Succ(self.term.substitute(var, replacement))

    def depth(self) -> int:
        return 1 + self.term.depth()

    def size(self) -> int:
        return 1 + self.term.size()


class Add(Term):
    """Addition: left + right."""

    def __init__(self, left: Term, right: Term):
        if not isinstance(left, Term) or not isinstance(right, Term):
            raise TypeError("Addition arguments must be Terms")
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Add) and
                self.left == other.left and
                self.right == other.right)

    def __hash__(self) -> int:
        return hash(("Add", self.left, self.right))

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute(self, var: str, replacement: Term) -> Term:
        return Add(
            self.left.substitute(var, replacement),
            self.right.substitute(var, replacement)
        )

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()


class Mul(Term):
    """Multiplication: left * right."""

    def __init__(self, left: Term, right: Term):
        if not isinstance(left, Term) or not isinstance(right, Term):
            raise TypeError("Multiplication arguments must be Terms")
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Mul) and
                self.left == other.left and
                self.right == other.right)

    def __hash__(self) -> int:
        return hash(("Mul", self.left, self.right))

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()

    def substitute(self, var: str, replacement: Term) -> Term:
        return Mul(
            self.left.substitute(var, replacement),
            self.right.substitute(var, replacement)
        )

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()


# Convenience functions for creating terms
def var(name: str) -> Var:
    """Create a variable."""
    return Var(name)


def zero() -> Zero:
    """Create zero."""
    return Zero()


def succ(term: Term) -> Succ:
    """Create successor."""
    return Succ(term)


def add(left: Term, right: Term) -> Add:
    """Create addition."""
    return Add(left, right)


def mul(left: Term, right: Term) -> Mul:
    """Create multiplication."""
    return Mul(left, right)


def nat(n: int) -> Term:
    """Create a natural number as nested successors of zero."""
    if n < 0:
        raise ValueError(f"Natural number must be non-negative, got {n}")
    term: Term = Zero()
    for _ in range(n):
        term = Succ(term)
    return term


# Example usage and tests
if __name__ == "__main__":
    # Create some terms
    x = var("x")
    y = var("y")
    two = nat(2)  # S(S(0))
    three = nat(3)  # S(S(S(0)))

    # Create compound terms
    expr1 = add(x, two)  # x + 2
    expr2 = mul(y, three)  # y * 3
    expr3 = add(expr1, expr2)  # (x + 2) + (y * 3)

    print(f"x = {x}")
    print(f"2 = {two}")
    print(f"x + 2 = {expr1}")
    print(f"y * 3 = {expr2}")
    print(f"(x + 2) + (y * 3) = {expr3}")
    print(f"Free vars in expr3: {expr3.free_vars()}")
    print(f"Depth of expr3: {expr3.depth()}")
    print(f"Size of expr3: {expr3.size()}")

    # Test substitution
    expr4 = expr3.substitute("x", nat(5))
    print(f"After substituting x=5: {expr4}")
