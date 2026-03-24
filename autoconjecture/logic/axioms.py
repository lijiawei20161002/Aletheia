"""
Peano axioms for natural numbers.

The Peano axioms define natural numbers using:
1. Zero is a natural number
2. Successor of any natural number is a natural number
3. Zero is not the successor of any natural number
4. Successor is injective (if S(x) = S(y), then x = y)
5. Induction principle
6. Addition axioms
7. Multiplication axioms
"""

from .terms import var, zero, succ, add, mul
from .expressions import eq, forall, not_, implies, and_

# Common variables
x = var("x")
y = var("y")
z = var("z")


# Peano axioms as a list of (name, expression) pairs
PEANO_AXIOMS = [
    # Axiom 1: Zero is not the successor of any number
    # ∀x. ¬(S(x) = 0)
    ("zero_not_succ",
     forall("x", not_(eq(succ(x), zero())))),

    # Axiom 2: Successor is injective
    # ∀x. ∀y. (S(x) = S(y) → x = y)
    ("succ_injective",
     forall("x", forall("y", implies(eq(succ(x), succ(y)), eq(x, y))))),

    # Addition axioms
    # Axiom 3: x + 0 = x
    # ∀x. (x + 0 = x)
    ("add_zero",
     forall("x", eq(add(x, zero()), x))),

    # Axiom 4: x + S(y) = S(x + y)
    # ∀x. ∀y. (x + S(y) = S(x + y))
    ("add_succ",
     forall("x", forall("y", eq(add(x, succ(y)), succ(add(x, y)))))),

    # Multiplication axioms
    # Axiom 5: x * 0 = 0
    # ∀x. (x * 0 = 0)
    ("mul_zero",
     forall("x", eq(mul(x, zero()), zero()))),

    # Axiom 6: x * S(y) = x * y + x
    # ∀x. ∀y. (x * S(y) = (x * y) + x)
    ("mul_succ",
     forall("x", forall("y", eq(mul(x, succ(y)), add(mul(x, y), x))))),

    # Note: Induction axiom is special - it's a schema, not a single statement
    # For any property P:
    # (P(0) ∧ (∀k. P(k) → P(S(k)))) → ∀n. P(n)
    # This will be handled specially in the prover
]


# Additional useful theorems that can be derived (for reference/testing)
DERIVED_THEOREMS = [
    # Commutativity
    ("add_comm", forall("x", forall("y", eq(add(x, y), add(y, x))))),
    ("mul_comm", forall("x", forall("y", eq(mul(x, y), mul(y, x))))),

    # Associativity
    ("add_assoc", forall("x", forall("y", forall("z",
        eq(add(add(x, y), z), add(x, add(y, z))))))),
    ("mul_assoc", forall("x", forall("y", forall("z",
        eq(mul(mul(x, y), z), mul(x, mul(y, z))))))),

    # Distributivity
    ("mul_dist_add", forall("x", forall("y", forall("z",
        eq(mul(x, add(y, z)), add(mul(x, y), mul(x, z))))))),

    # Identity
    ("mul_one", forall("x", eq(mul(x, succ(zero())), x))),

    # Zero absorption
    ("add_zero_right", forall("x", eq(add(zero(), x), x))),
    ("mul_zero_right", forall("x", eq(mul(zero(), x), zero()))),
]


def get_axiom_by_name(name: str):
    """Retrieve an axiom by name."""
    for ax_name, expr in PEANO_AXIOMS:
        if ax_name == name:
            return expr
    raise ValueError(f"Unknown axiom: {name}")


def get_all_axioms():
    """Get all Peano axioms as a list of expressions."""
    return [expr for _, expr in PEANO_AXIOMS]


def get_axiom_names():
    """Get all axiom names."""
    return [name for name, _ in PEANO_AXIOMS]


# Display axioms
if __name__ == "__main__":
    print("Peano Axioms:")
    print("=" * 60)
    for name, expr in PEANO_AXIOMS:
        print(f"{name:20} | {expr}")
        print(f"{'':20} | Complexity: {expr.complexity()}")
        print()

    print("\nDerived Theorems (for reference):")
    print("=" * 60)
    for name, expr in DERIVED_THEOREMS:
        print(f"{name:20} | {expr}")
        print(f"{'':20} | Complexity: {expr.complexity()}")
        print()
