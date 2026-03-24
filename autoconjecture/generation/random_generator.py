"""
Random conjecture generator.
Generates valid logical expressions randomly for initial exploration.
"""
import random
from typing import List, Set
from ..logic.terms import Term, Var, Zero, Succ, Add, Mul
from ..logic.expressions import Expression, Equation, Forall


class RandomConjectureGenerator:
    """
    Generates random conjectures about Peano arithmetic.
    Uses controlled randomness to generate well-formed but interesting statements.
    """

    def __init__(
        self,
        min_complexity: int = 2,
        max_complexity: int = 10,
        var_names: List[str] = None,
        seed: int = None
    ):
        """
        Args:
            min_complexity: Minimum expression complexity
            max_complexity: Maximum expression complexity
            var_names: Variable names to use (default: x, y, z, w)
            seed: Random seed for reproducibility
        """
        self.min_complexity = min_complexity
        self.max_complexity = max_complexity
        self.var_names = var_names if var_names else ["x", "y", "z", "w"]

        if seed is not None:
            random.seed(seed)

    def generate(self, n: int = 1) -> List[Expression]:
        """
        Generate n random conjectures.

        Args:
            n: Number of conjectures to generate

        Returns:
            List of generated expressions
        """
        conjectures = []
        for _ in range(n):
            conjecture = self._generate_single()
            conjectures.append(conjecture)
        return conjectures

    def _generate_single(self) -> Expression:
        """Generate a single random conjecture."""
        # Randomly choose conjecture type
        conjecture_type = random.choice([
            "universal_equation",  # forall x. P(x)
            "equation",  # Simple equation
        ])

        if conjecture_type == "universal_equation":
            return self._generate_universal_equation()
        else:
            return self._generate_equation()

    def _generate_universal_equation(self) -> Forall:
        """Generate universally quantified equation: forall x. t1 = t2"""
        # Pick variable to quantify over
        var_name = random.choice(self.var_names)
        var = Var(var_name)

        # Generate two terms with this variable
        term1 = self._generate_term(max_depth=3, required_vars={var_name})
        term2 = self._generate_term(max_depth=3, required_vars={var_name})

        eq = Equation(term1, term2)
        return Forall(var, eq)

    def _generate_equation(self) -> Equation:
        """Generate simple equation: t1 = t2"""
        term1 = self._generate_term(max_depth=3)
        term2 = self._generate_term(max_depth=3)
        return Equation(term1, term2)

    def _generate_term(
        self,
        max_depth: int,
        required_vars: Set[str] = None,
        current_depth: int = 0
    ) -> Term:
        """
        Generate a random term with bounded depth.

        Args:
            max_depth: Maximum nesting depth
            required_vars: Variables that must appear in the term
            current_depth: Current recursion depth

        Returns:
            Random term
        """
        if required_vars is None:
            required_vars = set()

        # Base case: reached max depth or randomly decide to stop
        if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
            # Must use required vars if specified
            if required_vars:
                var_name = random.choice(list(required_vars))
                return Var(var_name)
            else:
                # Return zero or variable
                if random.random() < 0.5:
                    return Zero()
                else:
                    var_name = random.choice(self.var_names)
                    return Var(var_name)

        # Recursive case: choose term constructor
        constructors = [
            "var",
            "zero",
            "succ",
            "add",
            "mul",
        ]

        # Weight towards operations if we have required vars
        if required_vars:
            constructors.extend(["add", "mul"] * 2)  # Bias towards operations

        constructor = random.choice(constructors)

        if constructor == "zero":
            return Zero()

        elif constructor == "var":
            if required_vars:
                var_name = random.choice(list(required_vars))
            else:
                var_name = random.choice(self.var_names)
            return Var(var_name)

        elif constructor == "succ":
            inner = self._generate_term(max_depth, required_vars, current_depth + 1)
            return Succ(inner)

        elif constructor == "add":
            # Split required vars between left and right
            left_vars = set()
            right_vars = set()
            if required_vars:
                # Ensure at least one side has a required var
                if random.random() < 0.5:
                    left_vars = required_vars
                else:
                    right_vars = required_vars

            left = self._generate_term(max_depth, left_vars, current_depth + 1)
            right = self._generate_term(max_depth, right_vars, current_depth + 1)
            return Add(left, right)

        elif constructor == "mul":
            # Similar to add
            left_vars = set()
            right_vars = set()
            if required_vars:
                if random.random() < 0.5:
                    left_vars = required_vars
                else:
                    right_vars = required_vars

            left = self._generate_term(max_depth, left_vars, current_depth + 1)
            right = self._generate_term(max_depth, right_vars, current_depth + 1)
            return Mul(left, right)

        # Fallback
        return Zero()

    def set_complexity_range(self, min_complexity: int, max_complexity: int):
        """Update complexity range for generation."""
        self.min_complexity = min_complexity
        self.max_complexity = max_complexity
