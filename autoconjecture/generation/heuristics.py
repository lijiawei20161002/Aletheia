"""
Heuristics for estimating conjecture properties.
Helps guide generation and filtering.
"""
from ..logic.expressions import Expression, Equation, Forall, Exists


class ComplexityEstimator:
    """
    Estimates the difficulty/complexity of proving a conjecture.
    """

    def estimate(self, conjecture: Expression) -> float:
        """
        Estimate proof complexity.
        Returns rough estimate of proof difficulty (higher = harder).

        Args:
            conjecture: Expression to analyze

        Returns:
            Complexity estimate (positive float)
        """
        # Start with syntactic complexity
        base_complexity = conjecture.complexity()

        # Add penalties for specific patterns
        penalty = 0.0

        # Quantifiers make things harder
        if isinstance(conjecture, Forall) or isinstance(conjecture, Exists):
            penalty += 2.0

        # Nested quantifiers are even harder
        penalty += self._count_nested_quantifiers(conjecture) * 3.0

        # Multiple free variables make things harder
        num_free_vars = len(conjecture.free_vars())
        penalty += num_free_vars * 1.0

        return base_complexity + penalty

    def _count_nested_quantifiers(self, expr: Expression, depth: int = 0) -> int:
        """Count maximum nesting depth of quantifiers."""
        if isinstance(expr, Forall) or isinstance(expr, Exists):
            return 1 + self._count_nested_quantifiers(expr.body, depth + 1)
        elif isinstance(expr, Equation):
            return 0
        # Add more cases as needed
        return 0

    def is_trivial(self, conjecture: Expression) -> bool:
        """
        Check if conjecture is trivially true or false.

        Args:
            conjecture: Expression to check

        Returns:
            True if trivially solvable
        """
        # Check for x = x (reflexivity)
        if isinstance(conjecture, Equation):
            if str(conjecture.left) == str(conjecture.right):
                return True

        # Check for forall x. x = x
        if isinstance(conjecture, Forall):
            if isinstance(conjecture.body, Equation):
                body_left = str(conjecture.body.left)
                body_right = str(conjecture.body.right)
                if body_left == body_right:
                    return True

        return False

    def is_well_formed(self, conjecture: Expression) -> bool:
        """
        Check if conjecture is well-formed and interesting.

        Args:
            conjecture: Expression to check

        Returns:
            True if well-formed
        """
        # Check complexity bounds
        complexity = conjecture.complexity()
        if complexity < 2 or complexity > 50:
            return False

        # Check that it's not trivial
        if self.is_trivial(conjecture):
            return False

        # Check that quantified variables are actually used
        if isinstance(conjecture, Forall) or isinstance(conjecture, Exists):
            bound_var = conjecture.var.name
            body_vars = conjecture.body.free_vars()
            if bound_var not in body_vars:
                return False  # Quantified var not used

        return True


class DiversityFilter:
    """
    Filters conjectures to maintain diversity in the training set.
    """

    def __init__(self, max_similar: int = 3):
        """
        Args:
            max_similar: Maximum number of similar conjectures to keep
        """
        self.max_similar = max_similar
        self.patterns = {}  # pattern -> count

    def should_keep(self, conjecture: Expression) -> bool:
        """
        Decide whether to keep a conjecture based on diversity.

        Args:
            conjecture: Expression to evaluate

        Returns:
            True if should be kept
        """
        pattern = self._get_pattern(conjecture)

        if pattern not in self.patterns:
            self.patterns[pattern] = 0

        if self.patterns[pattern] >= self.max_similar:
            return False  # Too many similar conjectures

        self.patterns[pattern] += 1
        return True

    def _get_pattern(self, expr: Expression) -> str:
        """
        Extract structural pattern from expression.
        Replaces specific terms with placeholders.
        """
        # Simple pattern: just use the structure
        # e.g., "forall x. (x + ?) = ?" where ? is any term
        s = str(expr)

        # Replace variables with X
        for var_name in expr.free_vars():
            s = s.replace(var_name, "X")

        # Replace numbers with N
        s = s.replace("0", "N")
        s = s.replace("S(N)", "N")

        return s

    def reset(self):
        """Clear pattern counts."""
        self.patterns.clear()
