"""
Novelty scoring for generated conjectures.
Measures how different a conjecture is from previously generated ones.
"""
from typing import List, Set
from ..logic.expressions import Expression


class NoveltyScorer:
    """
    Scores how novel a conjecture is compared to existing ones.
    Uses simple edit distance on string representations.
    """

    def __init__(self):
        self.seen_conjectures: Set[str] = set()
        self.conjecture_history: List[str] = []

    def score(self, conjecture: Expression) -> float:
        """
        Compute novelty score for a conjecture.
        Returns value in [0, 1] where 1 is most novel.

        Args:
            conjecture: Expression to score

        Returns:
            Novelty score (higher = more novel)
        """
        conj_str = str(conjecture)

        # Check if we've seen this exact conjecture
        if conj_str in self.seen_conjectures:
            return 0.0  # Not novel at all

        # If this is the first conjecture, it's maximally novel
        if not self.conjecture_history:
            return 1.0

        # Compute minimum edit distance to any previous conjecture
        min_distance = float('inf')
        for prev_conj in self.conjecture_history[-100:]:  # Only check recent ones
            distance = self._edit_distance(conj_str, prev_conj)
            min_distance = min(min_distance, distance)

        # Normalize by length
        max_len = max(len(conj_str), len(self.conjecture_history[-1]))
        normalized_distance = min_distance / max(max_len, 1)

        return min(1.0, normalized_distance)

    def add(self, conjecture: Expression):
        """Add a conjecture to history."""
        conj_str = str(conjecture)
        self.seen_conjectures.add(conj_str)
        self.conjecture_history.append(conj_str)

    def _edit_distance(self, s1: str, s2: str) -> int:
        """
        Compute Levenshtein edit distance between two strings.
        Classic dynamic programming algorithm.
        """
        m, n = len(s1), len(s2)

        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )

        return dp[m][n]

    def reset(self):
        """Clear history."""
        self.seen_conjectures.clear()
        self.conjecture_history.clear()

    def size(self) -> int:
        """Return number of unique conjectures seen."""
        return len(self.seen_conjectures)
