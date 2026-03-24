"""
Knowledge base for storing and retrieving proven theorems.
"""
from __future__ import annotations
from typing import List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

from ..logic.expressions import Expression
from ..prover.proof_engine import Proof, ProofResult


@dataclass
class Theorem:
    """A proven theorem with metadata."""
    statement: Expression
    proof: Proof
    complexity: float
    timestamp: str  # ISO format
    epoch: int  # Training epoch when discovered
    cycle: int  # Cycle within epoch

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "statement": str(self.statement),
            "proof_length": self.proof.length(),
            "proof_steps": [str(step) for step in self.proof.steps],
            "complexity": self.complexity,
            "timestamp": self.timestamp,
            "epoch": self.epoch,
            "cycle": self.cycle,
        }

    def __str__(self) -> str:
        return f"Theorem[{self.epoch}.{self.cycle}]: {self.statement}"


class KnowledgeBase:
    """
    Storage for proven theorems.
    Maintains index for efficient retrieval and search.
    """

    def __init__(self, axioms: Optional[List[Expression]] = None):
        """
        Args:
            axioms: Initial axioms to include in knowledge base
        """
        self.theorems: List[Theorem] = []
        self.axioms: List[Expression] = axioms if axioms else []

        # Index for fast lookup
        self.statement_index: Set[str] = set()

    def add_theorem(
        self,
        statement: Expression,
        proof: Proof,
        complexity: float,
        epoch: int,
        cycle: int
    ) -> bool:
        """
        Add a proven theorem to the knowledge base.

        Args:
            statement: The proven statement
            proof: Proof of the statement
            complexity: Estimated complexity
            epoch: Current training epoch
            cycle: Current cycle

        Returns:
            True if added (new theorem), False if already exists
        """
        # Check if already proven
        stmt_str = str(statement)
        if stmt_str in self.statement_index:
            return False  # Already have this theorem

        # Create theorem object
        theorem = Theorem(
            statement=statement,
            proof=proof,
            complexity=complexity,
            timestamp=datetime.now().isoformat(),
            epoch=epoch,
            cycle=cycle
        )

        # Add to storage
        self.theorems.append(theorem)
        self.statement_index.add(stmt_str)

        return True

    def contains(self, statement: Expression) -> bool:
        """Check if statement is already proven."""
        return str(statement) in self.statement_index

    def get_all_statements(self) -> List[Expression]:
        """Get all proven statements (theorems + axioms)."""
        return self.axioms + [thm.statement for thm in self.theorems]

    def get_theorems(self) -> List[Theorem]:
        """Get all theorems."""
        return self.theorems

    def get_all_theorems(self) -> List[Theorem]:
        """Get all theorems (alias for get_theorems for compatibility)."""
        return self.theorems

    def get_recent_theorems(self, n: int = 10) -> List[Theorem]:
        """Get n most recent theorems."""
        return self.theorems[-n:]

    def size(self) -> int:
        """Return number of theorems (excluding axioms)."""
        return len(self.theorems)

    def total_size(self) -> int:
        """Return total number of statements (axioms + theorems)."""
        return len(self.axioms) + len(self.theorems)

    def get_by_complexity(self, min_complexity: float, max_complexity: float) -> List[Theorem]:
        """Get theorems within complexity range."""
        return [
            thm for thm in self.theorems
            if min_complexity <= thm.complexity <= max_complexity
        ]

    def get_statistics(self) -> dict:
        """Get statistics about the knowledge base."""
        if not self.theorems:
            return {
                "num_theorems": 0,
                "num_axioms": len(self.axioms),
                "total_statements": len(self.axioms),
                "avg_proof_length": 0.0,
                "avg_complexity": 0.0,
            }

        proof_lengths = [thm.proof.length() for thm in self.theorems]
        complexities = [thm.complexity for thm in self.theorems]

        return {
            "num_theorems": len(self.theorems),
            "num_axioms": len(self.axioms),
            "total_statements": self.total_size(),
            "avg_proof_length": sum(proof_lengths) / len(proof_lengths),
            "avg_complexity": sum(complexities) / len(complexities),
            "min_complexity": min(complexities),
            "max_complexity": max(complexities),
        }

    def save(self, filepath: str):
        """
        Save knowledge base to file.

        Args:
            filepath: Path to save file
        """
        data = {
            "axioms": [str(axiom) for axiom in self.axioms],
            "theorems": [thm.to_dict() for thm in self.theorems],
            "metadata": {
                "num_theorems": len(self.theorems),
                "saved_at": datetime.now().isoformat(),
            }
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """
        Load knowledge base from file.

        Args:
            filepath: Path to load from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Note: This is a simplified load that only loads metadata
        # Full reconstruction of Expression objects would require parsing
        # For now, we just rebuild the statement index
        self.statement_index.clear()
        for thm_data in data.get("theorems", []):
            self.statement_index.add(thm_data["statement"])

    def clear(self):
        """Clear all theorems (keep axioms)."""
        self.theorems.clear()
        self.statement_index.clear()

    def __len__(self) -> int:
        """Return number of theorems."""
        return len(self.theorems)

    def __str__(self) -> str:
        stats = self.get_statistics()
        return (f"KnowledgeBase(theorems={stats['num_theorems']}, "
                f"axioms={stats['num_axioms']}, "
                f"avg_complexity={stats['avg_complexity']:.2f})")
