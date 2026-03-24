"""
Proof engine that searches for proofs using tactics.
Uses best-first search with heuristics.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import heapq

from ..logic.expressions import Expression
from .tactics import Tactic, ProofState, DEFAULT_TACTICS


class ProofResult(Enum):
    """Result of a proof attempt."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


@dataclass
class ProofStep:
    """A step in a proof."""
    tactic: str  # Name of tactic used
    state_before: ProofState
    state_after: Optional[ProofState]  # None if proof complete

    def __str__(self) -> str:
        if self.state_after is None:
            return f"  {self.tactic} → QED"
        return f"  {self.tactic} → {self.state_after.goal}"


@dataclass
class Proof:
    """Complete proof of a statement."""
    statement: Expression
    steps: List[ProofStep]
    result: ProofResult

    def __str__(self) -> str:
        steps_str = "\n".join(str(step) for step in self.steps)
        return f"Proof of: {self.statement}\n{steps_str}\nResult: {self.result.value}"

    def length(self) -> int:
        """Return proof length (number of steps)."""
        return len(self.steps)


@dataclass
class SearchNode:
    """
    Node in proof search tree.
    Priority queue is sorted by heuristic score (lower is better).
    """
    state: ProofState
    parent: Optional[SearchNode]
    tactic_used: Optional[str]
    heuristic_score: float

    def __lt__(self, other: SearchNode) -> bool:
        """Comparison for priority queue."""
        return self.heuristic_score < other.heuristic_score


class ProofEngine:
    """
    Engine for searching for proofs.
    Uses best-first search with configurable tactics.
    """

    def __init__(
        self,
        tactics: Optional[List[Tactic]] = None,
        max_depth: int = 50,
        max_iterations: int = 1000,
        knowledge_base: Optional[List[Expression]] = None
    ):
        """
        Args:
            tactics: List of tactics to use (default: DEFAULT_TACTICS)
            max_depth: Maximum proof depth
            max_iterations: Maximum search iterations
            knowledge_base: Known theorems/axioms to use
        """
        self.tactics = tactics if tactics is not None else DEFAULT_TACTICS
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.knowledge_base = knowledge_base if knowledge_base is not None else []

    def prove(
        self,
        goal: Expression,
        hypotheses: Optional[List[Expression]] = None
    ) -> Proof:
        """
        Attempt to prove a goal given hypotheses.

        Args:
            goal: Expression to prove
            hypotheses: Known facts to use in proof

        Returns:
            Proof object with result and steps
        """
        initial_state = ProofState(
            goal=goal,
            hypotheses=hypotheses if hypotheses is not None else [],
            depth=0
        )

        # Priority queue for best-first search
        # (heuristic_score, counter, node)
        pq = []
        counter = 0  # Tie-breaker for heap
        initial_node = SearchNode(
            state=initial_state,
            parent=None,
            tactic_used=None,
            heuristic_score=self._heuristic(initial_state)
        )
        heapq.heappush(pq, (initial_node.heuristic_score, counter, initial_node))
        counter += 1

        iterations = 0
        visited = set()  # Track visited states to avoid loops

        while pq and iterations < self.max_iterations:
            iterations += 1

            # Get most promising node
            _, _, current_node = heapq.heappop(pq)
            current_state = current_node.state

            # Check if we've seen this state before
            state_str = str(current_state.goal)
            if state_str in visited:
                continue
            visited.add(state_str)

            # Check depth limit
            if current_state.depth > self.max_depth:
                continue

            # Try each tactic
            for tactic in self.tactics:
                new_states = tactic.apply(current_state, self.knowledge_base)

                # If no new states, proof is complete!
                if len(new_states) == 0:
                    proof_steps = self._reconstruct_proof(current_node, tactic)
                    return Proof(
                        statement=goal,
                        steps=proof_steps,
                        result=ProofResult.SUCCESS
                    )

                # Add new states to search queue
                for new_state in new_states:
                    # Skip if state didn't change (tactic was not applicable)
                    if str(new_state.goal) == str(current_state.goal):
                        continue

                    new_node = SearchNode(
                        state=new_state,
                        parent=current_node,
                        tactic_used=tactic.name(),
                        heuristic_score=self._heuristic(new_state)
                    )
                    heapq.heappush(pq, (new_node.heuristic_score, counter, new_node))
                    counter += 1

        # Search failed
        return Proof(
            statement=goal,
            steps=[],
            result=ProofResult.TIMEOUT if iterations >= self.max_iterations else ProofResult.FAILURE
        )

    def _heuristic(self, state: ProofState) -> float:
        """
        Heuristic function for search.
        Lower score = more promising state.
        """
        # Simple heuristic: prefer states with simpler goals
        goal_complexity = state.goal.complexity()

        # Penalize deeper proofs
        depth_penalty = state.depth * 0.5

        return goal_complexity + depth_penalty

    def _reconstruct_proof(self, final_node: SearchNode, final_tactic: Tactic) -> List[ProofStep]:
        """Reconstruct proof steps by walking back through search tree."""
        steps = []

        # Walk back from final node to initial node
        current = final_node
        while current.parent is not None:
            step = ProofStep(
                tactic=current.tactic_used,
                state_before=current.parent.state,
                state_after=current.state
            )
            steps.append(step)
            current = current.parent

        # Add final step
        final_step = ProofStep(
            tactic=final_tactic.name(),
            state_before=final_node.state,
            state_after=None  # None means proof complete
        )
        steps.append(final_step)

        # Reverse to get correct order
        steps.reverse()
        return steps

    def add_to_knowledge_base(self, theorem: Expression):
        """Add a proven theorem to the knowledge base."""
        self.knowledge_base.append(theorem)

    def get_knowledge_base_size(self) -> int:
        """Get number of theorems in knowledge base."""
        return len(self.knowledge_base)
