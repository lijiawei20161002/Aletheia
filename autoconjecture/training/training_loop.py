"""
Main training loop for AutoConjecture.
Implements the generate-prove-learn cycle.
"""
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import time

from ..logic.axioms import PEANO_AXIOMS
from ..generation.random_generator import RandomConjectureGenerator
from ..generation.novelty import NoveltyScorer
from ..generation.heuristics import ComplexityEstimator, DiversityFilter
from ..prover.proof_engine import ProofEngine, ProofResult
from ..knowledge.knowledge_base import KnowledgeBase


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Training parameters
    num_epochs: int = 10
    cycles_per_epoch: int = 1000
    random_seed: int = 42

    # Generation parameters
    min_complexity: int = 2
    max_complexity: int = 10
    conjectures_per_cycle: int = 10

    # Prover parameters
    max_proof_depth: int = 50
    max_proof_iterations: int = 1000

    # Logging
    log_interval: int = 100  # Log every N cycles
    checkpoint_interval: int = 1000  # Save every N cycles


class TrainingLoop:
    """
    Main training loop that orchestrates the generate-prove-learn cycle.
    """

    def __init__(
        self,
        config: TrainingConfig,
        logger: Optional[object] = None
    ):
        """
        Args:
            config: Training configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger

        # Initialize components
        self.generator = RandomConjectureGenerator(
            min_complexity=config.min_complexity,
            max_complexity=config.max_complexity,
            seed=config.random_seed
        )

        self.knowledge_base = KnowledgeBase(axioms=PEANO_AXIOMS)

        self.prover = ProofEngine(
            max_depth=config.max_proof_depth,
            max_iterations=config.max_proof_iterations,
            knowledge_base=self.knowledge_base.get_all_statements()
        )

        self.novelty_scorer = NoveltyScorer()
        self.complexity_estimator = ComplexityEstimator()
        self.diversity_filter = DiversityFilter()

        # Training state
        self.current_epoch = 0
        self.current_cycle = 0
        self.total_conjectures_generated = 0
        self.total_proofs_attempted = 0
        self.total_proofs_succeeded = 0

    def train(self):
        """Run the main training loop."""
        self._log(f"Starting training with config: {self.config}")
        self._log(f"Initial knowledge base: {self.knowledge_base}")

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            self._log(f"\n{'='*60}")
            self._log(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            self._log(f"{'='*60}")

            epoch_start = time.time()
            epoch_proofs = 0

            for cycle in range(self.config.cycles_per_epoch):
                self.current_cycle = cycle

                # Run one cycle
                cycle_proofs = self._run_cycle()
                epoch_proofs += cycle_proofs

                # Logging
                if cycle % self.config.log_interval == 0:
                    self._log_progress(epoch, cycle, epoch_proofs)

                # Checkpointing
                if cycle % self.config.checkpoint_interval == 0 and cycle > 0:
                    self._checkpoint()

            # End of epoch summary
            epoch_time = time.time() - epoch_start
            success_rate = epoch_proofs / self.config.cycles_per_epoch if self.config.cycles_per_epoch > 0 else 0
            self._log(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
            self._log(f"Proofs found: {epoch_proofs}")
            self._log(f"Success rate: {success_rate:.2%}")
            self._log(f"Knowledge base size: {self.knowledge_base.size()}")
            self._log(f"{self.knowledge_base}")

        # Training complete
        total_time = time.time() - start_time
        self._log(f"\n{'='*60}")
        self._log(f"Training completed!")
        self._log(f"{'='*60}")
        self._log(f"Total time: {total_time:.2f}s")
        self._log(f"Total conjectures generated: {self.total_conjectures_generated}")
        self._log(f"Total proofs attempted: {self.total_proofs_attempted}")
        self._log(f"Total proofs succeeded: {self.total_proofs_succeeded}")
        overall_success = self.total_proofs_succeeded / self.total_proofs_attempted if self.total_proofs_attempted > 0 else 0
        self._log(f"Overall success rate: {overall_success:.2%}")
        self._log(f"Final knowledge base: {self.knowledge_base}")

        # Final checkpoint
        self._checkpoint()

    def _run_cycle(self) -> int:
        """
        Run one generate-prove-learn cycle.

        Returns:
            Number of successful proofs in this cycle
        """
        proofs_found = 0

        # 1. Generate conjectures
        conjectures = self.generator.generate(self.config.conjectures_per_cycle)
        self.total_conjectures_generated += len(conjectures)

        # 2. Filter conjectures
        filtered_conjectures = []
        for conj in conjectures:
            # Check if well-formed
            if not self.complexity_estimator.is_well_formed(conj):
                continue

            # Check if novel
            novelty = self.novelty_scorer.score(conj)
            if novelty < 0.3:  # Too similar to existing
                continue

            # Check if diverse
            if not self.diversity_filter.should_keep(conj):
                continue

            # Check if already proven
            if self.knowledge_base.contains(conj):
                continue

            filtered_conjectures.append(conj)

        # 3. Attempt to prove each conjecture
        for conj in filtered_conjectures:
            self.total_proofs_attempted += 1

            # Update prover's knowledge base
            self.prover.knowledge_base = self.knowledge_base.get_all_statements()

            # Attempt proof
            proof = self.prover.prove(conj)

            # 4. Learn from results
            if proof.result == ProofResult.SUCCESS:
                proofs_found += 1
                self.total_proofs_succeeded += 1

                # Add to knowledge base
                complexity = self.complexity_estimator.estimate(conj)
                self.knowledge_base.add_theorem(
                    statement=conj,
                    proof=proof,
                    complexity=complexity,
                    epoch=self.current_epoch,
                    cycle=self.current_cycle
                )

                # Update novelty tracker
                self.novelty_scorer.add(conj)

                self._log(f"  ✓ Proved: {conj}")
                self._log(f"    Proof length: {proof.length()} steps")

        return proofs_found

    def _log_progress(self, epoch: int, cycle: int, epoch_proofs: int):
        """Log training progress."""
        kb_size = self.knowledge_base.size()
        success_rate = self.total_proofs_succeeded / self.total_proofs_attempted if self.total_proofs_attempted > 0 else 0

        self._log(f"Epoch {epoch + 1}, Cycle {cycle}/{self.config.cycles_per_epoch}: "
                 f"KB size = {kb_size}, Success rate = {success_rate:.2%}, "
                 f"Epoch proofs = {epoch_proofs}")

    def _checkpoint(self):
        """Save checkpoint of current state."""
        checkpoint_path = f"data/checkpoints/epoch_{self.current_epoch}_cycle_{self.current_cycle}.json"
        self.knowledge_base.save(checkpoint_path)
        self._log(f"Checkpoint saved: {checkpoint_path}")

    def _log(self, message: str):
        """Log a message."""
        if self.logger:
            self.logger.log(message)
        else:
            print(message)

    def get_statistics(self) -> dict:
        """Get current training statistics."""
        success_rate = self.total_proofs_succeeded / self.total_proofs_attempted if self.total_proofs_attempted > 0 else 0

        return {
            "epoch": self.current_epoch,
            "cycle": self.current_cycle,
            "total_conjectures_generated": self.total_conjectures_generated,
            "total_proofs_attempted": self.total_proofs_attempted,
            "total_proofs_succeeded": self.total_proofs_succeeded,
            "success_rate": success_rate,
            "knowledge_base_size": self.knowledge_base.size(),
            **self.knowledge_base.get_statistics()
        }
