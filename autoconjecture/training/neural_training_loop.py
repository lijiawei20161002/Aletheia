"""
Neural training loop for AutoConjecture Phase 2.
Integrates neural generator with curriculum learning.
"""
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import time
import torch

from ..logic.axioms import get_all_axioms
from ..generation.neural_generator import NeuralConjectureGenerator
from ..generation.random_generator import RandomConjectureGenerator
from ..generation.novelty import NoveltyScorer
from ..generation.heuristics import ComplexityEstimator, DiversityFilter
from ..prover.proof_engine import ProofEngine, ProofResult
from ..knowledge.knowledge_base import KnowledgeBase
from ..models.transformer_generator import TransformerGenerator
from ..models.tokenizer import ExpressionTokenizer
from ..models.generator_trainer import GeneratorTrainer, GeneratorTrainingConfig
from ..models.curriculum import CurriculumScheduler, CurriculumConfig


@dataclass
class NeuralTrainingConfig:
    """Configuration for neural training."""
    # Training parameters
    num_epochs: int = 10
    cycles_per_epoch: int = 1000
    random_seed: int = 42

    # Generation parameters
    conjectures_per_cycle: int = 10
    initial_complexity: int = 2
    final_complexity: int = 15

    # Neural model parameters
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.1

    # Generator training
    generator_lr: float = 1e-4
    generator_batch_size: int = 32
    generator_warmup_steps: int = 500

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_progression: str = "success_rate"  # "success_rate" or "steps"
    success_threshold: float = 0.3

    # Prover parameters
    max_proof_depth: int = 50
    max_proof_iterations: int = 1000

    # Training strategy
    pretrain_epochs: int = 5  # Epochs of supervised pretraining
    mixed_generation: bool = True  # Mix neural and random generation
    neural_ratio: float = 0.3  # Ratio of neural vs random (if mixed) - start low for undertrained models

    # Device
    device: str = "cpu"  # "cuda" or "cpu"

    # Logging
    log_interval: int = 100
    checkpoint_interval: int = 1000

    # Resume support
    start_epoch: int = 0


class NeuralTrainingLoop:
    """
    Training loop with neural conjecture generator.

    Implements:
    1. Supervised pretraining on existing proofs
    2. Curriculum learning (simple to complex)
    3. Online learning from new successful proofs
    4. Optional mixing with random generation
    """

    def __init__(
        self,
        config: NeuralTrainingConfig,
        logger: Optional[object] = None
    ):
        """
        Args:
            config: Training configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger

        # Set random seed
        torch.manual_seed(config.random_seed)

        # Initialize tokenizer
        self.tokenizer = ExpressionTokenizer(
            max_length=128,
            var_names=["x", "y", "z", "w"]
        )

        # Initialize neural model
        self.model = TransformerGenerator(
            vocab_size=self.tokenizer.vocab_size,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dropout=config.dropout
        ).to(config.device)

        # Initialize neural generator
        self.generator = NeuralConjectureGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=config.device,
            temperature=1.0,
            top_k=50
        )

        # Initialize random generator for mixed generation
        self.random_generator = RandomConjectureGenerator(
            min_complexity=config.initial_complexity,
            max_complexity=config.final_complexity,
            var_names=["x", "y", "z", "w"],
            seed=config.random_seed
        )

        # Initialize knowledge base
        self.knowledge_base = KnowledgeBase(axioms=get_all_axioms())

        # Initialize prover
        self.prover = ProofEngine(
            max_depth=config.max_proof_depth,
            max_iterations=config.max_proof_iterations,
            knowledge_base=self.knowledge_base.get_all_statements()
        )

        # Initialize filters
        self.novelty_scorer = NoveltyScorer()
        self.complexity_estimator = ComplexityEstimator()
        self.diversity_filter = DiversityFilter()

        # Initialize trainer for generator
        trainer_config = GeneratorTrainingConfig(
            learning_rate=config.generator_lr,
            batch_size=config.generator_batch_size,
            warmup_steps=config.generator_warmup_steps,
            device=config.device
        )
        self.generator_trainer = GeneratorTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=trainer_config,
            knowledge_base=self.knowledge_base
        )

        # Initialize curriculum
        if config.use_curriculum:
            curriculum_config = CurriculumConfig(
                initial_complexity=config.initial_complexity,
                final_complexity=config.final_complexity,
                success_threshold=config.success_threshold
            )
            self.curriculum = CurriculumScheduler(curriculum_config)
        else:
            self.curriculum = None

        # Training state
        self.current_epoch = 0
        self.current_cycle = 0
        self.total_conjectures_generated = 0
        self.total_proofs_attempted = 0
        self.total_proofs_succeeded = 0

    def train(self):
        """Run the main training loop."""
        self._log(f"Starting Phase 2 neural training with config: {self.config}")
        self._log(f"Device: {self.config.device}")
        self._log(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        # Phase 1: Pretrain on existing knowledge (axioms + theorems)
        if self.knowledge_base.total_size() > 0 and self.config.pretrain_epochs > 0:
            self._log("\n" + "="*60)
            self._log("Phase 1: Supervised Pretraining")
            self._log("="*60)
            self._pretrain()

        # Phase 2: Main training loop with curriculum
        self._log("\n" + "="*60)
        self._log("Phase 2: Curriculum Learning")
        self._log("="*60)

        for epoch in range(self.config.start_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self._log(f"\n{'='*60}")
            self._log(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            self._log(f"{'='*60}")

            epoch_start = time.time()
            epoch_proofs = 0
            self.novelty_scorer.reset()  # Reset per-epoch so model can explore fresh each epoch

            for cycle in range(self.config.cycles_per_epoch):
                self.current_cycle = cycle

                # Run one cycle
                cycle_proofs = self._run_cycle()
                epoch_proofs += cycle_proofs

                # Update curriculum
                if self.curriculum and cycle_proofs > 0:
                    if self.curriculum.should_advance_stage():
                        self.curriculum.advance_stage()
                        # Update generator temperature
                        self.generator.set_temperature(
                            self.curriculum.get_current_temperature()
                        )

                # Logging
                if cycle % self.config.log_interval == 0:
                    self._log_progress(epoch, cycle, epoch_proofs)

                # Checkpointing
                if cycle % self.config.checkpoint_interval == 0 and cycle > 0:
                    self._checkpoint()

                # Online training: periodically retrain on new proofs
                if cycle % 100 == 0 and cycle > 0 and self.knowledge_base.size() > 10:
                    self._online_training_step()

            # End of epoch summary
            epoch_time = time.time() - epoch_start
            success_rate = epoch_proofs / self.config.cycles_per_epoch if self.config.cycles_per_epoch > 0 else 0
            self._log(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
            self._log(f"Proofs found: {epoch_proofs}")
            self._log(f"Success rate: {success_rate:.2%}")
            self._log(f"Knowledge base size: {self.knowledge_base.size()}")

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

        # Final checkpoint
        self._checkpoint()

    def _pretrain(self):
        """Pretrain generator on axioms and existing proven theorems."""
        num_axioms = len(self.knowledge_base.axioms)
        num_theorems = self.knowledge_base.size()
        self._log(f"Pretraining on {num_axioms} axioms and {num_theorems} existing theorems")

        if num_theorems > 0:
            # Have existing theorems, use normal knowledge base training
            self.generator_trainer.train_on_knowledge_base(
                num_epochs=self.config.pretrain_epochs,
                curriculum_strategy="complexity"
            )
        else:
            # No theorems yet, bootstrap with axioms only
            self._log("Bootstrapping with axioms only")
            axiom_expressions = self.knowledge_base.axioms
            self.generator_trainer.train_on_expressions(
                expressions=axiom_expressions,
                num_epochs=self.config.pretrain_epochs
            )

        self._log("Pretraining completed")

    def _run_cycle(self) -> int:
        """
        Run one generate-prove-learn cycle.

        Returns:
            Number of successful proofs in this cycle
        """
        proofs_found = 0

        # Generate conjectures using mixed generation strategy
        self.generator.eval_mode()
        conjectures = []

        if self.config.mixed_generation:
            # Mix neural and random generation
            num_neural = int(self.config.conjectures_per_cycle * self.config.neural_ratio)
            num_random = self.config.conjectures_per_cycle - num_neural

            # Generate from neural model
            neural_conjectures = [c for c in self.generator.generate(num_neural) if c is not None]
            conjectures.extend(neural_conjectures)

            # Generate from random generator
            random_conjectures = self.random_generator.generate(num_random)
            conjectures.extend(random_conjectures)
        else:
            # Pure neural generation
            conjectures = [c for c in self.generator.generate(self.config.conjectures_per_cycle) if c is not None]

        self.total_conjectures_generated += len(conjectures)
        print(f"  Generated {len(conjectures)} conjectures")
        if len(conjectures) > 0:
            print(f"  Sample conjectures:")
            for i, conj in enumerate(conjectures[:3]):  # Show first 3
                print(f"    [{i+1}] {conj} (complexity: {conj.complexity()})")

        # Apply curriculum filtering if enabled
        if self.curriculum:
            before_curriculum = len(conjectures)
            min_c, max_c = self.curriculum.get_current_complexity_range()
            print(f"  Curriculum range: [{min_c}, {max_c}]")
            conjectures = self.curriculum.filter_by_complexity(conjectures)
            print(f"  After curriculum filter: {len(conjectures)}/{before_curriculum}")

        # Filter conjectures
        filtered_conjectures = []
        filter_stats = {
            "malformed": 0,
            "not_novel": 0,
            "not_diverse": 0,
            "already_known": 0
        }

        for conj in conjectures:
            # Check if well-formed
            if not self.complexity_estimator.is_well_formed(conj):
                filter_stats["malformed"] += 1
                continue

            # Check if novel
            novelty = self.novelty_scorer.score(conj)
            if novelty < 0.3:
                filter_stats["not_novel"] += 1
                continue

            # Check if diverse
            if not self.diversity_filter.should_keep(conj):
                filter_stats["not_diverse"] += 1
                continue

            # Check if already proven
            if self.knowledge_base.contains(conj):
                filter_stats["already_known"] += 1
                continue

            filtered_conjectures.append(conj)

        print(f"  Filtering: malformed={filter_stats['malformed']}, "
              f"not_novel={filter_stats['not_novel']}, "
              f"not_diverse={filter_stats['not_diverse']}, "
              f"already_known={filter_stats['already_known']}")
        print(f"  Conjectures after filtering: {len(filtered_conjectures)}")

        # Attempt to prove each conjecture
        for conj in filtered_conjectures:
            self.total_proofs_attempted += 1

            # Update prover's knowledge base
            self.prover.knowledge_base = self.knowledge_base.get_all_statements()

            # Attempt proof
            proof = self.prover.prove(conj)

            # Record result for curriculum
            if self.curriculum:
                success = (proof.result == ProofResult.SUCCESS)
                self.curriculum.record_result(success, conj.complexity())

            # Learn from results
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
                self._log(f"    Complexity: {complexity}")

        return proofs_found

    def _online_training_step(self):
        """Perform one step of online training on recent proofs."""
        # Get recent theorems
        recent_count = min(50, self.knowledge_base.size())
        if recent_count < 10:
            return

        theorems = self.knowledge_base.get_all_theorems()[-recent_count:]
        expressions = [t.statement for t in theorems]

        # Quick training step
        self.generator.train_mode()
        loss = self.generator_trainer._train_batch(expressions)
        self.generator.eval_mode()

        self._log(f"  Online training step, loss: {loss:.4f}")

    def _log_progress(self, epoch: int, cycle: int, epoch_proofs: int):
        """Log training progress."""
        kb_size = self.knowledge_base.size()
        success_rate = self.total_proofs_succeeded / self.total_proofs_attempted if self.total_proofs_attempted > 0 else 0

        log_msg = (f"Epoch {epoch + 1}, Cycle {cycle}/{self.config.cycles_per_epoch}: "
                  f"KB size = {kb_size}, Success rate = {success_rate:.2%}, "
                  f"Epoch proofs = {epoch_proofs}")

        if self.curriculum:
            stats = self.curriculum.get_statistics()
            log_msg += f", Complexity = {stats['current_complexity']}, Temp = {stats['current_temperature']:.2f}"

        self._log(log_msg)

    def _checkpoint(self):
        """Save checkpoint of current state."""
        # Save knowledge base
        kb_path = f"data/checkpoints/neural_epoch_{self.current_epoch}_cycle_{self.current_cycle}.json"
        self.knowledge_base.save(kb_path)

        # Save generator model
        model_path = f"data/checkpoints/generator_epoch_{self.current_epoch}_cycle_{self.current_cycle}.pt"
        self.generator.save(model_path)

        # Save trainer checkpoint
        trainer_path = f"data/checkpoints/trainer_epoch_{self.current_epoch}_cycle_{self.current_cycle}.pt"
        self.generator_trainer.save_checkpoint(trainer_path)

        self._log(f"Checkpoint saved: epoch {self.current_epoch}, cycle {self.current_cycle}")

    def _log(self, message: str):
        """Log a message."""
        if self.logger:
            self.logger.log(message)
        else:
            print(message)

    def get_statistics(self) -> dict:
        """Get current training statistics."""
        success_rate = self.total_proofs_succeeded / self.total_proofs_attempted if self.total_proofs_attempted > 0 else 0

        stats = {
            "epoch": self.current_epoch,
            "cycle": self.current_cycle,
            "total_conjectures_generated": self.total_conjectures_generated,
            "total_proofs_attempted": self.total_proofs_attempted,
            "total_proofs_succeeded": self.total_proofs_succeeded,
            "success_rate": success_rate,
            "knowledge_base_size": self.knowledge_base.size(),
            **self.knowledge_base.get_statistics()
        }

        if self.curriculum:
            stats['curriculum'] = self.curriculum.get_statistics()

        return stats
