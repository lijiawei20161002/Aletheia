"""
Training logic for neural conjecture generator.
Learns from successful proofs to generate more provable conjectures.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np

from ..logic.expressions import Expression
from ..knowledge.knowledge_base import KnowledgeBase
from .transformer_generator import TransformerGenerator
from .tokenizer import ExpressionTokenizer


@dataclass
class GeneratorTrainingConfig:
    """Configuration for generator training."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    device: str = "cpu"
    checkpoint_interval: int = 1000
    log_interval: int = 100


class GeneratorTrainer:
    """
    Trains the neural generator to produce provable conjectures.

    Training strategy:
    1. Supervised learning on successful proofs (learn to imitate)
    2. Curriculum learning (start with simple, progress to complex)
    3. Diversity regularization (avoid mode collapse)
    """

    def __init__(
        self,
        model: TransformerGenerator,
        tokenizer: ExpressionTokenizer,
        config: GeneratorTrainingConfig,
        knowledge_base: Optional[KnowledgeBase] = None
    ):
        """
        Args:
            model: Transformer generator model
            tokenizer: Expression tokenizer
            config: Training configuration
            knowledge_base: Knowledge base with successful proofs
        """
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.config = config
        self.knowledge_base = knowledge_base

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )

        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: self._lr_lambda(step)
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Metrics
        self.train_losses = []
        self.val_losses = []

    def _lr_lambda(self, step: int) -> float:
        """Learning rate schedule with warmup."""
        if step < self.config.warmup_steps:
            return step / max(1, self.config.warmup_steps)
        return max(0.1, 1.0 / np.sqrt(step / self.config.warmup_steps))

    def train_on_expressions(
        self,
        expressions: List[Expression],
        num_epochs: Optional[int] = None,
        curriculum_strategy: str = "complexity"
    ):
        """
        Train on a list of expressions directly.

        Args:
            expressions: List of expressions to train on
            num_epochs: Number of training epochs (uses config if None)
            curriculum_strategy: How to order training examples
                - "complexity": Simple to complex
                - "chronological": Keep given order
                - "random": Random order
        """
        if not expressions:
            raise ValueError("No expressions provided for training")

        num_epochs = num_epochs or self.config.num_epochs

        print(f"Training on {len(expressions)} expressions")

        # Apply curriculum strategy
        if curriculum_strategy == "complexity":
            # Sort by complexity (simple first)
            expressions = sorted(
                expressions,
                key=lambda e: e.complexity()
            )
        elif curriculum_strategy == "chronological":
            # Keep given order
            pass
        elif curriculum_strategy == "random":
            np.random.shuffle(expressions)

        # Train for multiple epochs
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch(expressions)
            self.train_losses.append(epoch_loss)

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")

    def train_on_knowledge_base(
        self,
        num_epochs: Optional[int] = None,
        curriculum_strategy: str = "complexity"
    ):
        """
        Train on expressions from knowledge base.

        Args:
            num_epochs: Number of training epochs (uses config if None)
            curriculum_strategy: How to order training examples
                - "complexity": Simple to complex
                - "chronological": Order they were proven
                - "random": Random order
        """
        if self.knowledge_base is None:
            raise ValueError("Knowledge base required for training")

        # Get training data from knowledge base (axioms + theorems)
        theorems = self.knowledge_base.get_all_theorems()
        axioms = self.knowledge_base.axioms

        # Combine axioms and theorem statements
        expressions = axioms + [theorem.statement for theorem in theorems]

        if not expressions:
            raise ValueError("Knowledge base is empty, cannot train")

        print(f"Training on {len(axioms)} axioms and {len(theorems)} proven theorems")

        # Use the common training method
        self.train_on_expressions(
            expressions=expressions,
            num_epochs=num_epochs,
            curriculum_strategy=curriculum_strategy
        )

    def _train_epoch(self, expressions: List[Expression]) -> float:
        """Train for one epoch on expressions."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Shuffle expressions
        indices = np.random.permutation(len(expressions))

        # Create batches
        for batch_start in range(0, len(expressions), self.config.batch_size):
            batch_indices = indices[batch_start:batch_start + self.config.batch_size]
            batch_expressions = [expressions[i] for i in batch_indices]

            # Train on batch
            loss = self._train_batch(batch_expressions)
            total_loss += loss
            num_batches += 1

            # Logging
            if self.global_step % self.config.log_interval == 0:
                print(f"  Step {self.global_step}, Batch loss: {loss:.4f}")

            self.global_step += 1

        return total_loss / max(num_batches, 1)

    def _train_batch(self, expressions: List[Expression]) -> float:
        """Train on a single batch of expressions."""
        # Encode expressions to token sequences
        token_sequences = self.tokenizer.batch_encode(expressions, pad=True)

        # Convert to tensors
        # Shape: (batch_size, seq_len)
        batch_tensor = torch.tensor(token_sequences, dtype=torch.long, device=self.config.device)

        # Prepare input and target
        # Input: all tokens except last
        # Target: all tokens except first
        input_seq = batch_tensor[:, :-1]
        target_seq = batch_tensor[:, 1:]

        # Transpose to (seq_len, batch_size) for transformer
        input_seq = input_seq.transpose(0, 1)
        target_seq = target_seq.transpose(0, 1)

        # Create empty source for unconditional generation
        src = torch.empty(0, input_seq.size(1), dtype=torch.long, device=self.config.device)

        # Forward pass
        logits = self.model(
            src=src,
            tgt=input_seq,
            tgt_mask=None,  # Model creates causal mask internally
            src_key_padding_mask=None,
            tgt_key_padding_mask=None
        )

        # Compute loss
        loss = self.model.compute_loss(
            logits,
            target_seq,
            pad_token_id=self.tokenizer.pad_id
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip
        )

        # Update weights
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train_with_reinforcement(
        self,
        prover,
        num_iterations: int = 100,
        samples_per_iteration: int = 10,
        reward_scale: float = 1.0
    ):
        """
        Train with reinforcement learning from proof attempts.

        Args:
            prover: Proof engine to evaluate generated conjectures
            num_iterations: Number of RL iterations
            samples_per_iteration: Conjectures to generate per iteration
            reward_scale: Scaling factor for rewards

        Strategy:
        - Generate conjectures
        - Attempt to prove them
        - Higher reward for successful proofs
        - Use policy gradient to update model
        """
        self.model.train()

        for iteration in range(num_iterations):
            # Generate conjectures
            generated_sequences = self.model.generate(
                batch_size=samples_per_iteration,
                max_length=self.tokenizer.max_length,
                sos_token_id=self.tokenizer.sos_id,
                eos_token_id=self.tokenizer.eos_id,
                pad_token_id=self.tokenizer.pad_id,
                device=self.config.device
            )

            # Decode and evaluate
            rewards = []
            valid_sequences = []

            for seq in generated_sequences:
                expr = self.tokenizer.decode_tokens(seq.cpu().tolist())

                if expr is None:
                    reward = -1.0  # Penalty for invalid expressions
                else:
                    # Try to prove
                    proof = prover.prove(expr)
                    if proof.result.value == "SUCCESS":
                        reward = reward_scale  # Reward for successful proof
                    else:
                        reward = -0.1  # Small penalty for unprovable

                rewards.append(reward)
                valid_sequences.append(seq)

            # Compute policy gradient loss
            # This is a simplified version - full RL would use PPO or similar
            rewards_tensor = torch.tensor(rewards, device=self.config.device)

            # Normalize rewards
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

            # Compute loss (simplified policy gradient)
            # In full implementation, would compute log probabilities and multiply by rewards
            # For now, we'll just do supervised learning on successful examples

            successful_seqs = [
                seq for seq, r in zip(valid_sequences, rewards) if r > 0
            ]

            if successful_seqs:
                # Train on successful generations
                expressions = []
                for seq in successful_seqs:
                    expr = self.tokenizer.decode_tokens(seq.cpu().tolist())
                    if expr is not None:
                        expressions.append(expr)

                if expressions:
                    loss = self._train_batch(expressions)
                    print(f"RL Iteration {iteration + 1}/{num_iterations}, "
                          f"Success rate: {len(expressions)}/{samples_per_iteration}, "
                          f"Loss: {loss:.4f}")

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'train_losses': self.train_losses,
            'config': self.config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])

    def get_statistics(self) -> Dict:
        """Get training statistics."""
        return {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'train_losses': self.train_losses,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
