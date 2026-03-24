"""
Phase 3 training loop: RL-based prover with PPO.

Jointly trains:
  - StateEncoder + ActorCritic (PPO on proof trajectories)
  - TransformerGenerator (online learning from new proofs)

Pipeline per epoch:
  1. Generate conjectures (neural + random)
  2. Attempt proofs with RL prover → collect trajectories
  3. PPO update on trajectories every update_interval conjectures
  4. Online generator update every generator_update_interval cycles
  5. Checkpoint and log statistics
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from ..logic.axioms import get_all_axioms
from ..logic.parser import parse_expression
from ..generation.neural_generator import NeuralConjectureGenerator
from ..generation.random_generator import RandomConjectureGenerator
from ..generation.novelty import NoveltyScorer
from ..generation.heuristics import ComplexityEstimator, DiversityFilter
from ..prover.proof_engine import Proof, ProofEngine, ProofResult
from ..prover.rl_proof_engine import RLProofEngine
from ..knowledge.knowledge_base import KnowledgeBase
from ..models.transformer_generator import TransformerGenerator
from ..models.tokenizer import ExpressionTokenizer
from ..models.generator_trainer import GeneratorTrainer, GeneratorTrainingConfig
from ..models.curriculum import CurriculumScheduler, CurriculumConfig
from ..rl.state_encoder import StateEncoder
from ..rl.actor_critic import ActorCritic
from ..rl.replay_buffer import RolloutBuffer, Transition
from ..rl.ppo_trainer import PPOTrainer, PPOConfig


@dataclass
class Phase3Config:
    """Full configuration for Phase 3 training."""

    # ── training schedule ───────────────────────────────────────────────
    num_epochs: int = 20
    cycles_per_epoch: int = 500
    conjectures_per_cycle: int = 10
    random_seed: int = 42

    # ── generation ──────────────────────────────────────────────────────
    initial_complexity: int = 6
    final_complexity: int = 15
    neural_ratio: float = 0.5        # Fraction of neural vs random generation
    use_curriculum: bool = True
    success_threshold: float = 0.25  # Curriculum advance threshold

    # ── RL prover ───────────────────────────────────────────────────────
    max_proof_steps: int = 30
    update_interval: int = 50        # PPO update every N conjectures

    # ── state encoder ───────────────────────────────────────────────────
    encoder_d_model: int = 256
    encoder_nhead: int = 4
    encoder_num_layers: int = 3
    encoder_dropout: float = 0.1

    # ── actor-critic ────────────────────────────────────────────────────
    ac_hidden_dim: int = 256

    # ── PPO ─────────────────────────────────────────────────────────────
    ppo_clip_epsilon: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_epochs: int = 4
    ppo_mini_batch_size: int = 64
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_lr: float = 3e-4
    ppo_max_grad_norm: float = 0.5
    ppo_target_kl: float = 0.05

    # ── neural generator ────────────────────────────────────────────────
    gen_d_model: int = 256
    gen_nhead: int = 8
    gen_num_layers: int = 6
    gen_lr: float = 1e-4
    gen_batch_size: int = 32
    gen_warmup_steps: int = 500
    gen_pretrain_epochs: int = 3
    gen_update_interval: int = 100   # Online generator update every N cycles

    # ── heuristic fallback ──────────────────────────────────────────────
    use_heuristic_fallback: bool = True   # Try heuristic prover on RL failures
    heuristic_max_depth: int = 50
    heuristic_max_iter: int = 500

    # ── warmup (behavioral cloning) ─────────────────────────────────────
    bc_warmup_cycles: int = 200      # Pure heuristic-prover cycles before PPO

    # ── device & paths ──────────────────────────────────────────────────
    device: str = "cuda"
    checkpoint_dir: str = "data/checkpoints"
    experiment_name: str = "phase3_a100"
    checkpoint_interval: int = 500
    log_interval: int = 50
    kb_checkpoint: Optional[str] = None  # Load Phase 2 KB for warm-start


class Phase3TrainingLoop:
    """
    Phase 3 training loop: RL prover + neural generator, trained jointly.
    """

    def __init__(
        self,
        config: Phase3Config,
        logger=None,
    ):
        self.config = config
        self.logger = logger

        torch.manual_seed(config.random_seed)

        self._log(f"Phase 3 | device={config.device} | experiment={config.experiment_name}")
        self._log(f"  Config: epochs={config.num_epochs}, cycles/epoch={config.cycles_per_epoch}")

        # ── tokenizer (shared between generator and encoder) ────────────
        self.tokenizer = ExpressionTokenizer(
            max_length=128, var_names=["x", "y", "z", "w"]
        )

        # ── knowledge base ───────────────────────────────────────────────
        self.kb = KnowledgeBase(axioms=get_all_axioms())
        if config.kb_checkpoint and os.path.exists(config.kb_checkpoint):
            self._load_kb_from_checkpoint(config.kb_checkpoint)

        # ── state encoder ────────────────────────────────────────────────
        self.encoder = StateEncoder(
            tokenizer=self.tokenizer,
            d_model=config.encoder_d_model,
            nhead=config.encoder_nhead,
            num_layers=config.encoder_num_layers,
            dropout=config.encoder_dropout,
        ).to(config.device)

        # ── actor-critic ─────────────────────────────────────────────────
        self.actor_critic = ActorCritic(
            d_model=config.encoder_d_model,
            hidden_dim=config.ac_hidden_dim,
        ).to(config.device)

        # ── PPO trainer ──────────────────────────────────────────────────
        ppo_config = PPOConfig(
            clip_epsilon=config.ppo_clip_epsilon,
            value_coef=config.ppo_value_coef,
            entropy_coef=config.ppo_entropy_coef,
            ppo_epochs=config.ppo_epochs,
            mini_batch_size=config.ppo_mini_batch_size,
            gamma=config.ppo_gamma,
            gae_lambda=config.ppo_gae_lambda,
            max_grad_norm=config.ppo_max_grad_norm,
            target_kl=config.ppo_target_kl,
        )
        self.ppo_trainer = PPOTrainer(
            encoder=self.encoder,
            actor_critic=self.actor_critic,
            config=ppo_config,
            device=config.device,
            learning_rate=config.ppo_lr,
        )

        # ── rollout buffer ───────────────────────────────────────────────
        self.rollout_buffer = RolloutBuffer(seq_len=256)

        # ── RL proof engine ──────────────────────────────────────────────
        self.rl_engine = RLProofEngine(
            encoder=self.encoder,
            actor_critic=self.actor_critic,
            knowledge_base=self.kb.get_all_statements(),
            max_steps=config.max_proof_steps,
            device=config.device,
        )

        # ── heuristic proof engine (fallback / warm-up) ──────────────────
        self.heuristic_engine = ProofEngine(
            max_depth=config.heuristic_max_depth,
            max_iterations=config.heuristic_max_iter,
            knowledge_base=self.kb.get_all_statements(),
        )

        # ── neural generator ─────────────────────────────────────────────
        self.gen_model = TransformerGenerator(
            vocab_size=self.tokenizer.vocab_size,
            d_model=config.gen_d_model,
            nhead=config.gen_nhead,
            num_layers=config.gen_num_layers,
        ).to(config.device)

        self.neural_gen = NeuralConjectureGenerator(
            model=self.gen_model,
            tokenizer=self.tokenizer,
            device=config.device,
            temperature=1.0,
            top_k=50,
        )

        self.random_gen = RandomConjectureGenerator(
            min_complexity=config.initial_complexity,
            max_complexity=config.final_complexity,
            var_names=["x", "y", "z", "w"],
            seed=config.random_seed,
        )

        gen_trainer_cfg = GeneratorTrainingConfig(
            learning_rate=config.gen_lr,
            batch_size=config.gen_batch_size,
            warmup_steps=config.gen_warmup_steps,
            device=config.device,
        )
        self.gen_trainer = GeneratorTrainer(
            model=self.gen_model,
            tokenizer=self.tokenizer,
            config=gen_trainer_cfg,
            knowledge_base=self.kb,
        )

        # ── curriculum ───────────────────────────────────────────────────
        if config.use_curriculum:
            self.curriculum = CurriculumScheduler(
                CurriculumConfig(
                    initial_complexity=config.initial_complexity,
                    final_complexity=config.final_complexity,
                    success_threshold=config.success_threshold,
                )
            )
        else:
            self.curriculum = None

        # ── filters ──────────────────────────────────────────────────────
        self.novelty_scorer = NoveltyScorer()
        self.complexity_est = ComplexityEstimator()
        self.diversity_filter = DiversityFilter(max_similar=20)

        # ── counters ─────────────────────────────────────────────────────
        self.total_generated = 0
        self.total_attempted = 0
        self.total_proved_rl = 0
        self.total_proved_heuristic = 0
        self.current_epoch = 0
        self.current_cycle = 0
        self.ppo_update_count = 0
        self._conjectures_since_update = 0

        # Log model sizes
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        ac_params = sum(p.numel() for p in self.actor_critic.parameters())
        gen_params = sum(p.numel() for p in self.gen_model.parameters())
        self._log(
            f"  Parameters: encoder={enc_params:,}  actor_critic={ac_params:,}  "
            f"generator={gen_params:,}"
        )

    # ── public entry point ────────────────────────────────────────────────

    def train(self):
        """Run full Phase 3 training."""
        self._log("\n" + "=" * 70)
        self._log("Phase 3: RL-Based Prover Training (PPO)")
        self._log("=" * 70)

        # Pre-train generator on existing KB
        if self.kb.total_size() > 0 and self.config.gen_pretrain_epochs > 0:
            self._log("\n[Pre-training generator on existing knowledge base...]")
            self.gen_trainer.train_on_knowledge_base(
                num_epochs=self.config.gen_pretrain_epochs,
                curriculum_strategy="complexity",
            )
            self._log("  Pre-training done.")

        # Behavioral cloning warm-up: use heuristic prover to initialize policy
        if self.config.bc_warmup_cycles > 0 and self.kb.size() > 0:
            self._behavioral_cloning_warmup()

        # Main training loop
        start_time = time.time()
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            epoch_rl_proofs = 0
            epoch_heuristic_proofs = 0
            self.novelty_scorer.reset()
            self.diversity_filter.reset()

            self._log(f"\n{'=' * 70}")
            self._log(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            self._log(f"{'=' * 70}")

            for cycle in range(self.config.cycles_per_epoch):
                self.current_cycle = cycle

                rl_p, h_p = self._run_cycle()
                epoch_rl_proofs += rl_p
                epoch_heuristic_proofs += h_p

                # Curriculum advance
                if self.curriculum and (rl_p + h_p) > 0:
                    if self.curriculum.should_advance_stage():
                        self.curriculum.advance_stage()
                        self.neural_gen.set_temperature(
                            self.curriculum.get_current_temperature()
                        )

                # Online generator update
                if (cycle + 1) % self.config.gen_update_interval == 0:
                    self._update_generator()

                # Logging
                if cycle % self.config.log_interval == 0:
                    self._log_progress(epoch, cycle, epoch_rl_proofs, epoch_heuristic_proofs)

                # Checkpoint
                if (cycle + 1) % self.config.checkpoint_interval == 0:
                    self._checkpoint()

            epoch_time = time.time() - epoch_start
            self._log(
                f"\nEpoch {epoch + 1} done in {epoch_time:.1f}s | "
                f"RL proofs: {epoch_rl_proofs} | "
                f"Heuristic proofs: {epoch_heuristic_proofs} | "
                f"KB size: {self.kb.size()}"
            )

        total_time = time.time() - start_time
        self._log("\n" + "=" * 70)
        self._log(f"Training complete in {total_time:.1f}s")
        self._log(f"Total RL proofs: {self.total_proved_rl}")
        self._log(f"Total heuristic proofs: {self.total_proved_heuristic}")
        self._log(f"Total PPO updates: {self.ppo_update_count}")
        self._log(f"Final KB size: {self.kb.size()}")
        self._log("=" * 70)
        self._checkpoint()

    # ── internal helpers ─────────────────────────────────────────────────

    def _run_cycle(self) -> tuple:
        """
        One generate-prove-learn cycle.
        Returns (rl_proofs_found, heuristic_proofs_found).
        """
        # ── generate conjectures ─────────────────────────────────────────
        self.neural_gen.eval_mode()
        conjectures = []

        n_neural = int(self.config.conjectures_per_cycle * self.config.neural_ratio)
        n_random = self.config.conjectures_per_cycle - n_neural

        neural_conjs = [c for c in self.neural_gen.generate(n_neural) if c is not None]
        conjectures.extend(neural_conjs)
        conjectures.extend(self.random_gen.generate(n_random))
        self.total_generated += len(conjectures)

        # ── curriculum filter ────────────────────────────────────────────
        if self.curriculum:
            conjectures = self.curriculum.filter_by_complexity(conjectures)

        # ── quality filter ───────────────────────────────────────────────
        filtered = []
        for c in conjectures:
            if not self.complexity_est.is_well_formed(c):
                continue
            if self.novelty_scorer.score(c) < 0.3:
                continue
            if not self.diversity_filter.should_keep(c):
                continue
            if self.kb.contains(c):
                continue
            filtered.append(c)

        # ── sync KB with engines ─────────────────────────────────────────
        kb_stmts = self.kb.get_all_statements()
        self.rl_engine.update_knowledge_base(kb_stmts)
        self.heuristic_engine.knowledge_base = kb_stmts

        rl_proofs = 0
        heuristic_proofs = 0

        for conj in filtered:
            self.total_attempted += 1
            proved = False

            # ── RL prover ─────────────────────────────────────────────────
            in_warmup = (self.ppo_update_count == 0 and
                         self._conjectures_since_update < self.config.bc_warmup_cycles)

            if not in_warmup:
                proof, traj = self.rl_engine.prove(
                    conj, greedy=False, collect_trajectory=True
                )

                if traj:
                    for step in traj:
                        self.rollout_buffer.add(
                            Transition(
                                state_tokens=step["state_tokens"],
                                attn_mask=step["attn_mask"],
                                action=step["action"],
                                reward=step["reward"],
                                done=step["done"],
                                log_prob=step["log_prob"],
                                value=step["value"],
                                goal_complexity=step["goal_complexity"],
                            )
                        )
                    self.rollout_buffer.mark_episode_end()

                if proof.result == ProofResult.SUCCESS:
                    rl_proofs += 1
                    self.total_proved_rl += 1
                    proved = True
                    self._add_to_kb(conj, proof)

            # ── heuristic fallback (or sole prover during warmup) ─────────
            if not proved and self.config.use_heuristic_fallback:
                proof_h = self.heuristic_engine.prove(conj)
                if proof_h.result == ProofResult.SUCCESS:
                    heuristic_proofs += 1
                    self.total_proved_heuristic += 1
                    proved = True
                    self._add_to_kb(conj, proof_h)

                    # During warmup, also generate synthetic RL trajectories
                    # from successful heuristic proofs using greedy rollout
                    if in_warmup:
                        self._collect_greedy_trajectory(conj)

            if self.curriculum:
                self.curriculum.record_result(proved, conj.complexity())
            if proved:
                self.novelty_scorer.add(conj)

            # ── PPO update trigger ────────────────────────────────────────
            self._conjectures_since_update += 1
            if (self._conjectures_since_update >= self.config.update_interval
                    and self.rollout_buffer.size() >= self.config.ppo_mini_batch_size):
                ppo_stats = self.ppo_trainer.update(self.rollout_buffer)
                self.rollout_buffer.clear()
                self._conjectures_since_update = 0
                self.ppo_update_count += 1

                if self.ppo_update_count % 5 == 0:
                    self._log(
                        f"  [PPO #{self.ppo_update_count}] "
                        f"pol_loss={ppo_stats.get('policy_loss', 0):.4f}  "
                        f"val_loss={ppo_stats.get('value_loss', 0):.4f}  "
                        f"entropy={ppo_stats.get('entropy', 0):.4f}  "
                        f"kl={ppo_stats.get('approx_kl', 0):.4f}"
                    )

        return rl_proofs, heuristic_proofs

    def _collect_greedy_trajectory(self, goal):
        """
        Run greedy RL rollout on a known-provable goal to collect
        imitation data for warm-up. The trajectory is added to the buffer.
        """
        try:
            proof, traj = self.rl_engine.prove(
                goal, greedy=True, collect_trajectory=True
            )
            if traj:
                for step in traj:
                    self.rollout_buffer.add(
                        Transition(
                            state_tokens=step["state_tokens"],
                            attn_mask=step["attn_mask"],
                            action=step["action"],
                            reward=step["reward"],
                            done=step["done"],
                            log_prob=step["log_prob"],
                            value=step["value"],
                            goal_complexity=step["goal_complexity"],
                        )
                    )
                self.rollout_buffer.mark_episode_end()
        except Exception:
            pass

    def _behavioral_cloning_warmup(self):
        """
        Warm up the policy on existing KB theorems using greedy rollouts.
        This bootstraps the policy before the main RL loop.
        """
        self._log(
            f"\n[Behavioral cloning warm-up on {self.kb.size()} existing theorems...]"
        )
        theorems = self.kb.get_all_theorems()
        warmup_goals = [t.statement for t in theorems[: self.config.bc_warmup_cycles]]

        kb_stmts = self.kb.get_all_statements()
        self.rl_engine.update_knowledge_base(kb_stmts)
        self.heuristic_engine.knowledge_base = kb_stmts

        proved = 0
        for goal in warmup_goals:
            self._collect_greedy_trajectory(goal)
            proved += 1

            if self.rollout_buffer.size() >= self.config.ppo_mini_batch_size * 4:
                ppo_stats = self.ppo_trainer.update(self.rollout_buffer)
                self.rollout_buffer.clear()
                self.ppo_update_count += 1

        # Final update on remaining transitions
        if self.rollout_buffer.size() >= self.config.ppo_mini_batch_size:
            self.ppo_trainer.update(self.rollout_buffer)
            self.rollout_buffer.clear()
            self.ppo_update_count += 1

        self._log(f"  Warm-up done. Processed {proved} goals. PPO updates: {self.ppo_update_count}")

    def _add_to_kb(self, statement, proof):
        """Add a proven statement to the knowledge base."""
        complexity = self.complexity_est.estimate(statement)
        self.kb.add_theorem(
            statement=statement,
            proof=proof,
            complexity=complexity,
            epoch=self.current_epoch,
            cycle=self.current_cycle,
        )

    def _update_generator(self):
        """Online update of the neural generator on recent proofs."""
        recent_count = min(50, self.kb.size())
        if recent_count < 10:
            return
        recent_theorems = self.kb.get_all_theorems()[-recent_count:]
        exprs = [t.statement for t in recent_theorems]
        self.neural_gen.train_mode()
        loss = self.gen_trainer._train_batch(exprs)
        self.neural_gen.eval_mode()
        self._log(f"  [Generator update] loss={loss:.4f}")

    def _log_progress(self, epoch, cycle, epoch_rl, epoch_heuristic):
        rl_rate = self.total_proved_rl / max(self.total_attempted, 1)
        h_rate = self.total_proved_heuristic / max(self.total_attempted, 1)
        msg = (
            f"Epoch {epoch + 1}, Cycle {cycle}/{self.config.cycles_per_epoch}: "
            f"KB={self.kb.size()}  "
            f"RL_rate={rl_rate:.2%}  H_rate={h_rate:.2%}  "
            f"epoch_rl={epoch_rl}  epoch_h={epoch_heuristic}  "
            f"ppo_updates={self.ppo_update_count}"
        )
        if self.curriculum:
            stats = self.curriculum.get_statistics()
            msg += f"  complexity={stats['current_complexity']}"
        self._log(msg)

    def _checkpoint(self):
        """Save all model checkpoints and KB."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        prefix = f"{self.config.checkpoint_dir}/{self.config.experiment_name}"
        tag = f"epoch_{self.current_epoch}_cycle_{self.current_cycle}"

        # Knowledge base
        self.kb.save(f"{prefix}_kb_{tag}.json")

        # RL model
        self.ppo_trainer.save(f"{prefix}_rl_{tag}.pt")

        # Generator
        self.neural_gen.save(f"{prefix}_gen_{tag}.pt")
        self.gen_trainer.save_checkpoint(f"{prefix}_gen_trainer_{tag}.pt")

        # Stats summary
        stats = {
            "epoch": self.current_epoch,
            "cycle": self.current_cycle,
            "total_generated": self.total_generated,
            "total_attempted": self.total_attempted,
            "total_proved_rl": self.total_proved_rl,
            "total_proved_heuristic": self.total_proved_heuristic,
            "ppo_updates": self.ppo_update_count,
            "kb_size": self.kb.size(),
        }
        with open(f"{prefix}_stats_{tag}.json", "w") as f:
            json.dump(stats, f, indent=2)

        self._log(f"  Checkpoint saved: {tag}")

    def _load_kb_from_checkpoint(self, path: str):
        """Load Phase 2 knowledge base for warm-starting."""
        self._log(f"  Loading KB from {path}...")
        try:
            with open(path) as f:
                data = json.load(f)
            loaded = 0
            index_only = 0
            for thm_data in data.get("theorems", []):
                stmt_str = thm_data.get("statement", "")
                if not stmt_str or stmt_str in self.kb.statement_index:
                    continue
                try:
                    stmt_expr = parse_expression(stmt_str)
                    complexity = float(thm_data.get("complexity", 10.0))
                    epoch = int(thm_data.get("epoch", 0))
                    cycle = int(thm_data.get("cycle", 0))
                    stub_proof = Proof(
                        statement=stmt_expr,
                        steps=[],
                        result=ProofResult.SUCCESS,
                    )
                    self.kb.add_theorem(stmt_expr, stub_proof, complexity, epoch, cycle)
                    loaded += 1
                except Exception:
                    # Fall back to index-only if parsing fails
                    self.kb.statement_index.add(stmt_str)
                    index_only += 1
            self._log(
                f"  Loaded {loaded} theorems from KB checkpoint"
                f" ({index_only} index-only fallbacks)."
            )
        except Exception as e:
            self._log(f"  Warning: failed to load KB checkpoint: {e}")

    def _log(self, msg: str):
        if self.logger:
            self.logger.log(msg)
        else:
            print(msg, flush=True)

    def get_statistics(self) -> dict:
        rl_rate = self.total_proved_rl / max(self.total_attempted, 1)
        h_rate = self.total_proved_heuristic / max(self.total_attempted, 1)
        return {
            "epoch": self.current_epoch,
            "cycle": self.current_cycle,
            "total_generated": self.total_generated,
            "total_attempted": self.total_attempted,
            "total_proved_rl": self.total_proved_rl,
            "total_proved_heuristic": self.total_proved_heuristic,
            "rl_success_rate": rl_rate,
            "heuristic_success_rate": h_rate,
            "ppo_update_count": self.ppo_update_count,
            "kb_size": self.kb.size(),
        }
