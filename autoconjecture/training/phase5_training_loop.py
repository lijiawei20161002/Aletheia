"""
Phase 5 training loop: Optimization.

New features over Phase 3:
  - Parallel heuristic proving via ProcessPoolExecutor
  - Advanced curriculum strategies (self-paced or adaptive-band)
  - Prioritized experience buffer for targeted generator training
  - Optional torch.compile() for neural models (PyTorch >= 2.0)
  - Throughput / timing instrumentation
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
from ..rl.state_encoder import StateEncoder
from ..rl.actor_critic import ActorCritic
from ..rl.replay_buffer import RolloutBuffer, Transition
from ..rl.ppo_trainer import PPOTrainer, PPOConfig
from ..models.advanced_curriculum import (
    SelfPacedCurriculum, SelfPacedConfig,
    AdaptiveBandCurriculum, AdaptiveBandConfig,
    PrioritizedExperienceBuffer,
)
from .parallel_prover import ParallelHeuristicProver


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class Phase5Config:
    """Full configuration for Phase 5 (Optimization) training."""

    # ── training schedule ───────────────────────────────────────────────
    num_epochs: int = 20
    cycles_per_epoch: int = 500
    conjectures_per_cycle: int = 10
    random_seed: int = 42

    # ── generation ──────────────────────────────────────────────────────
    initial_complexity: int = 6
    final_complexity: int = 20
    neural_ratio: float = 0.5

    # ── curriculum strategy ─────────────────────────────────────────────
    # "self_paced" | "adaptive_band" | "linear" (Phase 3 style)
    curriculum_strategy: str = "self_paced"

    # SelfPacedConfig knobs
    spc_ema_alpha: float = 0.1
    spc_target_lo: float = 0.10
    spc_target_hi: float = 0.60
    spc_frontier_margin: int = 3

    # AdaptiveBandConfig knobs
    abc_initial_halfwidth: int = 2
    abc_advance_threshold: float = 0.40
    abc_retreat_threshold: float = 0.05
    abc_patience: int = 200

    # ── parallel heuristic prover ────────────────────────────────────────
    parallel_workers: int = 4          # 0 or 1 = sequential (safe for debugging)
    heuristic_max_depth: int = 50
    heuristic_max_iter: int = 500
    heuristic_timeout: float = 30.0   # seconds per proof

    # ── RL prover ───────────────────────────────────────────────────────
    max_proof_steps: int = 30
    ppo_update_interval: int = 50
    bc_warmup_cycles: int = 200
    use_heuristic_fallback: bool = True

    # ── state encoder ───────────────────────────────────────────────────
    encoder_d_model: int = 256
    encoder_nhead: int = 4
    encoder_num_layers: int = 3
    encoder_dropout: float = 0.1

    # ── actor-critic ─────────────────────────────────────────────────────
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

    # ── neural generator ─────────────────────────────────────────────────
    gen_d_model: int = 256
    gen_nhead: int = 8
    gen_num_layers: int = 6
    gen_lr: float = 1e-4
    gen_batch_size: int = 32
    gen_warmup_steps: int = 500
    gen_pretrain_epochs: int = 3
    gen_update_interval: int = 100

    # ── prioritized experience buffer ────────────────────────────────────
    exp_buffer_maxlen: int = 2000
    exp_buffer_target_rate: float = 0.30
    gen_use_prioritized_buffer: bool = True

    # ── torch.compile ────────────────────────────────────────────────────
    use_torch_compile: bool = False    # Requires PyTorch >= 2.0

    # ── device & paths ───────────────────────────────────────────────────
    device: str = "cuda"
    checkpoint_dir: str = "data/checkpoints"
    experiment_name: str = "phase5_opt"
    checkpoint_interval: int = 500
    log_interval: int = 50
    kb_checkpoint: Optional[str] = None


# ── Training Loop ─────────────────────────────────────────────────────────────

class Phase5TrainingLoop:
    """
    Phase 5: Optimization training loop.

    Extends the Phase 3 loop with:
      - Parallel heuristic proving
      - Advanced (self-paced / adaptive-band) curriculum
      - Prioritized experience replay for the generator
      - Optional torch.compile acceleration
    """

    def __init__(self, config: Phase5Config, logger=None):
        self.config = config
        self.logger = logger

        torch.manual_seed(config.random_seed)

        self._log(
            f"Phase 5 | device={config.device} | "
            f"curriculum={config.curriculum_strategy} | "
            f"parallel_workers={config.parallel_workers}"
        )

        # ── tokenizer ────────────────────────────────────────────────────
        self.tokenizer = ExpressionTokenizer(
            max_length=128, var_names=["x", "y", "z", "w"]
        )

        # ── knowledge base ────────────────────────────────────────────────
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

        # ── actor-critic ──────────────────────────────────────────────────
        self.actor_critic = ActorCritic(
            d_model=config.encoder_d_model,
            hidden_dim=config.ac_hidden_dim,
        ).to(config.device)

        # ── PPO trainer ───────────────────────────────────────────────────
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

        # ── rollout buffer ────────────────────────────────────────────────
        self.rollout_buffer = RolloutBuffer(seq_len=256)

        # ── RL proof engine ───────────────────────────────────────────────
        self.rl_engine = RLProofEngine(
            encoder=self.encoder,
            actor_critic=self.actor_critic,
            knowledge_base=self.kb.get_all_statements(),
            max_steps=config.max_proof_steps,
            device=config.device,
        )

        # ── parallel heuristic prover ─────────────────────────────────────
        self.parallel_prover = ParallelHeuristicProver(
            max_workers=max(1, config.parallel_workers),
            max_depth=config.heuristic_max_depth,
            max_iterations=config.heuristic_max_iter,
            timeout_per_proof=config.heuristic_timeout,
        )
        # Sequential fallback engine (used for RL warm-up trajectories)
        self.heuristic_engine = ProofEngine(
            max_depth=config.heuristic_max_depth,
            max_iterations=config.heuristic_max_iter,
            knowledge_base=self.kb.get_all_statements(),
        )

        # ── neural generator ──────────────────────────────────────────────
        self.gen_model = TransformerGenerator(
            vocab_size=self.tokenizer.vocab_size,
            d_model=config.gen_d_model,
            nhead=config.gen_nhead,
            num_layers=config.gen_num_layers,
        ).to(config.device)

        # Optional torch.compile (PyTorch 2.0+)
        if config.use_torch_compile and hasattr(torch, "compile"):
            try:
                self.gen_model = torch.compile(self.gen_model)
                self.encoder = torch.compile(self.encoder)
                self._log("  torch.compile() applied to generator and encoder.")
            except Exception as e:
                self._log(f"  torch.compile() failed (skipping): {e}")

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

        # ── advanced curriculum ───────────────────────────────────────────
        self.curriculum = self._build_curriculum()

        # ── prioritized experience buffer ─────────────────────────────────
        self.exp_buffer = PrioritizedExperienceBuffer(
            maxlen=config.exp_buffer_maxlen,
            target_success_rate=config.exp_buffer_target_rate,
        )

        # ── filters ───────────────────────────────────────────────────────
        self.novelty_scorer = NoveltyScorer()
        self.complexity_est = ComplexityEstimator()
        self.diversity_filter = DiversityFilter(max_similar=20)

        # ── counters / timing ─────────────────────────────────────────────
        self.total_generated = 0
        self.total_attempted = 0
        self.total_proved_rl = 0
        self.total_proved_heuristic = 0
        self.current_epoch = 0
        self.current_cycle = 0
        self.ppo_update_count = 0
        self._conjectures_since_ppo_update = 0
        self._prove_time_total = 0.0
        self._cycle_time_total = 0.0

        enc_p = sum(p.numel() for p in self.encoder.parameters())
        ac_p = sum(p.numel() for p in self.actor_critic.parameters())
        gen_p = sum(p.numel() for p in self.gen_model.parameters())
        self._log(
            f"  Parameters: encoder={enc_p:,}  actor_critic={ac_p:,}  "
            f"generator={gen_p:,}"
        )

    # ── public entry point ────────────────────────────────────────────────

    def train(self):
        """Run full Phase 5 training."""
        self._log("\n" + "=" * 70)
        self._log("Phase 5: Optimization Training")
        self._log("=" * 70)

        if self.kb.total_size() > 0 and self.config.gen_pretrain_epochs > 0:
            self._log("\n[Pre-training generator on existing KB...]")
            self.gen_trainer.train_on_knowledge_base(
                num_epochs=self.config.gen_pretrain_epochs,
                curriculum_strategy="complexity",
            )

        if self.config.bc_warmup_cycles > 0 and self.kb.size() > 0:
            self._behavioral_cloning_warmup()

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
                t0 = time.time()
                rl_p, h_p = self._run_cycle()
                self._cycle_time_total += time.time() - t0
                epoch_rl_proofs += rl_p
                epoch_heuristic_proofs += h_p

                if cycle % self.config.gen_update_interval == 0 and cycle > 0:
                    self._update_generator()

                if cycle % self.config.log_interval == 0:
                    self._log_progress(epoch, cycle, epoch_rl_proofs, epoch_heuristic_proofs)

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
        avg_cycle = self._cycle_time_total / max(
            self.config.num_epochs * self.config.cycles_per_epoch, 1
        )
        self._log(f"Avg cycle time: {avg_cycle*1000:.1f} ms")
        self._log("=" * 70)
        self._checkpoint()

    # ── cycle ─────────────────────────────────────────────────────────────

    def _run_cycle(self) -> tuple:
        """One generate-prove-learn cycle. Returns (rl_proofs, h_proofs)."""

        # ── generate ─────────────────────────────────────────────────────
        self.neural_gen.eval_mode()
        n_neural = int(self.config.conjectures_per_cycle * self.config.neural_ratio)
        n_random = self.config.conjectures_per_cycle - n_neural

        conjectures = []
        if n_neural > 0:
            conjectures.extend(c for c in self.neural_gen.generate(n_neural) if c is not None)
        if n_random > 0:
            conjectures.extend(self.random_gen.generate(n_random))
        self.total_generated += len(conjectures)

        # ── curriculum filter ─────────────────────────────────────────────
        conjectures = self._curriculum_filter(conjectures)

        # ── quality filter ────────────────────────────────────────────────
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

        # ── sync KB with engines ──────────────────────────────────────────
        kb_stmts = self.kb.get_all_statements()
        self.rl_engine.update_knowledge_base(kb_stmts)
        self.heuristic_engine.knowledge_base = kb_stmts
        kb_strs = [str(s) for s in kb_stmts]

        rl_proofs = 0
        heuristic_proofs = 0

        in_warmup = (self.ppo_update_count == 0 and
                     self._conjectures_since_ppo_update < self.config.bc_warmup_cycles)

        # ── RL proving pass (sequential, on GPU) ──────────────────────────
        rl_proved_set: set = set()
        if not in_warmup:
            for conj in filtered:
                self.total_attempted += 1
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
                    rl_proved_set.add(str(conj))
                    self._add_to_kb(conj, proof)

                self.exp_buffer.add(
                    str(conj),
                    self.complexity_est.estimate(conj),
                    proof.result == ProofResult.SUCCESS,
                    proof.length(),
                )
                self._record_curriculum(proof.result == ProofResult.SUCCESS, conj)

        # ── parallel heuristic fallback ───────────────────────────────────
        if self.config.use_heuristic_fallback:
            fallback_conjs = [
                c for c in filtered
                if str(c) not in rl_proved_set and not self.kb.contains(c)
            ]
            if in_warmup:
                fallback_conjs = filtered  # Use all during warmup
                for conj in fallback_conjs:
                    self.total_attempted += 1

            if fallback_conjs:
                t_prove = time.time()
                results = self._parallel_heuristic_prove(fallback_conjs, kb_strs)
                self._prove_time_total += time.time() - t_prove

                for conj, res in zip(fallback_conjs, results):
                    if res["success"]:
                        if not self.kb.contains(conj):
                            # Reconstruct minimal Proof stub
                            stub_proof = Proof(
                                statement=conj,
                                steps=[],
                                result=ProofResult.SUCCESS,
                            )
                            self._add_to_kb(conj, stub_proof)
                            heuristic_proofs += 1
                            self.total_proved_heuristic += 1

                        if in_warmup:
                            self._collect_greedy_trajectory(conj)

                    if in_warmup:
                        self._record_curriculum(res["success"], conj)

                    self.exp_buffer.add(
                        str(conj),
                        self.complexity_est.estimate(conj),
                        res["success"],
                        res["proof_length"],
                    )

        # ── novelty update ────────────────────────────────────────────────
        for conj in filtered:
            if self.kb.contains(conj):
                self.novelty_scorer.add(conj)

        # ── PPO update ────────────────────────────────────────────────────
        self._conjectures_since_ppo_update += len(filtered)
        if (self._conjectures_since_ppo_update >= self.config.ppo_update_interval
                and self.rollout_buffer.size() >= self.config.ppo_mini_batch_size):
            ppo_stats = self.ppo_trainer.update(self.rollout_buffer)
            self.rollout_buffer.clear()
            self._conjectures_since_ppo_update = 0
            self.ppo_update_count += 1

            if self.ppo_update_count % 5 == 0:
                self._log(
                    f"  [PPO #{self.ppo_update_count}] "
                    f"pol={ppo_stats.get('policy_loss', 0):.4f}  "
                    f"val={ppo_stats.get('value_loss', 0):.4f}  "
                    f"ent={ppo_stats.get('entropy', 0):.4f}  "
                    f"kl={ppo_stats.get('approx_kl', 0):.4f}"
                )

        return rl_proofs, heuristic_proofs

    # ── helpers ────────────────────────────────────────────────────────────

    def _parallel_heuristic_prove(self, conjectures, kb_strs):
        """Route to parallel or sequential proving based on worker count."""
        expr_strs = [str(c) for c in conjectures]
        if self.config.parallel_workers <= 1:
            return self.parallel_prover.prove_batch_sequential(expr_strs, kb_strs)
        try:
            return self.parallel_prover.prove_batch(expr_strs, kb_strs)
        except Exception as e:
            self._log(f"  [Parallel prover error, falling back]: {e}")
            return self.parallel_prover.prove_batch_sequential(expr_strs, kb_strs)

    def _curriculum_filter(self, conjectures):
        """Apply the chosen curriculum strategy's complexity filter."""
        if self.curriculum is None:
            return conjectures
        if isinstance(self.curriculum, SelfPacedCurriculum):
            return self.curriculum.filter_by_complexity(conjectures)
        if isinstance(self.curriculum, AdaptiveBandCurriculum):
            return self.curriculum.filter_by_complexity(conjectures)
        return conjectures

    def _record_curriculum(self, success: bool, conj):
        """Feed a result back into the active curriculum."""
        if self.curriculum is None:
            return
        complexity = int(self.complexity_est.estimate(conj))
        self.curriculum.record_result(success, complexity)

    def _collect_greedy_trajectory(self, goal):
        """Greedy RL rollout for BC warm-up data collection."""
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
        """Warm up the policy on existing KB theorems via greedy rollouts."""
        self._log(
            f"\n[BC warm-up on {self.kb.size()} existing theorems...]"
        )
        theorems = self.kb.get_all_theorems()
        warmup_goals = [t.statement for t in theorems[: self.config.bc_warmup_cycles]]

        kb_stmts = self.kb.get_all_statements()
        self.rl_engine.update_knowledge_base(kb_stmts)
        self.heuristic_engine.knowledge_base = kb_stmts

        for goal in warmup_goals:
            self._collect_greedy_trajectory(goal)
            if self.rollout_buffer.size() >= self.config.ppo_mini_batch_size * 4:
                self.ppo_trainer.update(self.rollout_buffer)
                self.rollout_buffer.clear()
                self.ppo_update_count += 1

        if self.rollout_buffer.size() >= self.config.ppo_mini_batch_size:
            self.ppo_trainer.update(self.rollout_buffer)
            self.rollout_buffer.clear()
            self.ppo_update_count += 1

        self._log(
            f"  BC warm-up done. PPO updates: {self.ppo_update_count}"
        )

    def _add_to_kb(self, statement, proof):
        complexity = self.complexity_est.estimate(statement)
        self.kb.add_theorem(
            statement=statement,
            proof=proof,
            complexity=complexity,
            epoch=self.current_epoch,
            cycle=self.current_cycle,
        )

    def _update_generator(self):
        """Online generator update using prioritized experience buffer or recent KB."""
        if self.config.gen_use_prioritized_buffer and self.exp_buffer.size() >= 10:
            expr_strs = self.exp_buffer.successful_exprs(n=50)
            if len(expr_strs) < 5:
                return
            exprs = []
            for s in expr_strs:
                try:
                    exprs.append(parse_expression(s))
                except Exception:
                    pass
            if len(exprs) < 5:
                return
            self.neural_gen.train_mode()
            loss = self.gen_trainer._train_batch(exprs)
            self.neural_gen.eval_mode()
            self._log(f"  [Gen update (prioritized)] loss={loss:.4f}")
        else:
            # Fallback: recent KB theorems
            recent_count = min(50, self.kb.size())
            if recent_count < 5:
                return
            recent = self.kb.get_all_theorems()[-recent_count:]
            exprs = [t.statement for t in recent]
            self.neural_gen.train_mode()
            loss = self.gen_trainer._train_batch(exprs)
            self.neural_gen.eval_mode()
            self._log(f"  [Gen update (recent KB)] loss={loss:.4f}")

    def _log_progress(self, epoch, cycle, epoch_rl, epoch_heuristic):
        rl_rate = self.total_proved_rl / max(self.total_attempted, 1)
        h_rate = self.total_proved_heuristic / max(self.total_attempted, 1)
        avg_prove = self._prove_time_total / max(self.total_attempted, 1)

        curriculum_info = ""
        if self.curriculum is not None:
            stats = self.curriculum.get_statistics()
            if isinstance(self.curriculum, SelfPacedCurriculum):
                curriculum_info = (
                    f"  frontier={stats['frontier']} "
                    f"range={stats['complexity_range']}"
                )
            elif isinstance(self.curriculum, AdaptiveBandCurriculum):
                curriculum_info = (
                    f"  center={stats['center']:.1f} "
                    f"hw={stats['halfwidth']} "
                    f"range={stats['complexity_range']}"
                )

        self._log(
            f"Epoch {epoch + 1}, Cycle {cycle}/{self.config.cycles_per_epoch}: "
            f"KB={self.kb.size()}  "
            f"RL_rate={rl_rate:.2%}  H_rate={h_rate:.2%}  "
            f"epoch_rl={epoch_rl}  epoch_h={epoch_heuristic}  "
            f"ppo#={self.ppo_update_count}  "
            f"prove_ms={avg_prove*1000:.1f}"
            + curriculum_info
        )

    def _checkpoint(self):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        prefix = f"{self.config.checkpoint_dir}/{self.config.experiment_name}"
        tag = f"epoch_{self.current_epoch}_cycle_{self.current_cycle}"

        self.kb.save(f"{prefix}_kb_{tag}.json")
        self.ppo_trainer.save(f"{prefix}_rl_{tag}.pt")
        self.neural_gen.save(f"{prefix}_gen_{tag}.pt")
        self.gen_trainer.save_checkpoint(f"{prefix}_gen_trainer_{tag}.pt")

        stats = {
            "epoch": self.current_epoch,
            "cycle": self.current_cycle,
            "total_generated": self.total_generated,
            "total_attempted": self.total_attempted,
            "total_proved_rl": self.total_proved_rl,
            "total_proved_heuristic": self.total_proved_heuristic,
            "ppo_updates": self.ppo_update_count,
            "kb_size": self.kb.size(),
            "exp_buffer_size": self.exp_buffer.size(),
            "avg_prove_ms": (
                self._prove_time_total / max(self.total_attempted, 1) * 1000
            ),
        }
        with open(f"{prefix}_stats_{tag}.json", "w") as f:
            json.dump(stats, f, indent=2)

        self._log(f"  Checkpoint saved: {tag}")

    def _build_curriculum(self):
        strategy = self.config.curriculum_strategy
        if strategy == "self_paced":
            return SelfPacedCurriculum(
                SelfPacedConfig(
                    initial_complexity=self.config.initial_complexity,
                    final_complexity=self.config.final_complexity,
                    ema_alpha=self.config.spc_ema_alpha,
                    target_lo=self.config.spc_target_lo,
                    target_hi=self.config.spc_target_hi,
                    frontier_margin=self.config.spc_frontier_margin,
                )
            )
        elif strategy == "adaptive_band":
            return AdaptiveBandCurriculum(
                AdaptiveBandConfig(
                    initial_complexity=self.config.initial_complexity,
                    final_complexity=self.config.final_complexity,
                    initial_halfwidth=self.config.abc_initial_halfwidth,
                    advance_threshold=self.config.abc_advance_threshold,
                    retreat_threshold=self.config.abc_retreat_threshold,
                    patience=self.config.abc_patience,
                )
            )
        else:
            return None  # No curriculum filter (linear, Phase 3 style)

    def _load_kb_from_checkpoint(self, path: str):
        self._log(f"  Loading KB from {path}...")
        try:
            with open(path) as f:
                data = json.load(f)
            loaded = 0
            for thm_data in data.get("theorems", []):
                stmt_str = thm_data.get("statement", "")
                if not stmt_str or stmt_str in self.kb.statement_index:
                    continue
                try:
                    stmt_expr = parse_expression(stmt_str)
                    complexity = float(thm_data.get("complexity", 10.0))
                    stub_proof = Proof(
                        statement=stmt_expr, steps=[], result=ProofResult.SUCCESS
                    )
                    self.kb.add_theorem(
                        stmt_expr, stub_proof, complexity,
                        int(thm_data.get("epoch", 0)),
                        int(thm_data.get("cycle", 0)),
                    )
                    loaded += 1
                except Exception:
                    self.kb.statement_index.add(stmt_str)
            self._log(f"  Loaded {loaded} theorems from KB checkpoint.")
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
        stats = {
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
            "exp_buffer_size": self.exp_buffer.size(),
            "avg_prove_ms": (
                self._prove_time_total / max(self.total_attempted, 1) * 1000
            ),
        }
        if self.curriculum is not None:
            stats["curriculum"] = self.curriculum.get_statistics()
        return stats
