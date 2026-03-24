"""
PPO (Proximal Policy Optimization) trainer for Phase 3.

Implements clipped surrogate objective with value function and entropy bonus.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim

from .state_encoder import StateEncoder
from .actor_critic import ActorCritic
from .replay_buffer import RolloutBuffer


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    target_kl: float = 0.05         # Early-stop if KL exceeds this
    value_clip: bool = True          # Clip value loss (PPO2 style)


class PPOTrainer:
    """
    PPO trainer that jointly updates the StateEncoder and ActorCritic.

    Implements the PPO-clip objective (Schulman et al., 2017):
      L = E[ min(r*A, clip(r, 1-ε, 1+ε)*A) ] - c1*VF_loss + c2*H[π]
    """

    def __init__(
        self,
        encoder: StateEncoder,
        actor_critic: ActorCritic,
        config: PPOConfig,
        device: str = "cuda",
        learning_rate: float = 3e-4,
    ):
        self.encoder = encoder
        self.actor_critic = actor_critic
        self.config = config
        self.device = device

        # Joint optimizer for encoder + actor_critic
        all_params = list(encoder.parameters()) + list(actor_critic.parameters())
        self.optimizer = optim.Adam(all_params, lr=learning_rate, eps=1e-5)

        self._total_updates = 0

    def update(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Run PPO update on the collected rollout.

        Args:
            rollout_buffer: Collected transitions

        Returns:
            Dictionary of loss statistics
        """
        if rollout_buffer.size() == 0:
            return {}

        # Compute returns and advantages via GAE
        returns, advantages = rollout_buffer.compute_returns_and_advantages(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "n_updates": 0,
        }

        self.encoder.train()
        self.actor_critic.train()

        for epoch in range(self.config.ppo_epochs):
            epoch_kl = 0.0
            n_batches = 0

            for batch in rollout_buffer.get_mini_batches(
                returns, advantages,
                mini_batch_size=self.config.mini_batch_size,
                device=self.device,
            ):
                # Re-encode states through encoder
                state_emb = self.encoder(
                    batch["state_tokens"], batch["attn_mask"]
                )  # (batch, d_model)

                # Evaluate current policy on old actions
                log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    state_emb, batch["actions"]
                )

                # Importance ratio
                ratio = torch.exp(log_probs - batch["old_log_probs"])

                # Clipped surrogate policy loss
                adv = batch["advantages"]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon,
                                    1.0 + self.config.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (optionally clipped)
                if self.config.value_clip:
                    v_pred_clipped = (
                        batch["old_values"]
                        + torch.clamp(
                            values - batch["old_values"],
                            -self.config.clip_epsilon,
                            self.config.clip_epsilon,
                        )
                    )
                    v_loss1 = (values - batch["returns"]).pow(2)
                    v_loss2 = (v_pred_clipped - batch["returns"]).pow(2)
                    value_loss = torch.max(v_loss1, v_loss2).mean()
                else:
                    value_loss = (values - batch["returns"]).pow(2).mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters())
                    + list(self.actor_critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                # Track statistics
                with torch.no_grad():
                    approx_kl = (
                        (batch["old_log_probs"] - log_probs).mean().item()
                    )
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.config.clip_epsilon)
                        .float()
                        .mean()
                        .item()
                    )

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["approx_kl"] += approx_kl
                stats["clip_fraction"] += clip_frac
                stats["n_updates"] += 1
                epoch_kl += approx_kl
                n_batches += 1

            # Early stopping on KL divergence
            if n_batches > 0 and epoch_kl / n_batches > self.config.target_kl:
                break

        self._total_updates += 1
        self.encoder.eval()
        self.actor_critic.eval()

        # Normalize by number of updates
        n = stats.pop("n_updates")
        if n > 0:
            for k in stats:
                stats[k] /= n
        stats["n_update_steps"] = n

        return stats

    def save(self, path: str):
        """Save encoder + actor-critic state dicts."""
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "actor_critic": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_updates": self._total_updates,
            },
            path,
        )

    def load(self, path: str):
        """Load encoder + actor-critic state dicts."""
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor_critic.load_state_dict(ckpt["actor_critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._total_updates = ckpt.get("total_updates", 0)
