"""
Actor-Critic network for Phase 3 RL prover.

Maps state embeddings to:
  - Policy: probability distribution over discrete tactic actions
  - Value: scalar estimate of expected return from this state
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Discrete action set: 4 tactic choices
NUM_ACTIONS = 4
ACTION_NAMES = ["reflexivity", "assumption", "simplify", "rewrite"]


class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic network.

    Architecture:
      State embedding → shared MLP trunk
                      → actor head (logits over NUM_ACTIONS)
                      → critic head (scalar value)
    """

    def __init__(self, d_model: int = 256, hidden_dim: int = 256):
        super().__init__()

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Actor head: outputs logits for each tactic
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, NUM_ACTIONS),
        )

        # Critic head: outputs scalar state value
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Orthogonal initialization
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, state_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_emb: (batch, d_model)

        Returns:
            logits: (batch, NUM_ACTIONS)
            value:  (batch, 1)
        """
        features = self.trunk(state_emb)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def get_action(
        self,
        state_emb: torch.Tensor,
        deterministic: bool = False,
        action_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or greedily select) an action.

        Args:
            state_emb:    (1, d_model) or (batch, d_model)
            deterministic: use argmax instead of sampling
            action_mask:  (batch, NUM_ACTIONS) bool, True = action allowed

        Returns:
            action:   (batch,) int64
            log_prob: (batch,) float32
            value:    (batch,) float32
        """
        logits, value = self.forward(state_emb)

        if action_mask is not None:
            # Mask invalid actions with large negative logits
            logits = logits.masked_fill(~action_mask, float("-inf"))

        if deterministic:
            action = logits.argmax(dim=-1)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(
        self,
        state_emb: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions taken during rollout for PPO update.

        Args:
            state_emb: (batch, d_model)
            actions:   (batch,) int64
            action_mask: (batch, NUM_ACTIONS) bool or None

        Returns:
            log_probs: (batch,) float32
            values:    (batch,) float32
            entropy:   scalar float32
        """
        logits, value = self.forward(state_emb)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, value.squeeze(-1), entropy
