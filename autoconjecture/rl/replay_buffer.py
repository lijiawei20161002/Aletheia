"""
Rollout buffer for on-policy PPO training.

Stores (state_tokens, attn_mask, action, reward, done, log_prob, value)
transitions collected during episode rollouts, then computes GAE advantages
before yielding mini-batches for PPO updates.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List, Tuple
import numpy as np
import torch


@dataclass
class Transition:
    """Single environment transition."""
    state_tokens: List[int]        # Flattened token IDs (variable length)
    attn_mask: List[bool]          # Attention mask (same length as tokens)
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float
    goal_complexity: float         # For curriculum / statistics


class RolloutBuffer:
    """
    On-policy rollout buffer for PPO.

    Collects full episodes, then computes returns and GAE advantages
    before yielding mini-batches for the PPO update.
    """

    def __init__(self, max_size: int = 10000, seq_len: int = 256):
        self.max_size = max_size
        self.seq_len = seq_len
        self.transitions: List[Transition] = []

        # Episode tracking for GAE computation
        self._episode_starts: List[int] = [0]  # Index where each episode starts

    def add(self, transition: Transition):
        """Add a transition from the current step."""
        self.transitions.append(transition)

    def mark_episode_end(self):
        """Mark the start of the next episode (called at episode end)."""
        self._episode_starts.append(len(self.transitions))

    def size(self) -> int:
        return len(self.transitions)

    def clear(self):
        self.transitions.clear()
        self._episode_starts = [0]

    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        final_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute discounted returns and GAE advantages.

        Args:
            gamma:      Discount factor
            gae_lambda: GAE lambda for variance reduction
            final_value: Bootstrap value at end of buffer (0 if terminal)

        Returns:
            returns:    (N,) discounted returns
            advantages: (N,) GAE advantages
        """
        n = len(self.transitions)
        if n == 0:
            return np.array([]), np.array([])

        rewards = np.array([t.reward for t in self.transitions], dtype=np.float32)
        dones = np.array([t.done for t in self.transitions], dtype=np.float32)
        values = np.array([t.value for t in self.transitions], dtype=np.float32)

        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        # Reverse scan for GAE
        for i in reversed(range(n)):
            if i == n - 1:
                next_value = final_value
                next_done = 1.0
            else:
                next_value = values[i + 1]
                next_done = dones[i]

            delta = rewards[i] + gamma * next_value * (1.0 - next_done) - values[i]
            last_gae = delta + gamma * gae_lambda * (1.0 - next_done) * last_gae
            advantages[i] = last_gae

        returns = advantages + values
        return returns, advantages

    def get_mini_batches(
        self,
        returns: np.ndarray,
        advantages: np.ndarray,
        mini_batch_size: int,
        device: str,
    ) -> Iterator[dict]:
        """
        Yield shuffled mini-batches for PPO update.

        Each mini-batch dict contains tensors:
          state_tokens, attn_mask, actions, old_log_probs,
          old_values, returns, advantages
        """
        n = len(self.transitions)
        indices = np.random.permutation(n)

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages_norm = (advantages - adv_mean) / adv_std

        for start in range(0, n, mini_batch_size):
            batch_idx = indices[start: start + mini_batch_size]
            if len(batch_idx) == 0:
                continue

            batch_tokens = []
            batch_masks = []
            for i in batch_idx:
                t = self.transitions[i]
                # Pad/truncate to seq_len
                tok = t.state_tokens
                msk = t.attn_mask
                if len(tok) < self.seq_len:
                    pad = self.seq_len - len(tok)
                    tok = tok + [0] * pad
                    msk = msk + [False] * pad
                else:
                    tok = tok[: self.seq_len]
                    msk = msk[: self.seq_len]
                batch_tokens.append(tok)
                batch_masks.append(msk)

            yield {
                "state_tokens": torch.tensor(
                    batch_tokens, dtype=torch.long, device=device
                ),
                "attn_mask": torch.tensor(
                    batch_masks, dtype=torch.bool, device=device
                ),
                "actions": torch.tensor(
                    [self.transitions[i].action for i in batch_idx],
                    dtype=torch.long,
                    device=device,
                ),
                "old_log_probs": torch.tensor(
                    [self.transitions[i].log_prob for i in batch_idx],
                    dtype=torch.float32,
                    device=device,
                ),
                "old_values": torch.tensor(
                    [self.transitions[i].value for i in batch_idx],
                    dtype=torch.float32,
                    device=device,
                ),
                "returns": torch.tensor(
                    returns[batch_idx], dtype=torch.float32, device=device
                ),
                "advantages": torch.tensor(
                    advantages_norm[batch_idx], dtype=torch.float32, device=device
                ),
            }
