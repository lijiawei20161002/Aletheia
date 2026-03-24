"""
State encoder for Phase 3 RL prover.

Encodes a ProofState (goal + hypotheses) into a fixed-size embedding
using a small transformer encoder with mean pooling.
"""
from __future__ import annotations
import math
from typing import List, Optional
import torch
import torch.nn as nn

from ..prover.tactics import ProofState
from ..models.tokenizer import ExpressionTokenizer
from ..logic.expressions import Expression

# Token limits per component
MAX_GOAL_TOKENS = 60
MAX_HYP_TOKENS = 20
MAX_HYPOTHESES_IN_STATE = 10  # How many hypotheses the policy can see
MAX_SEQ_LEN = 256


class StateEncoder(nn.Module):
    """
    Encodes a ProofState into a fixed-size embedding vector.

    Input: goal expression + (optional) hypotheses/KB theorems
    Output: d_model-dimensional vector

    Architecture:
    - Token embedding (vocab + SEP token)
    - Learned positional encoding
    - Transformer encoder (bidirectional attention)
    - Mean pooling over non-padded tokens
    """

    def __init__(
        self,
        tokenizer: ExpressionTokenizer,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.d_model = d_model

        # SEP token ID extends vocabulary by 1
        self.sep_id = tokenizer.vocab_size
        extended_vocab = tokenizer.vocab_size + 1

        self.embedding = nn.Embedding(
            extended_vocab, d_model, padding_idx=tokenizer.pad_id
        )
        self.scale = math.sqrt(d_model)
        self.pos_encoding = nn.Embedding(MAX_SEQ_LEN, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def tokenize_state(self, state: ProofState) -> List[int]:
        """
        Flatten ProofState into a token sequence:
          [SOS] goal_tokens [SEP] hyp1_tokens [SEP] hyp2_tokens ... [EOS]
        """
        tokens: List[int] = [self.tokenizer.sos_id]

        # Goal
        goal_toks = self.tokenizer.encode_expression(
            state.goal, add_special_tokens=False
        )
        tokens.extend(goal_toks[:MAX_GOAL_TOKENS])

        # Hypotheses: sort by complexity (simpler first = more likely useful)
        hypotheses = state.hypotheses
        if len(hypotheses) > MAX_HYPOTHESES_IN_STATE:
            try:
                hypotheses = sorted(hypotheses, key=lambda h: h.complexity())[
                    :MAX_HYPOTHESES_IN_STATE
                ]
            except Exception:
                hypotheses = hypotheses[:MAX_HYPOTHESES_IN_STATE]

        for hyp in hypotheses:
            tokens.append(self.sep_id)
            hyp_toks = self.tokenizer.encode_expression(
                hyp, add_special_tokens=False
            )
            tokens.extend(hyp_toks[:MAX_HYP_TOKENS])

        tokens.append(self.tokenizer.eos_id)

        # Truncate to max length
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[: MAX_SEQ_LEN - 1] + [self.tokenizer.eos_id]

        return tokens

    def forward(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            token_ids:      (batch, seq_len) int64
            attention_mask: (batch, seq_len) bool, True = valid token

        Returns:
            state_embedding: (batch, d_model)
        """
        seq_len = token_ids.size(1)

        tok_emb = self.embedding(token_ids) * self.scale
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        pos_emb = self.pos_encoding(positions)  # (1, seq_len, d_model)
        x = tok_emb + pos_emb

        # TransformerEncoder expects padding_mask: True = IGNORE
        padding_mask = ~attention_mask  # (batch, seq_len)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)

        # Mean pooling over valid positions
        mask_f = attention_mask.float().unsqueeze(-1)  # (batch, seq, 1)
        state_emb = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return state_emb  # (batch, d_model)

    def encode_state(self, state: ProofState, device: str) -> torch.Tensor:
        """Encode a single ProofState → (1, d_model)."""
        tokens = self.tokenize_state(state)
        pad_len = MAX_SEQ_LEN - len(tokens)
        attn = [True] * len(tokens) + [False] * pad_len
        tokens = tokens + [self.tokenizer.pad_id] * pad_len

        tok_t = torch.tensor([tokens], dtype=torch.long, device=device)
        attn_t = torch.tensor([attn], dtype=torch.bool, device=device)
        return self.forward(tok_t, attn_t)

    def encode_states_batch(
        self, states: List[ProofState], device: str
    ) -> torch.Tensor:
        """Encode a batch of ProofStates → (batch, d_model)."""
        all_tokens: List[List[int]] = []
        for s in states:
            all_tokens.append(self.tokenize_state(s))

        max_len = min(max(len(t) for t in all_tokens), MAX_SEQ_LEN)
        padded_tokens: List[List[int]] = []
        padded_masks: List[List[bool]] = []
        for tokens in all_tokens:
            tokens = tokens[:max_len]
            pad_len = max_len - len(tokens)
            padded_tokens.append(tokens + [self.tokenizer.pad_id] * pad_len)
            padded_masks.append([True] * len(tokens) + [False] * pad_len)

        tok_t = torch.tensor(padded_tokens, dtype=torch.long, device=device)
        attn_t = torch.tensor(padded_masks, dtype=torch.bool, device=device)
        return self.forward(tok_t, attn_t)
