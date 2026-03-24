"""
Transformer-based neural conjecture generator.
Uses a decoder-only transformer to generate logical expressions autoregressively.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    Adds sinusoidal position information to embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerGenerator(nn.Module):
    """
    Transformer-based generator for logical expressions.

    Architecture:
    - Token embedding layer
    - Positional encoding
    - Multiple transformer decoder layers
    - Output projection to vocabulary
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 128
    ):
        """
        Args:
            vocab_size: Size of token vocabulary
            d_model: Dimension of embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # (seq_len, batch, features)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            src: Source sequence (seq_len, batch_size) - can be empty for unconditional generation
            tgt: Target sequence (seq_len, batch_size)
            tgt_mask: Causal mask for target
            src_key_padding_mask: Padding mask for source
            tgt_key_padding_mask: Padding mask for target

        Returns:
            Output logits (seq_len, batch_size, vocab_size)
        """
        # Embed and encode target
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Create memory (for unconditional generation, use learned tokens)
        if src.size(0) == 0:
            # Unconditional generation - create dummy memory
            batch_size = tgt.size(1)
            memory = torch.zeros(1, batch_size, self.d_model, device=tgt.device)
        else:
            # Conditional generation
            src_emb = self.embedding(src) * math.sqrt(self.d_model)
            memory = self.pos_encoder(src_emb)

        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(0), tgt.device)

        # Transformer decoder
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        # Project to vocabulary
        logits = self.output_projection(output)

        return logits

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask for autoregressive generation.

        Args:
            sz: Sequence length
            device: Device to create mask on

        Returns:
            Causal mask (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Generate sequences autoregressively.

        Args:
            batch_size: Number of sequences to generate
            max_length: Maximum sequence length
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Keep tokens with cumulative probability <= p
            sos_token_id: Start of sequence token ID
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            device: Device to generate on

        Returns:
            Generated sequences (batch_size, max_length)
        """
        self.eval()

        # Initialize with SOS token
        generated = torch.full(
            (batch_size, 1),
            sos_token_id,
            dtype=torch.long,
            device=device
        )

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Empty source for unconditional generation
        src = torch.empty(0, batch_size, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            # Transpose for transformer (seq_len, batch)
            tgt = generated.transpose(0, 1)

            # Forward pass
            logits = self.forward(src, tgt)

            # Get logits for last position
            next_token_logits = logits[-1, :, :]  # (batch_size, vocab_size)

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                # Clamp k to vocab size
                vocab_size = next_token_logits.size(-1)
                k = min(top_k, vocab_size)
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Set to PAD if sequence already finished
            next_token[finished] = pad_token_id

            # Mark sequences that just finished
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences finished
            if finished.all():
                break

        return generated

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for sequence generation.

        Args:
            logits: Model output (seq_len, batch_size, vocab_size)
            targets: Target tokens (seq_len, batch_size)
            pad_token_id: Padding token ID (ignored in loss)

        Returns:
            Average loss
        """
        # Reshape for cross entropy
        # (seq_len * batch_size, vocab_size) and (seq_len * batch_size,)
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)

        # Compute loss (ignore padding)
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=pad_token_id,
            reduction='mean'
        )

        return loss

    def save_checkpoint(self, path: str, optimizer_state: Optional[dict] = None, epoch: int = 0):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'max_seq_len': self.max_seq_len,
            'epoch': epoch
        }
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)

        model = cls(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['d_model'],
            max_seq_len=checkpoint['max_seq_len']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model, checkpoint.get('optimizer_state_dict'), checkpoint.get('epoch', 0)
