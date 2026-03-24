"""
Neural conjecture generator using transformer models.
Replaces random generation with learned generation.
"""
import torch
from typing import List, Optional
from ..logic.expressions import Expression
from ..models.tokenizer import ExpressionTokenizer
from ..models.transformer_generator import TransformerGenerator


class NeuralConjectureGenerator:
    """
    Neural network-based conjecture generator.

    Uses a transformer model to generate logical expressions
    that are more likely to be provable, learned from successful proofs.
    """

    def __init__(
        self,
        model: Optional[TransformerGenerator] = None,
        tokenizer: Optional[ExpressionTokenizer] = None,
        device: str = "cpu",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: Optional[float] = None,
        max_length: int = 128,
        var_names: List[str] = None
    ):
        """
        Args:
            model: Pre-trained transformer model (if None, creates new one)
            tokenizer: Expression tokenizer (if None, creates new one)
            device: Device to run model on ('cpu' or 'cuda')
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            max_length: Maximum sequence length
            var_names: Variable names to use
        """
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_length = max_length
        self.var_names = var_names if var_names else ["x", "y", "z", "w"]

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = ExpressionTokenizer(
                max_length=max_length,
                var_names=self.var_names
            )
        else:
            self.tokenizer = tokenizer

        # Initialize model
        if model is None:
            self.model = TransformerGenerator(
                vocab_size=self.tokenizer.vocab_size,
                d_model=256,
                nhead=8,
                num_layers=6,
                max_seq_len=max_length
            ).to(device)
        else:
            self.model = model.to(device)

        self.model.eval()

    def generate(self, n: int = 1) -> List[Expression]:
        """
        Generate n conjectures using the neural model.

        Args:
            n: Number of conjectures to generate

        Returns:
            List of generated expressions
        """
        conjectures = []

        # Generate in batches for efficiency
        batch_size = min(n, 32)
        num_batches = (n + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in range(num_batches):
                current_batch_size = min(batch_size, n - batch_idx * batch_size)

                # Generate token sequences
                token_sequences = self.model.generate(
                    batch_size=current_batch_size,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    sos_token_id=self.tokenizer.sos_id,
                    eos_token_id=self.tokenizer.eos_id,
                    pad_token_id=self.tokenizer.pad_id,
                    device=self.device
                )

                # Decode to expressions
                for seq in token_sequences:
                    expr = self.tokenizer.decode_tokens(seq.cpu().tolist())
                    if expr is not None:
                        conjectures.append(expr)

        # If we didn't generate enough valid expressions, pad with None
        # (caller should filter out None values)
        while len(conjectures) < n:
            conjectures.append(None)

        return conjectures[:n]

    def set_temperature(self, temperature: float):
        """Update sampling temperature."""
        self.temperature = temperature

    def set_top_k(self, top_k: int):
        """Update top-k sampling parameter."""
        self.top_k = top_k

    def set_device(self, device: str):
        """Move model to different device."""
        self.device = device
        self.model = self.model.to(device)

    def save(self, path: str):
        """Save generator to file."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer_vocab': {
                'token_to_id': self.tokenizer.token_to_id,
                'id_to_token': self.tokenizer.id_to_token,
                'var_names': self.tokenizer.var_names,
                'max_length': self.tokenizer.max_length
            },
            'config': {
                'temperature': self.temperature,
                'top_k': self.top_k,
                'top_p': self.top_p,
                'max_length': self.max_length,
                'var_names': self.var_names
            }
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NeuralConjectureGenerator":
        """Load generator from file."""
        checkpoint = torch.load(path, map_location=device)

        # Recreate tokenizer
        tokenizer = ExpressionTokenizer(
            max_length=checkpoint['tokenizer_vocab']['max_length'],
            var_names=checkpoint['tokenizer_vocab']['var_names']
        )
        tokenizer.token_to_id = checkpoint['tokenizer_vocab']['token_to_id']
        tokenizer.id_to_token = checkpoint['tokenizer_vocab']['id_to_token']

        # Recreate model
        model = TransformerGenerator(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=tokenizer.max_length
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        # Create generator
        config = checkpoint['config']
        generator = cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=config['temperature'],
            top_k=config['top_k'],
            top_p=config.get('top_p'),
            max_length=config['max_length'],
            var_names=config['var_names']
        )

        return generator

    def train_mode(self):
        """Set model to training mode."""
        self.model.train()

    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()

    def get_model(self) -> TransformerGenerator:
        """Get underlying transformer model."""
        return self.model

    def get_tokenizer(self) -> ExpressionTokenizer:
        """Get tokenizer."""
        return self.tokenizer
