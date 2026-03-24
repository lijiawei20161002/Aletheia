"""
Tokenizer for logical expressions.
Converts expressions to token sequences for neural network processing.
"""
from typing import List, Dict, Optional, Tuple
from ..logic.terms import Term, Var, Zero, Succ, Add, Mul
from ..logic.expressions import Expression, Equation, Forall


class ExpressionTokenizer:
    """
    Tokenizes logical expressions for neural network input.

    Vocabulary:
    - Special tokens: <PAD>, <SOS>, <EOS>, <UNK>
    - Term constructors: VAR, ZERO, SUCC, ADD, MUL
    - Expression constructors: EQ, FORALL
    - Variable names: x, y, z, w, ...
    - Structural tokens: (, )
    """

    def __init__(self, max_length: int = 128, var_names: List[str] = None):
        """
        Args:
            max_length: Maximum sequence length
            var_names: List of allowed variable names
        """
        self.max_length = max_length
        self.var_names = var_names if var_names else ["x", "y", "z", "w", "v", "u"]

        # Build vocabulary
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Special tokens
        special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

        # Structural tokens
        structural_tokens = ["(", ")"]

        # Term constructors
        term_tokens = ["VAR", "ZERO", "SUCC", "ADD", "MUL"]

        # Expression constructors
        expr_tokens = ["EQ", "FORALL"]

        # Variable names
        var_tokens = [f"var_{name}" for name in self.var_names]

        # Build vocabulary
        all_tokens = special_tokens + structural_tokens + term_tokens + expr_tokens + var_tokens
        for idx, token in enumerate(all_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        # Special token IDs
        self.pad_id = self.token_to_id["<PAD>"]
        self.sos_id = self.token_to_id["<SOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]

        self.vocab_size = len(self.token_to_id)

    def encode_expression(self, expr: Expression, add_special_tokens: bool = True) -> List[int]:
        """
        Convert expression to token IDs.

        Args:
            expr: Expression to encode
            add_special_tokens: Whether to add <SOS> and <EOS>

        Returns:
            List of token IDs
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.sos_id)

        # Encode the expression recursively
        tokens.extend(self._encode_expr(expr))

        if add_special_tokens:
            tokens.append(self.eos_id)

        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length-1] + [self.eos_id]

        return tokens

    def _encode_expr(self, expr: Expression) -> List[int]:
        """Recursively encode an expression."""
        if isinstance(expr, Equation):
            return (
                [self.token_to_id["EQ"], self.token_to_id["("]] +
                self._encode_term(expr.left) +
                self._encode_term(expr.right) +
                [self.token_to_id[")"]]
            )
        elif isinstance(expr, Forall):
            var_token = f"var_{expr.var.name}"
            var_id = self.token_to_id.get(var_token, self.unk_id)
            return (
                [self.token_to_id["FORALL"], self.token_to_id["("], var_id] +
                self._encode_expr(expr.body) +
                [self.token_to_id[")"]]
            )
        else:
            # Unknown expression type
            return [self.unk_id]

    def _encode_term(self, term: Term) -> List[int]:
        """Recursively encode a term."""
        if isinstance(term, Var):
            var_token = f"var_{term.name}"
            var_id = self.token_to_id.get(var_token, self.unk_id)
            return [self.token_to_id["VAR"], var_id]

        elif isinstance(term, Zero):
            return [self.token_to_id["ZERO"]]

        elif isinstance(term, Succ):
            return (
                [self.token_to_id["SUCC"], self.token_to_id["("]] +
                self._encode_term(term.term) +
                [self.token_to_id[")"]]
            )

        elif isinstance(term, Add):
            return (
                [self.token_to_id["ADD"], self.token_to_id["("]] +
                self._encode_term(term.left) +
                self._encode_term(term.right) +
                [self.token_to_id[")"]]
            )

        elif isinstance(term, Mul):
            return (
                [self.token_to_id["MUL"], self.token_to_id["("]] +
                self._encode_term(term.left) +
                self._encode_term(term.right) +
                [self.token_to_id[")"]]
            )

        else:
            # Unknown term type
            return [self.unk_id]

    def decode_tokens(self, token_ids: List[int], skip_special: bool = True) -> Optional[Expression]:
        """
        Convert token IDs back to expression.

        Args:
            token_ids: List of token IDs
            skip_special: Whether to skip special tokens

        Returns:
            Reconstructed expression or None if invalid
        """
        # Remove special tokens if requested
        if skip_special:
            token_ids = [
                tid for tid in token_ids
                if tid not in [self.pad_id, self.sos_id, self.eos_id]
            ]

        if not token_ids:
            return None

        try:
            expr, _ = self._decode_expr(token_ids, 0)
            return expr
        except Exception:
            return None

    def _decode_expr(self, token_ids: List[int], pos: int) -> Tuple[Expression, int]:
        """
        Recursively decode an expression from token IDs.

        Returns:
            (expression, new_position)
        """
        if pos >= len(token_ids):
            raise ValueError("Unexpected end of tokens")

        token_id = token_ids[pos]
        token = self.id_to_token.get(token_id, "<UNK>")

        if token == "EQ":
            # Equation: EQ ( term term )
            if pos + 1 >= len(token_ids) or self.id_to_token[token_ids[pos + 1]] != "(":
                raise ValueError("Expected '(' after EQ")
            pos += 2  # Skip EQ and (

            left, pos = self._decode_term(token_ids, pos)
            right, pos = self._decode_term(token_ids, pos)

            if pos >= len(token_ids) or self.id_to_token[token_ids[pos]] != ")":
                raise ValueError("Expected ')' after equation terms")
            pos += 1  # Skip )

            return Equation(left, right), pos

        elif token == "FORALL":
            # Forall: FORALL ( var expr )
            if pos + 1 >= len(token_ids) or self.id_to_token[token_ids[pos + 1]] != "(":
                raise ValueError("Expected '(' after FORALL")
            pos += 2  # Skip FORALL and (

            if pos >= len(token_ids):
                raise ValueError("Expected variable after FORALL")

            var_token = self.id_to_token[token_ids[pos]]
            if not var_token.startswith("var_"):
                raise ValueError(f"Expected variable, got {var_token}")
            var_name = var_token[4:]  # Remove "var_" prefix
            var = Var(var_name)
            pos += 1

            body, pos = self._decode_expr(token_ids, pos)

            if pos >= len(token_ids) or self.id_to_token[token_ids[pos]] != ")":
                raise ValueError("Expected ')' after forall body")
            pos += 1

            return Forall(var, body), pos

        else:
            raise ValueError(f"Unexpected token: {token}")

    def _decode_term(self, token_ids: List[int], pos: int) -> Tuple[Term, int]:
        """
        Recursively decode a term from token IDs.

        Returns:
            (term, new_position)
        """
        if pos >= len(token_ids):
            raise ValueError("Unexpected end of tokens")

        token_id = token_ids[pos]
        token = self.id_to_token.get(token_id, "<UNK>")

        if token == "VAR":
            # Variable: VAR var_name
            if pos + 1 >= len(token_ids):
                raise ValueError("Expected variable name after VAR")
            var_token = self.id_to_token[token_ids[pos + 1]]
            if not var_token.startswith("var_"):
                raise ValueError(f"Expected variable, got {var_token}")
            var_name = var_token[4:]
            return Var(var_name), pos + 2

        elif token == "ZERO":
            return Zero(), pos + 1

        elif token == "SUCC":
            # Successor: SUCC ( term )
            if pos + 1 >= len(token_ids) or self.id_to_token[token_ids[pos + 1]] != "(":
                raise ValueError("Expected '(' after SUCC")
            pos += 2

            inner, pos = self._decode_term(token_ids, pos)

            if pos >= len(token_ids) or self.id_to_token[token_ids[pos]] != ")":
                raise ValueError("Expected ')' after successor term")
            pos += 1

            return Succ(inner), pos

        elif token == "ADD":
            # Addition: ADD ( term term )
            if pos + 1 >= len(token_ids) or self.id_to_token[token_ids[pos + 1]] != "(":
                raise ValueError("Expected '(' after ADD")
            pos += 2

            left, pos = self._decode_term(token_ids, pos)
            right, pos = self._decode_term(token_ids, pos)

            if pos >= len(token_ids) or self.id_to_token[token_ids[pos]] != ")":
                raise ValueError("Expected ')' after addition terms")
            pos += 1

            return Add(left, right), pos

        elif token == "MUL":
            # Multiplication: MUL ( term term )
            if pos + 1 >= len(token_ids) or self.id_to_token[token_ids[pos + 1]] != "(":
                raise ValueError("Expected '(' after MUL")
            pos += 2

            left, pos = self._decode_term(token_ids, pos)
            right, pos = self._decode_term(token_ids, pos)

            if pos >= len(token_ids) or self.id_to_token[token_ids[pos]] != ")":
                raise ValueError("Expected ')' after multiplication terms")
            pos += 1

            return Mul(left, right), pos

        else:
            raise ValueError(f"Unexpected token: {token}")

    def pad_sequence(self, token_ids: List[int], max_len: Optional[int] = None) -> List[int]:
        """
        Pad sequence to max_len with PAD tokens.

        Args:
            token_ids: Token IDs to pad
            max_len: Maximum length (defaults to self.max_length)

        Returns:
            Padded sequence
        """
        if max_len is None:
            max_len = self.max_length

        if len(token_ids) >= max_len:
            return token_ids[:max_len]

        return token_ids + [self.pad_id] * (max_len - len(token_ids))

    def batch_encode(self, expressions: List[Expression], pad: bool = True) -> List[List[int]]:
        """
        Encode a batch of expressions.

        Args:
            expressions: List of expressions
            pad: Whether to pad to same length

        Returns:
            List of token ID sequences
        """
        encoded = [self.encode_expression(expr) for expr in expressions]

        if pad:
            max_len = max(len(seq) for seq in encoded) if encoded else self.max_length
            max_len = min(max_len, self.max_length)
            encoded = [self.pad_sequence(seq, max_len) for seq in encoded]

        return encoded
