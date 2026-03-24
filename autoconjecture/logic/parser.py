"""
Simple parser for logical expressions.
Supports parsing strings like:
- Terms: "0", "S(x)", "x + y", "x * y"
- Equations: "x + 0 = x"
- Quantifiers: "forall x. x + 0 = x", "exists y. y = S(x)"
- Logical: "P and Q", "P or Q", "not P", "P implies Q"
"""
import re
from typing import Union, Tuple
from .terms import Term, Var, Zero, Succ, Add, Mul
from .expressions import Expression, Equation, Forall, Exists, Implies, And, Or, Not


class ParseError(Exception):
    """Raised when parsing fails."""
    pass


class Parser:
    """Simple recursive descent parser for logical expressions."""

    def __init__(self, text: str):
        self.text = text.strip()
        self.pos = 0

    def peek(self) -> str:
        """Look at current character without consuming it."""
        if self.pos >= len(self.text):
            return ''
        return self.text[self.pos]

    def consume(self, expected: str = None) -> str:
        """Consume and return current character."""
        if self.pos >= len(self.text):
            raise ParseError(f"Unexpected end of input")
        char = self.text[self.pos]
        if expected and char != expected:
            raise ParseError(f"Expected '{expected}' but got '{char}' at position {self.pos}")
        self.pos += 1
        return char

    def skip_whitespace(self):
        """Skip whitespace characters."""
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def parse_identifier(self) -> str:
        """Parse an identifier (variable name or keyword)."""
        self.skip_whitespace()
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            self.pos += 1
        if start == self.pos:
            raise ParseError(f"Expected identifier at position {self.pos}")
        return self.text[start:self.pos]

    def parse_term(self) -> Term:
        """Parse a term."""
        self.skip_whitespace()

        # Check for parenthesized term (may be (t) or (t op t))
        if self.peek() == '(':
            self.consume('(')
            term = self.parse_term()
            self.skip_whitespace()
            if self.peek() == '+':
                self.consume('+')
                right = self.parse_term()
                self.skip_whitespace()
                self.consume(')')
                return Add(term, right)
            elif self.peek() == '*':
                self.consume('*')
                right = self.parse_term()
                self.skip_whitespace()
                self.consume(')')
                return Mul(term, right)
            self.consume(')')
            return term

        # Check for zero
        if self.peek() == '0':
            self.consume('0')
            return Zero()

        # Check for successor S(term)
        if self.peek() == 'S':
            self.consume('S')
            self.skip_whitespace()
            self.consume('(')
            inner = self.parse_term()
            self.skip_whitespace()
            self.consume(')')
            return Succ(inner)

        # Check for variable
        if self.peek().isalpha():
            name = self.parse_identifier()
            var = Var(name)

            # Check for operations
            self.skip_whitespace()
            if self.peek() == '+':
                self.consume('+')
                right = self.parse_term()
                return Add(var, right)
            elif self.peek() == '*':
                self.consume('*')
                right = self.parse_term()
                return Mul(var, right)

            return var

        raise ParseError(f"Unexpected character '{self.peek()}' at position {self.pos}")

    def parse_equation(self) -> Equation:
        """Parse an equation (term = term)."""
        left = self.parse_term()
        self.skip_whitespace()
        self.consume('=')
        right = self.parse_term()
        return Equation(left, right)

    def parse_expression(self) -> Expression:
        """Parse a logical expression."""
        self.skip_whitespace()

        # Check for quantifiers
        if self.text[self.pos:self.pos+6] == 'forall':
            self.pos += 6
            self.skip_whitespace()
            var_name = self.parse_identifier()
            var = Var(var_name)
            self.skip_whitespace()
            self.consume('.')
            body = self.parse_expression()
            return Forall(var, body)

        if self.text[self.pos:self.pos+6] == 'exists':
            self.pos += 6
            self.skip_whitespace()
            var_name = self.parse_identifier()
            var = Var(var_name)
            self.skip_whitespace()
            self.consume('.')
            body = self.parse_expression()
            return Exists(var, body)

        # Check for negation
        if self.text[self.pos:self.pos+3] == 'not':
            self.pos += 3
            inner = self.parse_expression()
            return Not(inner)

        # Check for parenthesized expression
        if self.peek() == '(':
            saved_pos = self.pos
            self.consume('(')
            # Try parsing as a parenthesized equation first (e.g. from KB checkpoints)
            try:
                expr = self.parse_equation()
                self.skip_whitespace()
                self.consume(')')
            except Exception:
                # Fall back to parenthesized sub-expression (e.g. logical connectives)
                self.pos = saved_pos
                self.consume('(')
                expr = self.parse_expression()
                self.skip_whitespace()
                self.consume(')')

            # Check for binary connectives
            self.skip_whitespace()
            if self.text[self.pos:self.pos+3] == 'and':
                self.pos += 3
                right = self.parse_expression()
                return And(expr, right)
            elif self.text[self.pos:self.pos+2] == 'or':
                self.pos += 2
                right = self.parse_expression()
                return Or(expr, right)
            elif self.text[self.pos:self.pos+7] == 'implies':
                self.pos += 7
                right = self.parse_expression()
                return Implies(expr, right)

            return expr

        # Otherwise, parse as equation
        return self.parse_equation()


def parse_term(text: str) -> Term:
    """Parse a term from string."""
    parser = Parser(text)
    try:
        result = parser.parse_term()
        parser.skip_whitespace()
        if parser.pos < len(parser.text):
            raise ParseError(f"Unexpected characters after term: '{parser.text[parser.pos:]}'")
        return result
    except Exception as e:
        raise ParseError(f"Failed to parse term '{text}': {e}")


def parse_expression(text: str) -> Expression:
    """Parse an expression from string."""
    parser = Parser(text)
    try:
        result = parser.parse_expression()
        parser.skip_whitespace()
        if parser.pos < len(parser.text):
            raise ParseError(f"Unexpected characters after expression: '{parser.text[parser.pos:]}'")
        return result
    except Exception as e:
        raise ParseError(f"Failed to parse expression '{text}': {e}")


def validate_syntax(text: str) -> Tuple[bool, str]:
    """
    Validate syntax of an expression.
    Returns (is_valid, error_message).
    """
    try:
        parse_expression(text)
        return True, ""
    except ParseError as e:
        return False, str(e)
