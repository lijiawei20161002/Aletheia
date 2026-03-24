"""
Bridge layer between natural language reasoning and formal logic.

Two directions:
  1. NL → Expression: extract formalizable mathematical claims from text so
     AutoConjecture's ProofEngine can verify them.
  2. Proof → CoT: serialize a Proof's ProofStep list into a natural-language
     chain-of-thought string so CoTShield can audit it.

The NL→Expression direction is inherently best-effort: most natural language
cannot be mechanically formalized. The bridge tries a battery of regex patterns
that cover the most common Peano-arithmetic phrasings and gracefully returns
None on failure, allowing the caller to fall back to UNVERIFIABLE.
"""

from __future__ import annotations
import re
from typing import Optional, List, Tuple

try:
    from autoconjecture.logic.parser import parse_expression, ParseError
    from autoconjecture.logic.expressions import Expression
    _AUTOCONJECTURE_AVAILABLE = True
except ImportError:
    _AUTOCONJECTURE_AVAILABLE = False
    Expression = object  # type: ignore
    ParseError = Exception  # type: ignore


# ---------------------------------------------------------------------------
# NL → Expression
# ---------------------------------------------------------------------------

# Patterns that rewrite common mathematical English into parser-compatible syntax.
# Order matters: more specific patterns before more general ones.
_NL_REWRITES: List[Tuple[re.Pattern, str]] = [
    # "for all x, x + 0 = x"  →  "forall x. x + 0 = x"
    (re.compile(r"for\s+all\s+(\w+)[,\s]+(.+)", re.IGNORECASE), r"forall \1. \2"),
    # "∀x. x + 0 = x" (unicode) → "forall x. x + 0 = x"
    (re.compile(r"∀\s*(\w+)\s*[.,]\s*(.+)"), r"forall \1. \2"),
    # "there exists x such that x + 0 = x" → "exists x. x + 0 = x"
    (re.compile(r"there\s+exists?\s+(\w+)\s+(?:such\s+that|s\.t\.)\s+(.+)", re.IGNORECASE), r"exists \1. \2"),
    # "∃x. ..." → "exists x. ..."
    (re.compile(r"∃\s*(\w+)\s*[.,]\s*(.+)"), r"exists \1. \2"),
    # Strip surrounding quotes / parentheses
    (re.compile(r'^["\'](.+)["\']$'), r"\1"),
]

# Phrases that signal a claim is being made, used to locate the claim itself
_CLAIM_SIGNALS = [
    r"we have\s+",
    r"it follows that\s+",
    r"therefore[,\s]+",
    r"thus[,\s]+",
    r"hence[,\s]+",
    r"so\s+",
    r"clearly[,\s]+",
    r"note that\s+",
    r"observe that\s+",
    r"claim[:\s]+",
    r"conjecture[:\s]+",
]
_CLAIM_RE = re.compile(
    r"(?:" + "|".join(_CLAIM_SIGNALS) + r")"
    r"((?:forall|exists|∀|∃|\w+\s*[+\-*=]|\w+\s*\+\s*\w+)[^.!?\n]*)",
    re.IGNORECASE,
)


def _apply_rewrites(text: str) -> str:
    """Apply NL-to-syntax rewrites in order."""
    text = text.strip()
    for pattern, replacement in _NL_REWRITES:
        text = pattern.sub(replacement, text, count=1)
    return text.strip()


def extract_formal_claims(text: str) -> List[str]:
    """
    Extract candidate formal-logic strings from natural language text.

    Returns a list of strings that may be parseable by AutoConjecture's parser.
    The list may be empty if no formalizable claims are found.
    """
    candidates: List[str] = []

    # Strategy 1: look for claim-signal phrases followed by an equation/formula
    for match in _CLAIM_RE.finditer(text):
        raw = match.group(1).strip()
        rewritten = _apply_rewrites(raw)
        candidates.append(rewritten)

    # Strategy 2: scan for bare equations like "x + 0 = x" or "S(x) = x + 1"
    bare_eq_re = re.compile(
        r"(?<![a-zA-Z])"
        r"((?:forall|exists|∀|∃)?\s*\w+\s*[.,\s]*"
        r"[\w\s+*()\[\]S]+=[^=\n]{1,60})"
    )
    for match in bare_eq_re.finditer(text):
        raw = match.group(1).strip()
        if "=" in raw:
            rewritten = _apply_rewrites(raw)
            if rewritten not in candidates:
                candidates.append(rewritten)

    return candidates


def nl_to_expression(text: str) -> Optional[Expression]:
    """
    Attempt to parse a natural language string into a formal Expression.

    Returns None if AutoConjecture is not available or parsing fails.
    This is intentionally graceful — callers should check for None.
    """
    if not _AUTOCONJECTURE_AVAILABLE:
        return None

    candidates = extract_formal_claims(text)
    # Also try the raw text directly
    candidates.append(_apply_rewrites(text))

    for candidate in candidates:
        try:
            expr = parse_expression(candidate)
            return expr
        except (ParseError, Exception):
            continue
    return None


# ---------------------------------------------------------------------------
# Proof → CoT
# ---------------------------------------------------------------------------

def proof_to_cot(proof) -> Tuple[str, str]:
    """
    Serialize an AutoConjecture Proof into (reasoning, output) strings
    that CoTShield's detector can analyze.

    Returns:
        reasoning: step-by-step proof trace as natural language
        output:    final statement ("Proved: ..." or "Failed to prove: ...")
    """
    if proof is None:
        return "", "No proof available."

    steps = []
    for i, step in enumerate(proof.steps, 1):
        if step.state_after is None:
            steps.append(f"Step {i}: Apply {step.tactic} → QED (proof complete)")
        else:
            steps.append(
                f"Step {i}: Apply {step.tactic} to goal '{step.state_before.goal}' "
                f"→ new goal '{step.state_after.goal}'"
            )

    reasoning = "\n".join(steps) if steps else "No proof steps recorded."

    result_val = proof.result.value if hasattr(proof.result, "value") else str(proof.result)
    if result_val == "success":
        output = f"Proved: {proof.statement}"
    else:
        output = f"Failed to prove: {proof.statement} (result: {result_val})"

    return reasoning, output
