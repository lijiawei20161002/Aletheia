"""
Bridge layer between natural language reasoning and formal logic.

Two directions:
  1. NL → Expression: extract formalizable mathematical claims from text so
     AutoConjecture's ProofEngine can verify them.
  2. Proof → CoT: serialize a Proof's ProofStep list into a natural-language
     chain-of-thought string so CoTShield can audit it.

Deep-integration additions:
  3. extract_step_claims: split reasoning into steps and try to extract a
     formal claim from EACH step (not just the whole text), enabling per-step
     alignment between CoT and proof trace.
  4. align_proof_to_cot: find the best word-overlap alignment between proof
     steps and CoT steps, returning (proof_idx, cot_idx, score) triples.
     Used by DualLayerAuditor to build StepAlignment records.
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
    Expression = object          # type: ignore
    ParseError = Exception       # type: ignore


# ---------------------------------------------------------------------------
# NL → Expression
# ---------------------------------------------------------------------------

_NL_REWRITES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"for\s+all\s+(\w+)[,\s]+(.+)", re.IGNORECASE),         r"forall \1. \2"),
    (re.compile(r"∀\s*(\w+)\s*[.,]\s*(.+)"),                            r"forall \1. \2"),
    (re.compile(r"there\s+exists?\s+(\w+)\s+(?:such\s+that|s\.t\.)\s+(.+)", re.IGNORECASE),
                                                                          r"exists \1. \2"),
    (re.compile(r"∃\s*(\w+)\s*[.,]\s*(.+)"),                            r"exists \1. \2"),
    (re.compile(r'^["\'](.+)["\']$'),                                    r"\1"),
]

_CLAIM_SIGNALS = [
    r"we have\s+",      r"it follows that\s+", r"therefore[,\s]+",
    r"thus[,\s]+",      r"hence[,\s]+",         r"so\s+",
    r"clearly[,\s]+",   r"note that\s+",         r"observe that\s+",
    r"claim[:\s]+",     r"conjecture[:\s]+",
]
_CLAIM_RE = re.compile(
    r"(?:" + "|".join(_CLAIM_SIGNALS) + r")"
    r"((?:forall|exists|∀|∃|\w+\s*[+\-*=]|\w+\s*\+\s*\w+)[^.!?\n]*)",
    re.IGNORECASE,
)

# Splitter shared with CoTDivergenceDetector._split_reasoning_steps
_STEP_SPLIT_RE = re.compile(
    r'\n+|\d+\.|Step \d+:|First,|Second,|Third,|Finally,|Therefore,|So,|Thus,'
)


def _apply_rewrites(text: str) -> str:
    text = text.strip()
    for pattern, replacement in _NL_REWRITES:
        text = pattern.sub(replacement, text, count=1)
    return text.strip()


def extract_formal_claims(text: str) -> List[str]:
    """
    Extract candidate formal-logic strings from natural language text.
    Returns strings that may be parseable by AutoConjecture's parser.
    """
    candidates: List[str] = []

    for match in _CLAIM_RE.finditer(text):
        raw = match.group(1).strip()
        candidates.append(_apply_rewrites(raw))

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
    Returns None if unavailable or parsing fails.
    """
    if not _AUTOCONJECTURE_AVAILABLE:
        return None

    candidates = extract_formal_claims(text)
    candidates.append(_apply_rewrites(text))  # try raw text last

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


# ---------------------------------------------------------------------------
# Step-level claim extraction (deep integration)
# ---------------------------------------------------------------------------

def extract_step_claims(
    reasoning: str,
) -> List[Tuple[int, str, Optional[str]]]:
    """
    Split reasoning into steps and attempt to extract a formal claim from each.

    Returns
    -------
    List of (step_index, step_text, formal_claim_str_or_None).

    When AutoConjecture is available, each step is passed through nl_to_expression.
    Steps where parsing succeeds carry a formal_claim string for per-step
    verification; steps where it fails carry None.

    This enables CoTShield and the combinator to reason at step granularity
    rather than treating the entire reasoning chain as a monolith.
    """
    raw_steps = _STEP_SPLIT_RE.split(reasoning)
    steps = [s.strip() for s in raw_steps if s.strip()]

    result: List[Tuple[int, str, Optional[str]]] = []
    for i, step in enumerate(steps):
        formal_claim: Optional[str] = None
        if _AUTOCONJECTURE_AVAILABLE:
            expr = nl_to_expression(step)
            if expr is not None:
                formal_claim = str(expr)
        result.append((i, step, formal_claim))

    return result


# ---------------------------------------------------------------------------
# Proof-to-CoT alignment (deep integration)
# ---------------------------------------------------------------------------

def align_proof_to_cot(
    proof_steps: List[str],
    cot_steps: List[str],
) -> List[Tuple[int, int, float]]:
    """
    Find the best word-overlap alignment between proof steps and CoT steps.

    For each proof step, finds the CoT step with the highest Jaccard similarity
    (intersection over union of content words longer than 2 characters).

    Returns
    -------
    List of (proof_step_idx, best_cot_step_idx, alignment_score).
    alignment_score is in [0.0, 1.0]; 0.0 means no shared content words.
    best_cot_step_idx is -1 when cot_steps is empty.

    This is intentionally simple — cosine similarity over sentence embeddings
    would be more accurate, but this runs in pure Python with no dependencies
    and is good enough to identify obvious correspondences (e.g., a proof step
    that mentions "induction" will align to the CoT step that mentions it).
    """
    alignments: List[Tuple[int, int, float]] = []

    for pi, pstep in enumerate(proof_steps):
        p_terms = set(re.findall(r'\b\w{3,}\b', pstep.lower()))
        best_ci: int = -1
        best_score: float = 0.0

        for ci, cstep in enumerate(cot_steps):
            c_terms = set(re.findall(r'\b\w{3,}\b', cstep.lower()))
            union = p_terms | c_terms
            if not union:
                continue
            score = len(p_terms & c_terms) / len(union)
            if score > best_score:
                best_ci, best_score = ci, score

        alignments.append((pi, best_ci, best_score))

    return alignments
