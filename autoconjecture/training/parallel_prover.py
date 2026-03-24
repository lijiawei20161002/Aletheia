"""
Parallel heuristic proof engine for Phase 5.

Uses concurrent.futures.ProcessPoolExecutor to prove multiple conjectures
simultaneously across CPU cores, with string-based serialization for
cross-process safety.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import List, Optional, Tuple

from ..prover.proof_engine import ProofEngine, ProofResult


# ── module-level worker (must be importable by child processes) ───────────────

def _prove_worker(args: Tuple) -> dict:
    """
    Worker function executed in a subprocess.

    Args:
        args: (expr_str, kb_strs, max_depth, max_iterations)

    Returns:
        dict with keys: expr_str, success, proof_length, steps_str
    """
    expr_str, kb_strs, max_depth, max_iterations = args

    # Import inside worker so forked processes don't inherit GPU tensors
    # Add project root to path (needed when spawning fresh processes)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from src.logic.parser import parse_expression
        from src.prover.proof_engine import ProofEngine, ProofResult

        goal = parse_expression(expr_str)
        kb = []
        for kb_str in kb_strs:
            try:
                kb.append(parse_expression(kb_str))
            except Exception:
                pass

        engine = ProofEngine(
            max_depth=max_depth,
            max_iterations=max_iterations,
            knowledge_base=kb,
        )
        proof = engine.prove(goal)

        return {
            "expr_str": expr_str,
            "success": proof.result == ProofResult.SUCCESS,
            "proof_length": proof.length(),
            "steps_str": [str(s) for s in proof.steps],
        }
    except Exception as e:
        return {
            "expr_str": expr_str,
            "success": False,
            "proof_length": 0,
            "steps_str": [],
            "error": str(e),
        }


# ── main class ────────────────────────────────────────────────────────────────

class ParallelHeuristicProver:
    """
    Proves a batch of conjectures in parallel using multiple CPU processes.

    Each worker runs an independent ProofEngine instance. Results are collected
    asynchronously so slow proofs do not block fast ones.
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_depth: int = 50,
        max_iterations: int = 500,
        timeout_per_proof: float = 30.0,
    ):
        """
        Args:
            max_workers: Number of parallel worker processes.
            max_depth: Max proof-search depth per worker.
            max_iterations: Max search iterations per worker.
            timeout_per_proof: Wall-clock seconds before a proof attempt is
                               abandoned (per conjecture).
        """
        self.max_workers = max_workers
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.timeout_per_proof = timeout_per_proof

    def prove_batch(
        self,
        expr_strs: List[str],
        kb_strs: List[str],
    ) -> List[dict]:
        """
        Attempt to prove a list of expressions in parallel.

        Args:
            expr_strs: String representations of expressions to prove.
            kb_strs: String representations of all known statements (KB).

        Returns:
            List of result dicts (same length as expr_strs), each with:
                - expr_str: str
                - success: bool
                - proof_length: int
                - steps_str: List[str]
        """
        if not expr_strs:
            return []

        worker_args = [
            (expr_str, kb_strs, self.max_depth, self.max_iterations)
            for expr_str in expr_strs
        ]

        results_by_expr: dict = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_expr = {
                executor.submit(_prove_worker, arg): arg[0]
                for arg in worker_args
            }

            for future in as_completed(
                future_to_expr, timeout=self.timeout_per_proof * len(expr_strs)
            ):
                expr_str = future_to_expr[future]
                try:
                    result = future.result(timeout=self.timeout_per_proof)
                except TimeoutError:
                    result = {
                        "expr_str": expr_str,
                        "success": False,
                        "proof_length": 0,
                        "steps_str": [],
                        "error": "timeout",
                    }
                except Exception as exc:
                    result = {
                        "expr_str": expr_str,
                        "success": False,
                        "proof_length": 0,
                        "steps_str": [],
                        "error": str(exc),
                    }
                results_by_expr[expr_str] = result

        # Preserve original order
        return [
            results_by_expr.get(
                expr_str,
                {
                    "expr_str": expr_str,
                    "success": False,
                    "proof_length": 0,
                    "steps_str": [],
                },
            )
            for expr_str in expr_strs
        ]

    def prove_batch_sequential(
        self,
        expr_strs: List[str],
        kb_strs: List[str],
    ) -> List[dict]:
        """
        Sequential fallback (single process) — used when max_workers == 1
        or multiprocessing is unavailable.
        """
        return [_prove_worker((es, kb_strs, self.max_depth, self.max_iterations))
                for es in expr_strs]
