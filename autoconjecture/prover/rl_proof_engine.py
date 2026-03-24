"""
RL-based proof engine for Phase 3.

Replaces the heuristic best-first search with a policy network that
selects tactics at each step. Provides a gym-like ProofEnvironment
and a RLProofEngine for collecting trajectories.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch

from ..logic.expressions import Expression
from ..prover.tactics import (
    ProofState,
    ReflexivityTactic,
    AssumptionTactic,
    SimplifyTactic,
    RewriteTactic,
)
from ..prover.proof_engine import Proof, ProofResult, ProofStep
from ..rl.state_encoder import StateEncoder
from ..rl.actor_critic import ActorCritic, NUM_ACTIONS

# Action index → tactic instance
_TACTICS = [
    ReflexivityTactic(),   # 0
    AssumptionTactic(),    # 1
    SimplifyTactic(),      # 2
    RewriteTactic(),       # 3  (tries all available rewrites, picks best)
]

# Reward shaping constants
REWARD_SUCCESS = 1.0
REWARD_STEP = -0.05          # Penalty per step (encourage short proofs)
REWARD_PROGRESS = 0.02       # Per unit of goal complexity decrease
REWARD_TIMEOUT = -0.5        # Penalty for hitting max_steps
REWARD_STUCK = -0.1          # Penalty for a tactic that changed nothing


@dataclass
class StepInfo:
    """Diagnostic information from one environment step."""
    tactic_name: str
    prev_complexity: float
    new_complexity: float
    changed: bool
    done: bool
    success: bool


class ProofEnvironment:
    """
    Single-conjecture proof environment.

    API:
        env.reset(goal, hypotheses) -> (state, tokens, attn_mask)
        env.step(action)            -> (state, tokens, attn_mask, reward, done, info)
    """

    def __init__(
        self,
        knowledge_base: List[Expression],
        max_steps: int = 30,
    ):
        self.knowledge_base = knowledge_base
        self.max_steps = max_steps

        self._state: Optional[ProofState] = None
        self._steps_taken: int = 0
        self._proof_steps: List[ProofStep] = []
        self._done: bool = False
        self._success: bool = False

    def reset(
        self,
        goal: Expression,
        extra_hypotheses: Optional[List[Expression]] = None,
    ) -> ProofState:
        """Start a new proof episode for the given goal."""
        hypotheses = list(self.knowledge_base)
        if extra_hypotheses:
            hypotheses = extra_hypotheses + hypotheses

        self._state = ProofState(goal=goal, hypotheses=hypotheses, depth=0)
        self._steps_taken = 0
        self._proof_steps = []
        self._done = False
        self._success = False
        return self._state

    def step(self, action: int) -> Tuple[ProofState, float, bool, StepInfo]:
        """
        Apply the selected tactic action.

        Returns:
            (new_state, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode already done. Call reset() first.")
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        tactic = _TACTICS[action]
        prev_state = self._state
        prev_complexity = float(prev_state.goal.complexity())

        # Apply tactic
        new_states = tactic.apply(prev_state, self.knowledge_base)

        reward = REWARD_STEP
        done = False
        success = False

        if len(new_states) == 0:
            # Proof complete (QED)!
            done = True
            success = True
            reward = REWARD_SUCCESS
            new_state = prev_state  # Final state (doesn't matter much)
            self._proof_steps.append(
                ProofStep(
                    tactic=tactic.name(),
                    state_before=prev_state,
                    state_after=None,
                )
            )
        else:
            # Pick new state: prefer minimum goal complexity
            new_state = min(new_states, key=lambda s: s.goal.complexity())
            new_complexity = float(new_state.goal.complexity())
            changed = str(new_state.goal) != str(prev_state.goal)

            if changed:
                # Progress reward: proportional to complexity decrease
                complexity_delta = prev_complexity - new_complexity
                if complexity_delta > 0:
                    reward += REWARD_PROGRESS * complexity_delta
                self._proof_steps.append(
                    ProofStep(
                        tactic=tactic.name(),
                        state_before=prev_state,
                        state_after=new_state,
                    )
                )
            else:
                # Tactic didn't change anything
                reward += REWARD_STUCK

        self._steps_taken += 1
        self._state = new_state
        self._success = success

        # Check step limit
        if not done and self._steps_taken >= self.max_steps:
            done = True
            reward += REWARD_TIMEOUT

        self._done = done

        info = StepInfo(
            tactic_name=tactic.name(),
            prev_complexity=prev_complexity,
            new_complexity=float(new_state.goal.complexity()),
            changed=(str(new_state.goal) != str(prev_state.goal)) if not success else True,
            done=done,
            success=success,
        )
        return new_state, reward, done, info

    def get_proof(self, original_goal: Expression) -> Proof:
        """Return the Proof object for the current episode."""
        result = ProofResult.SUCCESS if self._success else (
            ProofResult.TIMEOUT if self._steps_taken >= self.max_steps
            else ProofResult.FAILURE
        )
        return Proof(
            statement=original_goal,
            steps=self._proof_steps,
            result=result,
        )


class RLProofEngine:
    """
    Policy-guided proof engine.

    Uses a trained ActorCritic + StateEncoder to select tactics
    at each step, replacing the heuristic best-first search.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        actor_critic: ActorCritic,
        knowledge_base: List[Expression],
        max_steps: int = 30,
        device: str = "cuda",
    ):
        self.encoder = encoder
        self.actor_critic = actor_critic
        self.knowledge_base = knowledge_base
        self.max_steps = max_steps
        self.device = device

        self.env = ProofEnvironment(
            knowledge_base=knowledge_base,
            max_steps=max_steps,
        )

    def update_knowledge_base(self, kb: List[Expression]):
        """Update the knowledge base used by the environment."""
        self.knowledge_base = kb
        self.env.knowledge_base = kb

    def prove(
        self,
        goal: Expression,
        greedy: bool = False,
        collect_trajectory: bool = True,
    ) -> Tuple[Proof, Optional[List[dict]]]:
        """
        Attempt to prove goal using the policy.

        Args:
            goal:                Conjecture to prove
            greedy:              Use argmax action selection
            collect_trajectory:  Return (state, action, reward, done, log_prob, value) list

        Returns:
            proof:      Proof object (may be SUCCESS or FAILURE/TIMEOUT)
            trajectory: List of step dicts (if collect_trajectory=True), else None
        """
        state = self.env.reset(goal)
        trajectory = [] if collect_trajectory else None

        self.encoder.eval()
        self.actor_critic.eval()

        with torch.no_grad():
            while True:
                # Encode current state
                state_emb = self.encoder.encode_state(state, self.device)

                # Select action
                action_t, log_prob_t, value_t = self.actor_critic.get_action(
                    state_emb, deterministic=greedy
                )
                action = int(action_t.item())
                log_prob = float(log_prob_t.item())
                value = float(value_t.item())

                # Record state tokens for replay buffer BEFORE stepping
                if collect_trajectory:
                    tokens = self.encoder.tokenize_state(state)
                    from ..rl.state_encoder import MAX_SEQ_LEN
                    pad_len = MAX_SEQ_LEN - len(tokens)
                    attn = [True] * len(tokens) + [False] * pad_len
                    tokens_padded = tokens + [self.encoder.tokenizer.pad_id] * pad_len

                # Step environment
                new_state, reward, done, info = self.env.step(action)

                if collect_trajectory:
                    trajectory.append({
                        "state_tokens": tokens_padded,
                        "attn_mask": attn,
                        "action": action,
                        "reward": reward,
                        "done": done,
                        "log_prob": log_prob,
                        "value": value,
                        "goal_complexity": info.prev_complexity,
                    })

                state = new_state
                if done:
                    break

        proof = self.env.get_proof(goal)
        return proof, trajectory
