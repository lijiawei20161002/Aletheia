"""
Phase 3: RL-Based Prover for AutoConjecture.

Implements PPO-trained policy network for tactic selection,
replacing the heuristic best-first search from Phase 2.
"""
from .state_encoder import StateEncoder
from .actor_critic import ActorCritic, NUM_ACTIONS, ACTION_NAMES
from .replay_buffer import RolloutBuffer, Transition
from .ppo_trainer import PPOTrainer, PPOConfig
