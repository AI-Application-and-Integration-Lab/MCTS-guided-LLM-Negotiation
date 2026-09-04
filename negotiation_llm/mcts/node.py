"""
MCTS Node class for open-loop Monte Carlo Tree Search in negotiation.
"""

import math
from typing import Dict, Any, List, Optional
import copy


class MCTSNode:
    """
    Represents a single state in the MCTS negotiation tree.

    In open-loop MCTS, each node represents a potential buyer offer,
    and the tree explores different negotiation strategies.
    """

    def __init__(
        self,
        action_id: Optional[int] = None,  # Bank action index (strategy/realization)
        seller_action: Optional[str] = None,  # "ask", "accept", "reject"
        parent: Optional['MCTSNode'] = None,
        prior_probability: float = 1.0,
        max_realizations: int = 3
    ):
        """
        Initialize an MCTS node representing a complete dialogue state.

        GDPZero-style architecture where each node represents:
        - A pool of possible negotiation histories (realizations)
        - The result of one buyer action (strategy + realization selection)
        - The immediate seller response to that action

        Args:
            action_id: Bank's action identifier (e.g., strategy index or realization index)
            seller_action: Customer's action type ("ask", "accept", "reject")
            parent: Parent node in the tree
            prior_probability: Prior probability P(s,a) for PUCT formula (default: 1.0)
            max_realizations: Maximum number of realizations to store per node
        """
        # Action and turn data
        self.action_id = action_id  # Which buyer action led to this node
        self.seller_action = seller_action  # Terminal detection uses this

        # Tree structure
        self.parent = parent
        self.children: List[MCTSNode] = []

        # MCTS statistics
        self.visit_count = 0
        self.total_reward = 0.0
        self.avg_reward = 0.0
        self.prior_probability = prior_probability  # P(s,a) for PUCT

        # Negotiation state
        self.is_terminal = False
        self.terminal_outcome: Optional[str] = None  # "accepted", "rejected", "max_rounds"

        # Depth in tree (for debugging/visualization)
        self.depth = parent.depth + 1 if parent else 0

        # Backward compatibility: keep strategy field for existing code
        self.strategy = None  # Set by _get_or_create_child() for DPO collection

        # Node-level realization storage (GDPZero)
        # Each realization is the full negotiation history up to this node
        self.realizations: List[List[Dict[str, Any]]] = []         # [history1, history2, ...]
        self.realization_values: List[float] = []                  # Value estimate per realization
        self.realization_visits: List[int] = []                    # Visit count per realization
        self.max_realizations: int = max_realizations              # Max pool size (k parameter)

    def select_best_action(self, exploration_weight: float = 1.4) -> int:
        """
        Select the best action using PUCT without requiring children to exist (GDPZero-style).

        This enables lazy child creation: we select the best action_id, then the caller
        creates the child for that action if it doesn't exist yet.

        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Where:
        - Q(s,a) = average action value (exploitation) - uses child stats if exists, 0 otherwise
        - P(s,a) = prior probability from stored strategy_priors
        - N(s) = parent visit count
        - N(s,a) = child visit count (0 if child doesn't exist yet)
        - c_puct = exploration constant

        Args:
            exploration_weight: Exploration constant (c_puct), typically ~1.4

        Returns:
            action_id: Index of best action to take
        """
        if not hasattr(self, 'available_strategies') or not self.available_strategies:
            raise ValueError("Node has not been expanded yet (no available_strategies)")

        if not hasattr(self, 'strategy_priors') or not self.strategy_priors:
            raise ValueError("Node has not been expanded yet (no strategy_priors)")

        num_actions = len(self.available_strategies)

        # Build mapping from action_id to child (if child exists)
        action_to_child = {}
        for child in self.children:
            if child.action_id is not None:
                action_to_child[child.action_id] = child

        best_score = float('-inf')
        best_action_id = 0

        for action_id in range(num_actions):
            strategy = self.available_strategies[action_id]

            # Get prior probability for this strategy
            prior = self.strategy_priors.get(strategy, 1.0 / num_actions)

            # Check if child exists for this action
            if action_id in action_to_child:
                child = action_to_child[action_id]

                if child.visit_count == 0:
                    # Unvisited child gets infinite priority
                    return action_id

                # Q(s,a) from existing child
                q_value = child.avg_reward
                visit_count = child.visit_count
            else:
                # No child exists yet - use prior only
                q_value = 0.25  # Neutral value for unvisited actions
                visit_count = 0

            # Exploration term: c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            
            exploration = exploration_weight * prior * math.sqrt(self.visit_count + 1) / (1 + visit_count)

            puct_score = q_value + exploration
            # print(puct_score, q_value, exploration)
            if puct_score > best_score:
                best_score = puct_score
                best_action_id = action_id

        return best_action_id

    def add_child(
        self,
        action_id: int,
        prior_probability: float = 1.0
    ) -> 'MCTSNode':
        """
        Add a child node representing a complete dialogue turn.

        GDPZero-style: Each child represents one buyer action followed by
        the immediate seller response.

        Args:
            action_id: Bank action identifier
            seller_action: Customer's action type ("ask", "accept", "reject")
            prior_probability: Prior probability P(s,a) for PUCT (default: 1.0)

        Returns:
            The newly created child node
        """
        child = MCTSNode(
            action_id=action_id,
            parent=self,
            prior_probability=prior_probability,
            max_realizations=self.max_realizations
        )

        self.children.append(child)
        return child

    def add_realization(
        self,
        negotiation_history: List[Dict[str, Any]],
        initial_value: float = 0.0
    ) -> int:
        """
        Add a new realization (negotiation history) to this node's pool.

        Args:
            negotiation_history: Complete negotiation history up to this node
            initial_value: Initial value estimate (default: 0.0)

        Returns:
            Index of the newly added realization
        """
        self.realizations.append(copy.deepcopy(negotiation_history))
        self.realization_values.append(initial_value)
        self.realization_visits.append(1)
        return len(self.realizations) - 1

    def sample_realization(self) -> Optional[tuple[int, List[Dict[str, Any]]]]:
        """
        Sample uniformly from this node's realization pool.

        Returns:
            Tuple of (index, negotiation_history) or None if pool is empty
        """
        if not self.realizations:
            return None

        import random
        idx = random.randint(0, len(self.realizations) - 1)
        return (idx, self.realizations[idx])

    def update_realization_value(self, idx: int, value: float) -> None:
        """
        Update realization value using incremental averaging.

        V_new = (N_old * V_old + value) / (N_old + 1)

        Args:
            idx: Realization index to update
            value: New reward value to incorporate
        """
        if idx < 0 or idx >= len(self.realizations):
            return

        old_N = self.realization_visits[idx]
        old_V = self.realization_values[idx]
        new_N = old_N + 1
        new_V = (old_N * old_V + value) / new_N

        self.realization_visits[idx] = new_N
        self.realization_values[idx] = new_V

    def get_best_realization(self) -> List[Dict[str, Any]]:
        """
        Get the highest-valued realization (negotiation history) from this node's pool.

        Returns:
            Negotiation history list for the best realization

        Raises:
            ValueError: If pool is empty
        """
        if not self.realizations:
            raise ValueError("Cannot get best realization from empty pool")

        best_idx = max(
            range(len(self.realization_values)),
            key=lambda i: self.realization_values[i]
        )
        return self.realizations[best_idx]

    def has_space_for_realization(self) -> bool:
        """
        Check if this node can accept more realizations.

        Returns:
            True if pool size < max_realizations
        """
        return len(self.realizations) < self.max_realizations

    def realization_exists(self, negotiation_history: List[Dict[str, Any]]) -> bool:
        """
        Check if a realization (negotiation history) already exists in this node's pool.

        Used for deduplication to avoid storing duplicate realizations.

        Args:
            negotiation_history: History to check

        Returns:
            True if history is already in the pool
        """
        # Convert to JSON string for comparison
        import json
        history_str = json.dumps(negotiation_history, sort_keys=True)
        for existing in self.realizations:
            existing_str = json.dumps(existing, sort_keys=True)
            if history_str == existing_str:
                return True
        return False

    def build_history_from_path(self) -> List[Dict[str, Any]]:
        """
        Build negotiation history by traversing from root to this node.

        This reconstructs the dialogue history by collecting all buyer offers
        and seller responses along the path from root to current node.
        This is useful for DPO data collection where we need the context at each node.

        Returns:
            List of dialogue exchanges (alternating buyer and seller turns)
        """
        history = []
        path = self.get_path_to_root()

        # Skip root (has no offer), add all other nodes' offers and seller responses
        for node in path[1:]:  # Skip root at path[0]
            # Add buyer's offer and dialogue
            if node.buyer_offer is not None and node.buyer_dialogue is not None:
                history.append({
                    "speaker": "buyer",
                    "offer": node.buyer_offer,
                    "dialogue": node.buyer_dialogue
                })

            # Add seller's response (if present)
            if node.seller_dialogue is not None:
                customer_entry = {
                    "speaker": "seller",
                    "dialogue": node.seller_dialogue
                }
                # Include seller offer if available
                if node.seller_offer is not None:
                    customer_entry["offer"] = node.seller_offer
                history.append(customer_entry)

        return history

    def get_path_to_root(self) -> List['MCTSNode']:
        """
        Get the path from this node to the root.

        Returns:
            List of nodes from root to this node
        """
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def get_best_action(self) -> 'MCTSNode':
        """
        Get the best action (child) based on visit count.

        After MCTS search, the best move is typically the most visited child,
        not the highest average reward (to be robust).

        Returns:
            Child node with most visits
        """
        if not self.children:
            raise ValueError("Cannot get best action from node with no children")

        return max(self.children, key=lambda c: (c.visit_count, c.avg_reward))

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"MCTSNode(depth={self.depth}, visits={self.visit_count}, "
            f"avg_reward={self.avg_reward:.3f}, children={len(self.children)}, "
            f"offer={self.offer})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert node to dictionary for serialization.

        Returns:
            Dictionary representation of the node
        """
        return {
            "offer": self.offer,
            "dialogue": self.dialogue,
            "visit_count": self.visit_count,
            "avg_reward": self.avg_reward,
            "total_reward": self.total_reward,
            "prior_probability": self.prior_probability,
            "depth": self.depth,
            "num_children": len(self.children),
            "is_terminal": self.is_terminal,
            "terminal_outcome": self.terminal_outcome
        }

    def get_dpo_preference_score(self) -> float:
        """
        Calculate preference score for DPO data generation.

        Uses visit_count × avg_reward as preference signal.
        Higher score = better action.

        Returns:
            Preference score
        """
        return self.visit_count * max(self.avg_reward, 0.0)
