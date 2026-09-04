"""
Strategy-level MCTS negotiation agent.

Implements open-loop Monte Carlo Tree Search where the agent searches over
high-level negotiation strategies and an LLM realizes the chosen strategy
into a concrete utterance and offer.

All dataset-specific behaviour is supplied by a `NegotiationDomain`.
"""

import sys
import os

from typing import Dict, Any, List, Optional, Tuple, Literal
import copy
import json
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from negotiation_llm.models import BaseModel
from .node import MCTSNode
from .domain import NegotiationDomain


class MCTSNegotiator:
    """
    MCTS-based negotiation agent.

    Uses Monte Carlo Tree Search to explore negotiation strategies and select
    the action maximizing expected reward, as defined by the domain.
    """

    def __init__(
        self,
        model: BaseModel,
        domain_config: Dict[str, Any],
        domain: NegotiationDomain,
        exploration_weight: float = 2.5,
        num_simulations: int = 50,
        max_children: Optional[int] = None,  # None = use all strategies in the domain
        heuristic_samples: int = 3,
        use_realizations: bool = True,
        max_realizations: int = 3,
        use_response_selection: bool = True,
        collect_dpo_data: bool = False,
        dpo_output_dir: str = "dpo_data",
        lora_adapter: Optional[str] = None,
    ):
        """
        Initialize MCTS negotiator with strategy-based search.

        Args:
            model: LLM model for simulations
            domain_config: Domain configuration passed to strategy prompts
                (e.g. the item config for a bargaining domain)
            domain: NegotiationDomain supplying strategies, prompts, parsing
                and reward. Required — the search has no dataset knowledge.
            exploration_weight: PUCT exploration constant (c_puct)
            num_simulations: Number of MCTS simulations per move
            max_children: Cap on strategies to explore; None (the default)
                uses every strategy the domain offers. Setting it below the
                strategy count silently drops the tail of the list.
            heuristic_samples: Number of samples for acceptance heuristic (unused in current version)
            use_realizations: Enable realization caching (GDPZero-style)
            max_realizations: Number of offer variants to cache (k in GDPZero)
            use_response_selection: Select best realization after search
            collect_dpo_data: Enable automatic DPO data collection during negotiation
            dpo_output_dir: Directory to save DPO training data
            lora_adapter: LoRA adapter name to use for generation (if model supports it)
        """
        if domain is None:
            raise ValueError(
                "MCTSNegotiator requires a NegotiationDomain; the search core "
                "has no dataset-specific behaviour of its own."
            )
        self.model = model
        self.domain_config = domain_config
        self.domain = domain
        self.exploration_weight = exploration_weight
        self.num_simulations = num_simulations
        self.max_children = max_children
        self.heuristic_samples = heuristic_samples
        self.use_realizations = use_realizations
        self.max_realizations = max_realizations
        self.use_response_selection = use_response_selection
        self.collect_dpo_data = collect_dpo_data
        self.dpo_output_dir = Path(dpo_output_dir)
        self.lora_adapter = lora_adapter

        # DPO data collection buffers
        self.dpo_seller_utterances: List[Dict[str, Any]] = []

        # Create output directory if needed
        if self.collect_dpo_data:
            self.dpo_output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Domain dispatch helpers                                              #
    # ------------------------------------------------------------------ #

    def _get_strategies(self) -> List[Any]:
        return self.domain.get_strategies()

    def _get_strategy_prior(self, strategy: Any, context: Dict[str, Any] = None) -> float:
        return self.domain.get_strategy_prior(strategy, context)

    def _create_strategy_selection_prompt(
        self, agent_profile: Dict[str, Any], strategies: List[Any], history: List[Dict[str, Any]]
    ) -> str:
        return self.domain.create_strategy_selection_prompt(agent_profile, strategies, history)

    def _create_strategy_prompt(self, strategy: Any, agent_profile: Dict[str, Any],
                                 history: List[Dict[str, Any]],
                                 current_opponent_offer: Optional[Dict[str, Any]] = None,
                                 current_opponent_dialogue: Optional[str] = None,
                                 previous_agent_offer: Optional[Dict[str, Any]] = None) -> str:
        return self.domain.create_strategy_prompt(
            strategy, agent_profile, self.domain_config, history,
            current_opponent_offer, current_opponent_dialogue, previous_agent_offer
        )

    def _create_opponent_prompt(self, opponent_profile: Dict[str, Any], history: List[Dict[str, Any]],
                                 current_agent_offer: Optional[Dict[str, Any]] = None,
                                 current_agent_dialogue: Optional[str] = None) -> str:
        return self.domain.create_opponent_prompt(
            opponent_profile, history, current_agent_offer, current_agent_dialogue
        )

    def _create_opponent_action_prompt(self, opponent_profile: Dict[str, Any],
                                        history: List[Dict[str, Any]],
                                        current_agent_offer: Optional[Dict[str, Any]] = None,
                                        current_agent_dialogue: Optional[str] = None) -> str:
        return self.domain.create_opponent_action_prompt(
            opponent_profile, history, current_agent_offer, current_agent_dialogue
        )

    def _parse_agent_response(self, response: str) -> Dict[str, Any]:
        return self.domain.parse_agent_response(response)

    def _parse_opponent_response(self, response: str) -> Dict[str, Any]:
        return self.domain.parse_opponent_response(response)

    def _build_agent_offer(self, parsed: Dict[str, Any],
                            agent_profile: Dict[str, Any]) -> Dict[str, Any]:
        return self.domain.build_agent_offer(parsed, agent_profile)

    def _build_opponent_offer(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        return self.domain.build_opponent_offer(parsed)

    def _calculate_reward(self, final_terms: Dict[str, Any], agent_profile: Dict[str, Any],
                           outcome: str, num_rounds: int, max_rounds: int) -> float:
        return self.domain.calculate_reward(final_terms, agent_profile, outcome, num_rounds, max_rounds)

    def _parse_strategy_from_response(self, response: str, available_strategies: List[Any]) -> Any:
        return self.domain.parse_strategy_from_response(response, available_strategies)

    # ------------------------------------------------------------------ #

    def select_action(
        self,
        agent_profile: Dict[str, Any],
        negotiation_history: List[Dict[str, Any]],
        max_negotiation_rounds: int = 15
    ) -> Tuple[Dict[str, Any], str, MCTSNode]:
        """
        Use MCTS to select the best buyer offer.

        GDPZero-style: Run recursive search from root, then extract best action.

        Args:
            agent_profile: Customer information
            negotiation_history: Current negotiation history
            max_negotiation_rounds: Maximum rounds allowed

        Returns:
            Tuple of (buyer_offer, buyer_dialogue, root_node)
        """
        # Create root node with current state
        root = MCTSNode(
            action_id=None,  # Root has no action
            seller_action=None,
            parent=None,
            max_realizations=self.max_realizations,
            prior_probability=1.0
        )

        # Initialize root with current negotiation history
        root.add_realization(copy.deepcopy(negotiation_history), initial_value=0.0)

        # Run MCTS simulations using recursive search
        for _ in tqdm(range(self.num_simulations), desc="MCTS Simulations"):
            self._search(root, negotiation_history,agent_profile, max_negotiation_rounds)

        # Select best action (child with most visits)
        if not root.children:
            raise ValueError("No children generated from root - MCTS expansion failed")

        best_child = root.get_best_action()

        # Collect DPO data if enabled.
        #
        # Only buffer here. This used to also save and clear every turn, using
        # whatever profile it was handed as the label — which is the *agent's*
        # profile, not the opponent's, so the analyzer was labelled with the
        # buyer's own target price. It also meant a caller saving after the
        # episode wrote an already-emptied buffer. Labelling and file naming
        # belong to the caller, which knows which party the analyzer predicts;
        # see negotiation_llm/data/collect_dpo.py.
        if self.collect_dpo_data:
            self._collect_dpo_data_from_tree(
                root=root,
                agent_profile=agent_profile,
                negotiation_stage=len(negotiation_history)
            )

        # GDPZero Line 30: Response selection - u_sys* ← arg max_usys vh(H(s*))
        # Pick best realization from node's realization pool based on value estimates

        if self.use_response_selection and self.use_realizations:
            # Query node's realization pool for best utterance
            if best_child.realizations:
                try:
                    best_history = best_child.get_best_realization()
                    # Extract last buyer turn from best history
                    for turn in reversed(best_history):
                        if turn.get("speaker") == "buyer":
                            final_offer = turn.get("offer")
                            final_dialogue = turn.get("dialogue")
                            break
                except ValueError:
                    # Pool empty (shouldn't happen), use default
                    pass

        # Return the buyer offer from the best child (possibly with best realization selected)
        # (Customer response is already generated and stored in the child node)
        return final_offer, final_dialogue, root


    def _check_terminal_state(
        self,
        negotiation_history: List[Dict[str, Any]],
        current_round: int,
        max_rounds: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Lightweight terminal state detection without LLM calls (GDPZero-style).

        Checks negotiation history for terminal conditions by examining
        the action_type field that was already parsed from LLM responses.

        Args:
            negotiation_history: Complete negotiation history
            current_round: Current negotiation round
            max_rounds: Maximum allowed rounds

        Returns:
            Tuple of (is_terminal, outcome)
            - is_terminal: True if negotiation has ended
            - outcome: "accepted", "rejected", "max_rounds", or None
        """
        # Check max rounds
        if current_round >= max_rounds:
            return True, "max_rounds"

        if not negotiation_history:
            return False, None

        # Resolve terminal action types (domain-aware)
        terminal = self.domain.get_terminal_actions()
        accept_actions = set(terminal.get("accept_actions", ["accept"]))
        reject_actions = set(terminal.get("reject_actions", ["reject"]))

        # Check last seller (opponent) turn for explicit accept/reject
        seller_turns = [h for h in negotiation_history if h.get('speaker') == 'seller']
        if seller_turns:
            last_seller = seller_turns[-1]
            offer = last_seller.get('offer', {})
            action_type = offer.get('action_type')

            if action_type in accept_actions:
                return True, 'accepted'
            elif action_type in reject_actions:
                return True, 'rejected'

        # Check last agent turn — buyer "Agree" strategy counts as acceptance
        bank_turns = [h for h in negotiation_history if h.get('speaker') == 'buyer']
        if bank_turns:
            last_bank = bank_turns[-1]
            offer = last_bank.get('offer', {})
            if offer.get('action_type') in accept_actions:
                return True, 'accepted'

        return False, None

    def _is_terminal_cached(
        self,
        negotiation_history: List[Dict[str, Any]],
        current_round: int,
        max_rounds: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check terminal state with caching (GDPZero-style optimization).

        Caches terminal state results to avoid redundant checks during simulation.
        This significantly speeds up MCTS by eliminating repeated terminal detection
        for the same history states.

        Args:
            negotiation_history: Complete negotiation history
            current_round: Current negotiation round
            max_rounds: Maximum allowed rounds

        Returns:
            Tuple of (is_terminal, outcome)
        """
        
        is_terminal, outcome = self._check_terminal_state(negotiation_history, current_round, max_rounds)

        return is_terminal, outcome

    def _search(
        self,
        node: MCTSNode,
        negotiation_history: List[Dict[str, Any]],
        agent_profile: Dict[str, Any],
        max_rounds: int
    ) -> float:
        """
        Recursive MCTS search method (GDPZero-style).

        Combines selection, expansion, and backpropagation in one recursive method.
        On first visit to a node, expands it with available actions.
        On subsequent visits, uses UCT to select best child and recurse.

        Args:
            node: Current node to search from
            agent_profile: Customer information
            max_rounds: Maximum negotiation rounds

        Returns:
            Value estimate for this node
        """
        # Sample from node's realization pool at the start
        sampled_realization_idx = None
        sampled_realization = None
        node_history = []

        
        node_history = negotiation_history
        # Check if this is a terminal state
        is_terminal, outcome = self._is_terminal_cached(
            node_history,
            len(node_history)//2,
            max_rounds
        )

        if is_terminal and outcome == "accepted":
            # Terminal node: return actual reward
            node.is_terminal = True
            node.terminal_outcome = outcome

            # Extract final terms from node history
            final_offer = {}
            for turn in reversed(node_history):
                if turn.get('speaker') == 'buyer':
                    final_offer = turn.get('offer', {})
                    break

            # Calculate terminal reward (domain-aware)
            reward = self._calculate_reward(
                final_terms=final_offer,
                agent_profile=agent_profile,
                outcome=outcome,
                num_rounds=len(node_history)//2,
                max_rounds=max_rounds
            )

            # Update node statistics
            node.visit_count += 1
            node.total_reward += reward
            node.avg_reward = node.total_reward / node.visit_count

            return reward

        # Check if this node has been expanded (GDPZero-style)
        # A node is "unexpanded" if it hasn't been initialized with available strategies
        needs_expansion = not hasattr(node, 'available_strategies') or node.available_strategies is None

        if needs_expansion:
            # First visit: initialize node with available actions and priors (GDPZero-style _init_node)
            v = self._expand_node(node, agent_profile, max_rounds)

            # Update node statistics
            node.visit_count += 1
            node.total_reward += v
            node.avg_reward = node.total_reward / node.visit_count

            return v
        
        if self.use_realizations and node.realizations:
            # Sample from node's realization pool (full history)
            result = node.sample_realization()
            if result is not None:
                sampled_realization_idx, sampled_realization = result
                node_history = sampled_realization
        elif node.realizations:
            # Use first realization if no sampling
            node_history = node.realizations[0]
            
        # Node is expanded: select best action using PUCT and get/create child (GDPZero-style lazy creation)
        best_action_id = node.select_best_action(self.exploration_weight)
        best_child, next_negotiation_history, sampled_realization_idx = self._get_or_create_child(
            node,
            best_action_id,
            agent_profile,
            sampled_realization=node_history  # Pass sampled realization
        )

        # Recursive search
        v = self._search(best_child, next_negotiation_history, agent_profile, max_rounds)

        # Backpropagate (update this node's statistics)
        node.visit_count += 1
        node.total_reward += v
        node.avg_reward = node.total_reward / node.visit_count

        # GDPZero Line 22: Update history-specific value vh(h_tr)
        # Update the sampled realization's value if applicable
        if self.use_realizations and sampled_realization_idx is not None:
            best_child.update_realization_value(sampled_realization_idx, v)

        return v

    def _expand_node(
        self,
        node: MCTSNode,
        agent_profile: Dict[str, Any],
        max_rounds: int
    ) -> float:
        """
        Initialize a leaf node (GDPZero-style _init_node).

        Unlike the old approach of creating all children immediately, this method:
        1. Determines available actions (strategies)
        2. Estimates prior probabilities via frequency sampling
        3. Estimates value for this state
        4. Stores priors in node for later child creation

        Children are created lazily during tree traversal via _get_or_create_child().

        Args:
            node: Leaf node to initialize
            agent_profile: Customer information
            max_rounds: Maximum negotiation rounds

        Returns:
            Value estimate for this node
        """
        # Get available strategies (domain-aware)
        available_strategies = self._get_strategies()

        # Use all strategies unless explicitly capped
        if self.max_children is not None:
            available_strategies = available_strategies[:self.max_children]
        node.available_strategies = available_strategies

        # Estimate logprob-based priors (more efficient than frequency sampling)
        strategy_priors = self._estimate_strategy_priors_logprobs(
            node=node,
            agent_profile=agent_profile,
            available_strategies=node.available_strategies,
            temperature=1.0  # Adjust to control exploration vs exploitation
        )

        # Store priors in node for later child creation
        node.strategy_priors = strategy_priors

        # Estimate value for this state (not per-child like before)
        # Use current node's state to estimate value
        index, negotiation_history = node.sample_realization()
        buyer_offer = {}
        value = 0
        if negotiation_history:
            for turn in reversed(negotiation_history):
                if turn.get('speaker') == 'buyer' and not buyer_offer:
                    buyer_offer = turn.get('offer', {})
                    break
            expected_profit_reward = self._calculate_reward(
                final_terms=buyer_offer,
                agent_profile=agent_profile,
                outcome="accepted",
                num_rounds=len(negotiation_history)//2,
                max_rounds=max_rounds,
            )
            acceptance_likelihood = self._estimate_acceptance_likelihood(
                buyer_offer=buyer_offer,
                negotiation_history=negotiation_history,
                agent_profile=agent_profile,
            )
            if expected_profit_reward > 0:
                value = acceptance_likelihood * expected_profit_reward
            else:
                value = expected_profit_reward
        return value

    def _get_or_create_child(
        self,
        node: MCTSNode,
        action_id: int,
        agent_profile: Dict[str, Any],
        sampled_realization: Optional[Tuple[Dict[str, Any], str]] = None
    ) -> MCTSNode:
        """
        Get existing child or create new child for action (lazy creation).

        This implements GDPZero-style lazy child creation where children
        are only generated when actually needed during tree traversal.

        Args:
            node: Parent node
            action_id: Action index (strategy index)
            agent_profile: Customer information
            sampled_realization: Optional pre-sampled realization from state pool

        Returns:
            Child node for this action
        """
        # Check if child already exists
        existing_child = None
        for child in node.children:
            if child.action_id == action_id:
                existing_child = child
                break

        # Extract or generate buyer offer/dialogue based on sampled realization
        strategy = node.available_strategies[action_id]

        buyer_offer, buyer_dialogue = None, None
        for turn in reversed(sampled_realization):
            if turn.get("speaker") == "buyer":
                buyer_offer = turn.get("offer")
                buyer_dialogue = turn.get("dialogue")
                break
        # Get prior probability from node's stored priors
        prior_prob = node.strategy_priors.get(strategy, 1.0 / len(node.available_strategies))

        # Create child node with complete turn
        if existing_child == None:
            existing_child = node.add_child(
                action_id=action_id,
                prior_probability=prior_prob
            )
            existing_child.strategy = strategy

        # If child has unfilled realization, generate new turn
       
        if self.use_realizations and existing_child.has_space_for_realization():
            # 1. Generate new turn based on sampled realization
            new_bank_offer, new_bank_dialogue = self._generate_offer_from_strategy_sync(
                negotiation_history=sampled_realization,
                strategy=strategy,
                agent_profile=agent_profile
            )

            # Generate seller response for new turn
            temp_history = copy.deepcopy(sampled_realization)
            temp_history.append({
                "speaker": "buyer",
                "offer": new_bank_offer,
                "dialogue": new_bank_dialogue
            })

            seller_prompt = self._create_opponent_prompt(
                agent_profile,
                temp_history,
                new_bank_offer,
                new_bank_dialogue
            )
            # Generate seller response with optional LoRA adapter
            if self.lora_adapter and hasattr(self.model, 'generate'):
                seller_response = self.model.generate(seller_prompt, lora_adapter=self.lora_adapter)
            else:
                seller_response = self.model.generate(seller_prompt)
            parsed_customer = self._parse_opponent_response(seller_response)

            seller_offer = self._build_opponent_offer(parsed_customer)
            seller_dialogue = parsed_customer.get("dialogue", seller_response)

            # 2. child_realization = sampled_realization + new_turn
            child_realization_history = copy.deepcopy(temp_history)
            child_realization_history.append({
                "speaker": "seller",
                "offer": seller_offer,
                "dialogue": seller_dialogue
            })

            # 3. Save child realization in child node
            if not existing_child.realization_exists(child_realization_history):
                existing_child.add_realization(child_realization_history, initial_value=0.0)

            return existing_child, child_realization_history, len(existing_child.realizations) - 1

        

        # Add the realization history to the child's pool
        index, realization = existing_child.sample_realization()

        # Set strategy field for backward compatibility (used by DPO collection)
        

        return existing_child, realization, index

    def _estimate_acceptance_likelihood(
        self,
        buyer_offer: Dict[str, Any],
        agent_profile: Dict[str, Any],
        negotiation_history: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """
        Use seller LLM to estimate acceptance likelihood of buyer offer.

        Uses log probability estimation over action tokens (accept/reject/ask)
        for a more accurate and efficient likelihood estimate.

        Args:
            buyer_offer: Bank's current offer
            agent_profile: Customer information and targets
            negotiation_history: Optional negotiation history for context

        Returns:
            Acceptance likelihood in [0, 1] range based on P(accept) / (P(accept) + P(reject) + P(ask))
        """
        for turn in reversed(negotiation_history):
            if turn.get('speaker') == 'buyer':
                buyer_offer = turn.get('offer')
                buyer_dialogue = turn.get('dialogue')
                break
        seller_prompt = self._create_opponent_prompt(
            agent_profile,
            negotiation_history,
            buyer_offer,
            buyer_dialogue
        )


        try:
            # Use batched probability estimation for action tokens
            if hasattr(self.model, 'estimate_phrases_probability_batch'):
                # Create prompt ending at "Action: " to estimate action probabilities
                action_prompt = self._create_opponent_action_prompt(
                    agent_profile,
                    negotiation_history,
                    buyer_offer,
                    buyer_dialogue
                )

                # Domain-aware action tokens (PFG: donate/continue/decline)
                terminal_info = self.domain.get_terminal_actions()
                action_phrases = terminal_info.get("action_tokens", [" accept", " reject", " ask"])

                # Estimate probabilities in single batched call
                batch_results = self.model.estimate_phrases_probability_batch(
                    prompt=action_prompt,
                    target_phrases=action_phrases,
                    use_chat_template=False,
                    logprobs=10
                )

                # get_terminal_actions documents the order as
                # [accept, reject, neutral...]. The first two were previously
                # read as [accept, neutral, reject]; the swap was inert because
                # all three are summed below, but the names lied.
                accept_prob = batch_results[action_phrases[0]]['probability']
                reject_prob = batch_results[action_phrases[1]]['probability']
                neutral_prob = sum(
                    batch_results[p]['probability'] for p in action_phrases[2:]
                )

                # Normalize to get acceptance likelihood
                total_prob = accept_prob + reject_prob + neutral_prob
                if total_prob > 0:
                    acceptance_likelihood = accept_prob / total_prob
                else:
                    acceptance_likelihood = 0.0
                return max(0.0, min(1.0, acceptance_likelihood))

            else:
                # Fallback to sampling-based approach if batched method not available
                num_samples = 10
                # Generate batch with optional LoRA adapter
                if self.lora_adapter and hasattr(self.model, 'generate_batch'):
                    responses = self.model.generate_batch(seller_prompt, num_samples=num_samples, lora_adapter=self.lora_adapter)
                else:
                    responses = self.model.generate_batch(seller_prompt, num_samples=num_samples)

                success = 0
                for response in responses:
                    result = self._parse_opponent_response(response)
                    if result.get("accepts") is True or result.get("action_type") == "accept":
                        success += 1

                avg_likelihood = success / num_samples
                return max(0.0, min(1.0, avg_likelihood))

        except Exception as e:
            # Fallback to heuristic if LLM fails
            print(f"Warning: LLM-based likelihood estimation failed ({e}), using heuristic fallback")
            return 0


    def _estimate_strategy_priors_logprobs(
        self,
        node: MCTSNode,
        agent_profile: Dict[str, Any],
        available_strategies: List[Any],
        temperature: float = 1.0
    ) -> Dict[Any, float]:
        """
        Estimate strategy prior probabilities using log probability estimation.

        Uses vLLM's logprobs feature to directly compute the probability of each
        strategy token given the negotiation context. This is more efficient and
        accurate than frequency-based sampling.

        Args:
            node: Current node being expanded
            agent_profile: Customer information
            available_strategies: List of strategies to consider
            temperature: Temperature for softmax normalization (default: 1.0)

        Returns:
            Dictionary mapping strategy → prior probability
        """
        # Create strategy selection prompt (domain-aware)
        node_history = node.realizations[0] if node.realizations else []
        selection_prompt = self._create_strategy_selection_prompt(
            agent_profile,
            available_strategies,
            node_history,
        )

        # Check if model supports probability estimation (VLLMModel)
        if not hasattr(self.model, 'estimate_phrase_probability'):
            # Fallback to static priors for models without logprobs support
            print("Warning: Model does not support logprobs, using static priors")
            return {
                strategy: self._get_strategy_prior(strategy, context={
                    "seller_profile": seller_profile,
                    "negotiation_stage": node.depth
                }) for strategy in available_strategies
            }

        try:
            # Estimate log probability for each strategy
            import math

            # Check if model supports batched probability estimation
            if hasattr(self.model, 'estimate_phrases_probability_batch'):
                # Use batched estimation (much faster - single vLLM call)
                strategy_texts = [f" [{strategy.value}]" for strategy in available_strategies]

                # Score under the actor adapter: the DPO-trained policy is what
                # the priors are meant to reflect. Scoring under the base model
                # makes the adapter unreachable from strategy selection.
                batch_results = self.model.estimate_phrases_probability_batch(
                    prompt=selection_prompt,
                    target_phrases=strategy_texts,
                    use_chat_template=False,  # Prompt already formatted
                    logprobs=10,
                    lora_adapter=self.lora_adapter,
                )

                # Extract logprobs from batch results
                strategy_logprobs = {}
                for strategy, strategy_text in zip(available_strategies, strategy_texts):
                    strategy_logprobs[strategy] = batch_results[strategy_text]['avg_logprob']
            else:
                # Fallback to sequential estimation
                strategy_logprobs = {}
                for strategy in available_strategies:
                    strategy_text = f" [{strategy.value}]"
                    result = self.model.estimate_phrase_probability(
                        prompt=selection_prompt,
                        target_phrase=strategy_text,
                        use_chat_template=False,
                        logprobs=10,
                        lora_adapter=self.lora_adapter,
                    )
                    strategy_logprobs[strategy] = result['avg_logprob']

            # Apply temperature scaling and convert to probabilities via softmax
            # P(strategy) = exp(logprob / T) / sum(exp(logprob_i / T))

            # Scale by temperature
            scaled_logprobs = {s: lp / temperature for s, lp in strategy_logprobs.items()}

            # Compute log-sum-exp for numerical stability
            max_logprob = max(scaled_logprobs.values())
            log_sum_exp = max_logprob + math.log(
                sum(math.exp(lp - max_logprob) for lp in scaled_logprobs.values())
            )

            # Compute normalized probabilities
            priors = {}
            for strategy, scaled_lp in scaled_logprobs.items():
                priors[strategy] = math.exp(scaled_lp - log_sum_exp)

            # Sanity check: probabilities should sum to ~1.0
            total_prob = sum(priors.values())
            if abs(total_prob - 1.0) > 0.01:
                print(f"Warning: Strategy priors sum to {total_prob:.4f}, renormalizing")
                priors = {s: p / total_prob for s, p in priors.items()}

            return priors

        except Exception as e:
            # Fallback to static priors if probability estimation fails
            print(f"Warning: Logprob-based prior estimation failed ({e}), using static priors")
            import traceback
            traceback.print_exc()
            return {
                strategy: self._get_strategy_prior(strategy, context={
                    "seller_profile": seller_profile,
                    "negotiation_stage": node.depth
                }) for strategy in available_strategies
            }

    def _generate_offer_from_strategy_sync(
        self,
        negotiation_history: List[Dict[str, Any]],
        strategy: Any,
        agent_profile: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generate buyer offer from strategy (synchronous version for expansion).

        Args:
            negotiation_history: Current negotiation history
            strategy: Negotiation strategy
            agent_profile: Customer information

        Returns:
            Tuple of (buyer_offer, buyer_dialogue)
        """
        # Extract current seller offer/dialogue from last seller turn
        current_customer_offer = None
        current_customer_dialogue = None
        for turn in reversed(negotiation_history):
            if turn.get('speaker') == 'seller':
                current_customer_offer = turn.get('offer')
                current_customer_dialogue = turn.get('dialogue')
                break

        # Extract previous buyer offer
        previous_bank_offer = None
        for turn in reversed(negotiation_history):
            if turn.get('speaker') == 'buyer':
                previous_bank_offer = turn.get('offer')
                break

        # Create strategy-specific prompt (domain-aware)
        strategy_prompt = self._create_strategy_prompt(
            strategy=strategy,
            agent_profile=agent_profile,
            history=negotiation_history,
            current_opponent_offer=current_customer_offer,
            current_opponent_dialogue=current_customer_dialogue,
            previous_agent_offer=previous_bank_offer
        )

        # Generate offer using LLM with optional LoRA adapter
        if self.lora_adapter and hasattr(self.model, 'generate'):
            buyer_response = self.model.generate(strategy_prompt, lora_adapter=self.lora_adapter)
        else:
            buyer_response = self.model.generate(strategy_prompt)
        parsed = self._parse_agent_response(buyer_response)

        # Build structured offer (domain-aware)
        buyer_offer = self._build_agent_offer(parsed, agent_profile)
        buyer_dialogue = parsed.get("dialogue", buyer_response)

        return buyer_offer, buyer_dialogue

    def get_tree_statistics(self, root: MCTSNode) -> Dict[str, Any]:
        """
        Get statistics about the MCTS tree.

        Args:
            root: Root node of tree

        Returns:
            Dictionary with tree statistics
        """
        total_nodes = self._count_nodes(root)
        max_depth = self._get_max_depth(root)

        return {
            "total_nodes": total_nodes,
            "max_depth": max_depth,
            "root_visits": root.visit_count,
            "root_reward": root.avg_reward,
            "num_children": len(root.children),
            "best_child_visits": max(c.visit_count for c in root.children) if root.children else 0
        }

    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _get_max_depth(self, node: MCTSNode, current_depth: int = 0) -> int:
        """Get maximum depth of tree."""
        if not node.children:
            return current_depth
        return max(self._get_max_depth(child, current_depth + 1) for child in node.children)


    def _collect_dpo_data_from_tree(
        self,
        root: MCTSNode,
        agent_profile: Dict[str, Any],
        negotiation_stage: int
    ):
        """
        Collect DPO training data from MCTS tree after search.

        Recursively traverses the entire tree to collect preference pairs from all nodes.

        Automatically extracts preference pairs for multiple components:
        1. Bank agent (strategy + offer generation)
        2. Strategy prior predictor
        3. Realization selection

        Args:
            root: Root node of completed MCTS search
            agent_profile: Customer profile for context
            negotiation_stage: Current turn number in negotiation
        """
        if not root.children:
            return

        # Strategy 1: Collect from nodes with multiple children (original approach)
        all_nodes = self._collect_all_nodes(root)


        # Collect all seller utterances from the tree
        seller_utterances = self.collect_seller_utterances(root)
        self.dpo_seller_utterances.extend(seller_utterances)

    def _collect_all_nodes(self, root: MCTSNode) -> List[MCTSNode]:
        """
        Recursively collect all nodes in the tree.

        Args:
            root: Root node of the tree

        Returns:
            List of all nodes in the tree
        """
        nodes = [root]
        for child in root.children:
            nodes.extend(self._collect_all_nodes(child))
        return nodes


    def save_dpo_data(self, seller_profile, prefix: str = ""):
        """
        Save the buffered seller utterances as analyzer training data.

        Args:
            seller_profile: Labels describing the SELLER — the party the
                analyzer predicts. The caller owns these; passing the agent's
                own profile here is what made "Seller Target Price" the
                buyer's target in earlier data.
            prefix: Optional prefix for filenames (e.g. the scenario id)
        """
        if not self.collect_dpo_data:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_str = f"{prefix}_" if prefix else ""
        # Save seller utterances data
        if self.dpo_seller_utterances:
            filepath = self.dpo_output_dir / f"{prefix_str}seller_utterances_{timestamp}.json"
            data = {
                "metadata": {
                    "component": "seller_utterances",
                    "num_examples": len(self.dpo_seller_utterances),
                    "generated_at": datetime.now().isoformat(),
                    "format": "utterances"
                },
                "seller_profile": seller_profile,
                "utterances": self.dpo_seller_utterances
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

    def clear_dpo_buffers(self):
        """Clear DPO data collection buffers."""
        self.dpo_seller_utterances.clear()

    def collect_seller_utterances(self, root):
        """
        Collect all seller utterances from the MCTS tree.

        Args:
            root: Root node of the MCTS tree

        Returns:
            List of dictionaries containing seller utterances with context:
            [
                {
                    "seller_dialogue": str,
                    "seller_offer": dict,
                    "seller_action": str,
                    "buyer_offer": dict (that prompted this response),
                    "buyer_dialogue": str,
                    "negotiation_history": list (full history up to this point),
                    "depth": int,
                    "visit_count": int,
                    "post_search_expanded": bool
                },
                ...
            ]
        """
        utterances = []

        # Collect all nodes in the tree
        all_nodes = self._collect_all_nodes(root)

        for node in all_nodes:
            # Skip root (no seller response yet)
            if node.depth == 0:
                continue

            # Skip nodes without realizations
            if not node.realizations:
                continue

            utterance = {
                "seller_action": node.seller_action,
                "realizations": node.realizations,
                "realization_values": node.realization_values,
                "realization_visits": node.realization_visits,
                "depth": node.depth,
                "visit_count": node.visit_count,
                "post_search_expanded": getattr(node, 'post_search_expanded', False)
            }

            utterances.append(utterance)

        return utterances
