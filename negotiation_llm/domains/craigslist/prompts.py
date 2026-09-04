"""
Prompt templates for the Craigslist Bargains (CB) negotiation domain.

Two prompts:
  1. Buyer prompt (MCTS-controlled agent): adapted from DMNA actor template
  2. Seller prompt (LLM-simulated opponent): adapted from user simulator template

Both use one-sentence responses and the [strategy]response format for the buyer.
"""

import re
from typing import Dict, Any, List, Optional

from .strategies import CraigslistBuyerStrategy, STRATEGY_CONFIGS


# ------------------------------------------------------------------ #
# History formatter                                                    #
# ------------------------------------------------------------------ #

def _format_history(
    history: List[Dict[str, Any]],
    buyer_label: str = "Buyer",
    seller_label: str = "Seller"
) -> str:
    """Format negotiation history for CB domain."""
    if not history:
        return "(No previous conversation)"

    lines = []
    for turn in history:
        speaker = turn.get('speaker', 'unknown')
        dialogue = turn.get('dialogue', '')
        offer = turn.get('offer', {}) or {}
        price = offer.get('price')

        label = buyer_label if speaker == 'buyer' else seller_label
        if price is not None:
            lines.append(f"{label}: {dialogue} [${price:.0f}]")
        else:
            lines.append(f"{label}: {dialogue}")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Buyer prompt (MCTS agent)                                           #
# ------------------------------------------------------------------ #

def create_cb_buyer_prompt(
    strategy: CraigslistBuyerStrategy,
    buyer_profile: Dict[str, Any],
    item_config: Dict[str, Any],
    history: List[Dict[str, Any]],
    current_seller_offer: Optional[Dict[str, Any]] = None,
    current_seller_dialogue: Optional[str] = None,
    previous_buyer_offer: Optional[Dict[str, Any]] = None
) -> str:
    """
    Buyer prompt for MCTS-controlled agent.

    Adapted from the DMNA actor template. The buyer has a target price
    (budget) and should negotiate to get the item as cheaply as possible.

    Output format: [strategy]one-sentence-response
    """
    title = item_config.get("title", "item")
    description = item_config.get("description", "")
    listing_price = item_config.get("listing_price", 0)
    target_price = item_config.get("buyer_target") or buyer_profile.get("target_price")

    # Selected strategy description
    strategy_name = strategy.value
    strategy_desc = STRATEGY_CONFIGS[strategy]['description']

    # Format target price display
    target_str = f"${target_price:.0f}" if target_price is not None else "as low as possible"

    # Format conversation history
    history_text = _format_history(history)

    # Build seller's latest response context
    seller_context = ""
    if current_seller_dialogue:
        price_info = ""
        if current_seller_offer and current_seller_offer.get('price') is not None:
            price_info = f" [asking ${current_seller_offer['price']:.0f}]"
        seller_context = f"\nSeller's last message: {current_seller_dialogue}{price_info}"

    prompt = f"""Now enter the role-playing mode. In the following conversation, you will play as a buyer in a price bargaining game.

Your strategy for this turn is: [{strategy_name}]: {strategy_desc}

Please chat with the Seller using a short and natural sentence.

You are the buyer who is trying to buy the {title} with the price of {target_str}.
Product description: {description}

You must use the strategy [{strategy_name}] and provide a corresponding response based on the Conversation Context.

Please reply with only one short and succinct sentence.

Conversation Context:
{history_text}{seller_context}

Please output the appropriate and high quality response and format your answer as '[{strategy_name}]response'.
Your Response:"""

    return prompt


def create_cb_buyer_strategy_selection_prompt(
    buyer_profile: Dict[str, Any],
    strategies: List[CraigslistBuyerStrategy],
    history: List[Dict[str, Any]],
    item_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Strategy selection prompt: ask LLM to pick the best buyer strategy.

    Used by MCTS for logprob-based prior estimation.
    """
    title = item_config.get("title", "item") if item_config else "item"
    listing_price = item_config.get("listing_price", 0) if item_config else 0
    target_price = (item_config.get("buyer_target") if item_config else None) or buyer_profile.get("target_price")
    target_str = f"${target_price:.0f}" if target_price is not None else "as low as possible"

    strategy_options = "\n".join([
        f"[{s.value}]: {STRATEGY_CONFIGS[s]['description']}"
        for s in strategies
    ])

    # Derive the worked examples from the list actually offered. Hardcoding them
    # let the prompt name [Disagree with a proposal], which MCTS truncates away
    # via max_children — the model was being shown an option it could not pick.
    example = f"[{strategies[0].value}]" if strategies else "[strategy name]"
    if len(strategies) > 1:
        example += f" or [{strategies[-1].value}]"

    history_text = _format_history(history)

    prompt = f"""You are a buyer negotiating to purchase {title} (listed at ${listing_price:.0f}).
Your target price is {target_str}.

Conversation so far:
{history_text}

Available strategies:
{strategy_options}

Select the MOST APPROPRIATE strategy for this moment in the negotiation.

Consider:
- How far along is the negotiation?
- What did the seller last say?
- Which action would best advance your goal of getting a lower price?

Respond with the strategy name in brackets, e.g., {example}.

Strategy:"""

    return prompt


# ------------------------------------------------------------------ #
# Seller prompt (LLM-simulated opponent)                              #
# ------------------------------------------------------------------ #

def create_cb_seller_prompt(
    seller_persona: str,
    item_config: Dict[str, Any],
    history: List[Dict[str, Any]],
    current_buyer_offer: Optional[Dict[str, Any]] = None,
    current_buyer_dialogue: Optional[str] = None,
    seller_target: Optional[float] = None,
) -> str:
    """
    Seller prompt for LLM-simulated opponent.

    Adapted from the CB user simulator template. The seller knows the
    listing price and target price. The seller tries to sell
    for the best possible price.
    """
    title = item_config.get("title", "item")
    description = item_config.get("description", "")
    listing_price = item_config.get("listing_price", 0)

    # Format conversation history
    history_text = _format_history(history)

    # Buyer's latest offer context
    buyer_context = ""
    if current_buyer_dialogue:
        price_info = ""
        if current_buyer_offer and current_buyer_offer.get('price') is not None:
            price_info = f" [offering ${current_buyer_offer['price']:.0f}]"
        buyer_context = f"\nBuyer's last message: {current_buyer_dialogue}{price_info}"

    # The docstring has always claimed the seller knows its target price, but
    # the argument was accepted and never rendered — leaving the simulated CB
    # seller with no reservation price at all. It empirically conceded to ~0.75x
    # its stated target. A2A's seller prompt carries the equivalent line.
    floor_note = ""
    if seller_target is not None:
        floor_note = f"\nYour cost floor is ${seller_target:.0f} — do not accept below this."

    prompt = f"""Now enter the role-playing mode. In the following conversation, you will play as a seller in a price bargaining game.
Your persona: {seller_persona}. You must follow the instructions below during chat.
1. Your utterances and bargain behavior need to strictly follow your persona. Varying your wording and avoid repeating yourself verbatim!
2. You can decide to change your target price flexibly based on your persona and the conversation.
Here are some conversation strategies you can follow:
1. "Source Derogation": Attacks the other party or questions the item.
2. "Counter Argument": Provides a non-personal argument/factual response to refute a previous claim or to justify a new claim.
3. "Personal Choice": Provides a personal reason for disagreeing with the current situation or chooses to agree with the situation provided some specific condition is met.
4. "Information Inquiry": Requests for clarification or asks additional information about the item or situation.
5. "Self Pity": Provides a reason (meant to elicit sympathy) for disagreeing with the current terms.
6. "Hesitance": Stalls for time and is hesitant to commit; specifically, they seek to further the conversation and provide a chance for the other party to make a better offer
7. "Self-assertion": Asserts a new claim or refutes a previous claim with an air of finality/ confidence.
8. "Others": Do not explicitly foil the negotiation attempts.
You are the seller who is trying to sell the {title} with the initial price of ${listing_price:.0f}. Product description: {description}.{floor_note}
Please reply with only one short and succinct sentence.
You must end your response with an Action line in this exact format:
- Action: accept (if you agree to the buyer's price)
- Action: counter(price=$XXX) (if you propose a different price)
- Action: reject (if you refuse to continue negotiating)
********
Conversation History
{history_text}{buyer_context}
********"""

    return prompt


# ------------------------------------------------------------------ #
# Seller action prompt (for acceptance likelihood estimation)         #
# ------------------------------------------------------------------ #

def create_cb_seller_action_prompt(
    seller_persona: str,
    item_config: Dict[str, Any],
    history: List[Dict[str, Any]],
    current_buyer_offer: Optional[Dict[str, Any]] = None,
    current_buyer_dialogue: Optional[str] = None,
    seller_target: Optional[float] = None,
) -> str:
    """
    Seller prompt ending at 'Action:' for logprob-based acceptance estimation.
    """
    base_prompt = create_cb_seller_prompt(
        seller_persona, item_config, history, current_buyer_offer, current_buyer_dialogue,
        seller_target=seller_target,
    )
    # Append a partial response to force the model to complete the Action:
    return base_prompt + "\nDialogue: [what you say]\nAction:"


# ------------------------------------------------------------------ #
# Response parsers                                                     #
# ------------------------------------------------------------------ #

def parse_cb_buyer_response(response: str) -> Dict[str, Any]:
    """
    Parse buyer response in '[strategy]dialogue' format.

    Returns:
        {
            "strategy": str | None,
            "dialogue": str,
            "price": float | None,
            "action_type": "propose" | "accept" | "reject" | "dialogue",
            "accepts": bool,
            "rejects": bool
        }
    """
    result = {
        "strategy": None,
        "dialogue": None,
        "price": None,
        "action_type": "dialogue",
        "accepts": False,
        "rejects": False,
    }

    text = response.strip()

    # Extract [strategy] prefix
    strategy_match = re.match(r'^\[([^\]]+)\]\s*(.*)', text, re.DOTALL)
    if strategy_match:
        result["strategy"] = strategy_match.group(1).strip()
        result["dialogue"] = strategy_match.group(2).strip()
    else:
        result["dialogue"] = text

    dialogue = result["dialogue"] or text

    # Determine action type from strategy name
    strategy_name = (result["strategy"] or "").lower()
    if "agree" in strategy_name:
        result["action_type"] = "accept"
        result["accepts"] = True
    elif "disagree" in strategy_name or "deny" in strategy_name:
        result["action_type"] = "reject"

    # Extract price from dialogue text
    price = _extract_price(dialogue)
    if price is not None:
        result["price"] = price
        # If strategy didn't set action_type yet, infer from context
        if result["action_type"] == "dialogue":
            if "propose" in strategy_name or "counter" in strategy_name or "first" in strategy_name:
                result["action_type"] = "propose"

    # Strategy says "Agree" but buyer stated a price → counter-offer, not acceptance
    if result["action_type"] == "accept" and result["price"] is not None:
        result["action_type"] = "propose"
        result["accepts"] = False

    # Fallback: detect reject from dialogue wording
    # (buyer accepts only via "Agree with the proposal" strategy, not dialogue keywords)
    if result["action_type"] == "dialogue":
        dialogue_lower = dialogue.lower()
        if any(w in dialogue_lower for w in ["no deal", "not interested", "walk away", "forget it"]):
            result["action_type"] = "reject"
            result["rejects"] = True

    return result


def parse_cb_seller_response(response: str) -> Dict[str, Any]:
    """
    Parse seller response (free-form text).

    Returns:
        {
            "dialogue": str,
            "price": float | None,
            "action_type": "accept" | "counter" | "reject" | "dialogue",
            "accepts": bool,
            "rejects": bool
        }
    """
    result = {
        "dialogue": response.strip(),
        "price": None,
        "action_type": "dialogue",
        "accepts": False,
        "rejects": False,
    }

    text = response.strip()

    # Extract dialogue: everything before the first "Action:" line
    action_idx = text.lower().find("action:")
    if action_idx > 0:
        dialogue_text = text[:action_idx].strip()
        dialogue_prefix = re.match(r'(?:Dialogue:\s*)?(.+)', dialogue_text, re.DOTALL)
        if dialogue_prefix:
            result["dialogue"] = dialogue_prefix.group(1).strip()

    # Check for a recognised Action: value (accept / counter / reject).
    action_match = re.search(
        r'Action:\s*(accept|counter|reject)\s*(?:\(\s*price\s*=\s*\$?([\d,]+(?:\.\d+)?)\s*\))?',
        text, re.IGNORECASE,
    )
    if action_match:
        action_type = action_match.group(1).lower()
        result["action_type"] = action_type
        if action_type == "accept":
            result["accepts"] = True
        elif action_type == "reject":
            result["rejects"] = True
        if action_match.group(2):
            result["price"] = float(action_match.group(2).replace(',', ''))
        if result["price"] is None:
            result["price"] = _extract_price(text)
        return result

    # Action: line present but with an unrecognised value (e.g. "hesitance",
    # "self-assertion") — the LLM used a strategy name instead of the mandated
    # accept/counter/reject.  Treat as counter; do NOT fall through to keyword
    # matching which would misfire on words like "deal" inside the dialogue.
    if action_idx >= 0:
        result["action_type"] = "counter"
        result["price"] = _extract_price(text)
        return result

    # No Action: line at all — use a tight keyword set to avoid false positives.
    # Intentionally excludes broad words like "deal", "sounds good", "great".
    text_lower = text.lower()
    accept_words = ["i accept your offer", "you've got a deal", "it's a deal", "it's yours", "sold"]
    reject_words = ["no deal", "not selling", "walk away", "i can't accept", "not interested", "too low"]

    if any(w in text_lower for w in accept_words):
        result["action_type"] = "accept"
        result["accepts"] = True
    elif any(w in text_lower for w in reject_words):
        result["action_type"] = "reject"
        result["rejects"] = True
    else:
        price = _extract_price(text)
        if price is not None:
            result["action_type"] = "counter"
        result["price"] = price

    if result["price"] is None:
        result["price"] = _extract_price(text)

    return result


def _extract_price(text: str) -> Optional[float]:
    """
    Extract the first dollar price mentioned in text, handling:
      - comma-separated numbers: $1,250
      - unit multipliers:        $2.5 million, $1.2 billion, $850 thousand
      - bare dollar amounts:     65 dollars
    """
    _MULTIPLIERS = {"billion": 1e9, "million": 1e6, "thousand": 1e3}
    _UNIT_PAT = re.compile(
        r'\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand)\b',
        re.IGNORECASE,
    )
    _SUFFIX_PAT = re.compile(
        r'\$\s*([\d,]+(?:\.\d+)?)\s*([kmb])\b',
        re.IGNORECASE,
    )
    _SUFFIX_MULTIPLIERS = {"k": 1e3, "m": 1e6, "b": 1e9}
    _DOLLAR_PAT = re.compile(r'\$\s*([\d,]+(?:\.\d+)?)', re.IGNORECASE)
    _WORD_PAT = re.compile(r'([\d,]+(?:\.\d+)?)\s*dollars?', re.IGNORECASE)

    # Try unit-multiplier form first ($2.5 million → 2_500_000)
    m = _UNIT_PAT.search(text)
    if m:
        try:
            val = float(m.group(1).replace(",", "")) * _MULTIPLIERS[m.group(2).lower()]
            if val > 0:
                return val
        except ValueError:
            pass

    # Single-letter suffix ($630k → 630_000, $4.9M → 4_900_000, $1.2B → 1_200_000_000)
    m = _SUFFIX_PAT.search(text)
    if m:
        try:
            val = float(m.group(1).replace(",", "")) * _SUFFIX_MULTIPLIERS[m.group(2).lower()]
            if val > 0:
                return val
        except ValueError:
            pass

    # Plain dollar amount ($1,250 or $65)
    m = _DOLLAR_PAT.search(text)
    if m:
        try:
            val = float(m.group(1).replace(",", ""))
            if val > 0:
                return val
        except ValueError:
            pass

    # "X dollars" fallback
    m = _WORD_PAT.search(text)
    if m:
        try:
            val = float(m.group(1).replace(",", ""))
            if val > 0:
                return val
        except ValueError:
            pass

    return None
