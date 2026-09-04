"""
Prompt templates and response parsers for the A2A consumer negotiation domain.

Buyer = agent (tries to purchase at or near wholesale price).
Seller = LLM opponent (professional seller who knows their cost floor).
"""

import re
from typing import Dict, Any, List, Optional

from .strategies import A2ABuyerStrategy, STRATEGY_CONFIGS


# ------------------------------------------------------------------ #
# History formatter                                                    #
# ------------------------------------------------------------------ #

def _format_history(
    history: List[Dict[str, Any]],
    buyer_label: str = "Buyer",
    seller_label: str = "Seller",
) -> str:
    if not history:
        return "(No previous conversation)"

    lines = []
    for turn in history:
        speaker = turn.get("speaker", "unknown")
        dialogue = turn.get("dialogue", "")
        offer = turn.get("offer", {}) or {}
        price = offer.get("price")

        label = buyer_label if speaker == "buyer" else seller_label
        if price is not None:
            lines.append(f"{label}: {dialogue} [${price:,.0f}]")
        else:
            lines.append(f"{label}: {dialogue}")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Buyer prompt (MCTS / strategy-based)                                #
# ------------------------------------------------------------------ #

def create_a2a_buyer_prompt(
    strategy: A2ABuyerStrategy,
    buyer_profile: Dict[str, Any],
    item_config: Dict[str, Any],
    history: List[Dict[str, Any]],
    current_seller_offer: Optional[Dict[str, Any]] = None,
    current_seller_dialogue: Optional[str] = None,
    previous_buyer_offer: Optional[Dict[str, Any]] = None,
) -> str:
    name = item_config.get("name", "product")
    features = item_config.get("features", "")
    product_type = item_config.get("type", "product")
    listing_price = item_config.get("listing_price", 0)
    target_price = buyer_profile.get("target_price")

    strategy_name = strategy.value
    strategy_desc = STRATEGY_CONFIGS[strategy]["description"]
    target_str = f"${target_price:,.0f}" if target_price is not None else "as low as possible"
    history_text = _format_history(history)

    seller_context = ""
    if current_seller_dialogue:
        price_info = ""
        if current_seller_offer and current_seller_offer.get("price") is not None:
            price_info = f" [asking ${current_seller_offer['price']:,.0f}]"
        seller_context = f"\nSeller's last message: {current_seller_dialogue}{price_info}"

    return f"""Now enter the role-playing mode. In the following conversation, you will play as a buyer negotiating to purchase a {product_type}.

Your strategy for this turn is: [{strategy_name}]: {strategy_desc}

You are trying to buy: {name}
Product features: {features}
Listed at: ${listing_price:,.0f}
Your target price: {target_str}

Use the strategy [{strategy_name}] and reply with only one short and succinct sentence.

Conversation Context:
{history_text}{seller_context}

Format your answer as '[{strategy_name}]response'.
Your Response:"""


# ------------------------------------------------------------------ #
# Buyer strategy selection prompt                                      #
# ------------------------------------------------------------------ #

def create_a2a_buyer_strategy_selection_prompt(
    buyer_profile: Dict[str, Any],
    strategies: List[A2ABuyerStrategy],
    history: List[Dict[str, Any]],
    item_config: Optional[Dict[str, Any]] = None,
) -> str:
    name = item_config.get("name", "product") if item_config else "product"
    listing_price = item_config.get("listing_price", 0) if item_config else 0
    target_price = buyer_profile.get("target_price")
    target_str = f"${target_price:,.0f}" if target_price is not None else "as low as possible"

    strategy_options = "\n".join(
        f"[{s.value}]: {STRATEGY_CONFIGS[s]['description']}"
        for s in strategies
    )
    history_text = _format_history(history)

    return f"""You are a buyer negotiating to purchase {name} (listed at ${listing_price:,.0f}).
Your target price is {target_str}.

Conversation so far:
{history_text}

Available strategies:
{strategy_options}

Select the MOST APPROPRIATE strategy for this moment.
Respond with only the strategy name in brackets, e.g., [Propose a counter price].

Strategy:"""


# ------------------------------------------------------------------ #
# Seller prompt (LLM opponent)                                        #
# ------------------------------------------------------------------ #

def create_a2a_seller_prompt(
    seller_persona: str,
    item_config: Dict[str, Any],
    history: List[Dict[str, Any]],
    current_buyer_offer: Optional[Dict[str, Any]] = None,
    current_buyer_dialogue: Optional[str] = None,
    seller_target: Optional[float] = None,
    predicted_action: Optional[str] = None,
    predicted_counter: Optional[float] = None,
) -> str:
    name = item_config.get("name", "product")
    features = item_config.get("features", "")
    product_type = item_config.get("type", "product")
    listing_price = item_config.get("listing_price", 0)
    wholesale_price = item_config.get("wholesale_price")

    history_text = _format_history(history)

    buyer_context = ""
    if current_buyer_dialogue:
        price_info = ""
        if current_buyer_offer and current_buyer_offer.get("price") is not None:
            price_info = f" [offering ${current_buyer_offer['price']:,.0f}]"
        buyer_context = f"\nBuyer's last message: {current_buyer_dialogue}{price_info}"

    floor_note = ""
    if seller_target is not None:
        floor_note = f"\nYour cost floor is ${seller_target:,.0f} — do not accept below this."

    # Soft behavioral lean inferred from the negotiation so far. Guides the
    # simulated seller toward the most likely response without overriding the
    # persona/floor, so MCTS rollouts model a more realistic opponent.
    inclination_note = ""
    hints = []
    if predicted_action:
        hints.append(f"you are inclined to {predicted_action} the buyer's latest offer")
    if predicted_counter is not None:
        hints.append(f"if you counter, a fair price for you is around ${predicted_counter:,.0f}")
    if hints:
        inclination_note = (
            "\nYour current inclination: " + "; ".join(hints)
            + " — but stay true to your persona, the conversation, and your cost floor."
        )

    return f"""Now enter the role-playing mode. In the following conversation, you will play as a {product_type} seller.
Your persona: {seller_persona}. Follow these instructions during the negotiation:
1. Your utterances and bargaining behavior must reflect your persona. Vary your wording and avoid repeating yourself verbatim.
2. You can adjust your price flexibly based on the conversation, but never go below your cost floor.
Here are some conversation strategies you can follow:
1. "Source Derogation": Attacks the other party or questions the item.
2. "Counter Argument": Provides a non-personal argument/factual response to refute a previous claim or to justify a new claim.
3. "Personal Choice": Provides a personal reason for disagreeing with the current situation or chooses to agree with the situation provided some specific condition is met.
4. "Information Inquiry": Requests for clarification or asks additional information about the item or situation.
5. "Self Pity": Provides a reason (meant to elicit sympathy) for disagreeing with the current terms.
6. "Hesitance": Stalls for time and is hesitant to commit; specifically, they seek to further the conversation and provide a chance for the other party to make a better offer
7. "Self-assertion": Asserts a new claim or refutes a previous claim with an air of finality/ confidence.
8. "Others": Do not explicitly foil the negotiation attempts.

You are selling: {name}
Product features: {features}
Listed at: ${listing_price:,.0f}{floor_note}{inclination_note}

Reply with only one short and succinct sentence, then end with an Action line:
- Action: accept (you agree to the buyer's offer)
- Action: counter(price=$XXX) (you propose a different price)
- Action: reject (you refuse to continue at this price)
********
Conversation History:
{history_text}{buyer_context}
********"""


# ------------------------------------------------------------------ #
# Seller action prompt (for acceptance likelihood estimation)         #
# ------------------------------------------------------------------ #

def create_a2a_seller_action_prompt(
    seller_persona: str,
    item_config: Dict[str, Any],
    history: List[Dict[str, Any]],
    current_buyer_offer: Optional[Dict[str, Any]] = None,
    current_buyer_dialogue: Optional[str] = None,
    seller_target: Optional[float] = None,
    predicted_action: Optional[str] = None,
    predicted_counter: Optional[float] = None,
) -> str:
    base = create_a2a_seller_prompt(
        seller_persona, item_config, history,
        current_buyer_offer, current_buyer_dialogue, seller_target,
        predicted_action, predicted_counter,
    )
    return base + "\nDialogue: [what you say]\nAction:"


# ------------------------------------------------------------------ #
# Response parsers                                                     #
# ------------------------------------------------------------------ #

def parse_a2a_buyer_response(response: str) -> Dict[str, Any]:
    """
    Parse buyer response in '[strategy]dialogue' format.

    Returns:
        {strategy, dialogue, price, action_type, accepts, rejects}
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

    strategy_match = re.match(r'^\[([^\]]+)\]\s*(.*)', text, re.DOTALL)
    if strategy_match:
        result["strategy"] = strategy_match.group(1).strip()
        result["dialogue"] = strategy_match.group(2).strip()
    else:
        result["dialogue"] = text

    dialogue = result["dialogue"] or text
    strategy_name = (result["strategy"] or "").lower()

    if "agree" in strategy_name:
        result["action_type"] = "accept"
        result["accepts"] = True
    elif "disagree" in strategy_name or "deny" in strategy_name:
        result["action_type"] = "reject"

    price = _extract_price(dialogue)
    if price is not None:
        result["price"] = price
        if result["action_type"] == "dialogue":
            if any(k in strategy_name for k in ("propose", "counter", "first")):
                result["action_type"] = "propose"

    # Strategy says "Agree" but buyer stated a price → counter-offer, not acceptance
    if result["action_type"] == "accept" and result["price"] is not None:
        result["action_type"] = "propose"
        result["accepts"] = False

    if result["action_type"] == "dialogue":
        dl = dialogue.lower()
        if any(w in dl for w in ["no deal", "not interested", "walk away", "forget it"]):
            result["action_type"] = "reject"
            result["rejects"] = True

    return result


def parse_a2a_seller_response(response: str) -> Dict[str, Any]:
    """
    Parse seller response (free-form + Action: line).

    Returns:
        {dialogue, price, action_type, accepts, rejects}
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
        m = re.match(r'(?:Dialogue:\s*)?(.+)', dialogue_text, re.DOTALL)
        if m:
            result["dialogue"] = m.group(1).strip()

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
            result["price"] = float(action_match.group(2).replace(",", ""))
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
    Extract the first dollar amount from text, handling:
      - comma-separated numbers: $1,250,000
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

    # Plain dollar amount ($1,250,000 or $65)
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
