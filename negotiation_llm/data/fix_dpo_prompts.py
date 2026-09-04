"""
Repair the strategy block in already-collected CB DPO files.

Rewrites each example's prompt so its "Available strategies:" block matches the
list MCTS actually expands at inference — the first MCTS_CONFIG["max_children"]
strategies, not all 11. An earlier version of this script injected all 11, which
is the opposite of what the search sees; files repaired with that version should
be repaired again with this one.

Usage:
    python -m negotiation_llm.data.fix_dpo_prompts --data-dir cb_dpo_train
    python -m negotiation_llm.data.fix_dpo_prompts --data-dir cb_dpo_train --dry-run
"""

import argparse
import json
import re
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


from negotiation_llm.config import MCTS_CONFIG
from negotiation_llm.domains.craigslist.strategies import CraigslistBuyerStrategy, STRATEGY_CONFIGS


# Mirror the search: MCTSNegotiator._expand_node slices the strategy list to
# max_children, so anything past that is never scored at inference and must not
# appear in a training prompt either.
_MAX_CHILDREN = MCTS_CONFIG.get("max_children")
SEARCH_STRATEGIES = list(CraigslistBuyerStrategy)[:_MAX_CHILDREN] if _MAX_CHILDREN else list(CraigslistBuyerStrategy)

ALL_STRATEGY_BLOCK = "\n".join([
    f"[{s.value}]: {STRATEGY_CONFIGS[s]['description']}"
    for s in SEARCH_STRATEGIES
])


def rebuild_prompt(old_prompt: str) -> str:
    """
    Replace the 'Available strategies:' block in a prompt with the block the
    search actually offers (max_children entries).
    """
    # Match everything between "Available strategies:\n" and the next "\n\n"
    pattern = r"(Available strategies:\n)(.*?)(\n\nSelect the MOST APPROPRIATE)"
    replacement = rf"\g<1>{ALL_STRATEGY_BLOCK}\g<3>"
    new_prompt, count = re.subn(pattern, replacement, old_prompt, flags=re.DOTALL)
    if count == 0:
        return None  # pattern not found — skip
    return new_prompt


def fix_file(filepath: Path, output_dir: Optional[Path] = None, dry_run: bool = False) -> Dict[str, int]:
    """Fix prompts in a single DPO JSON file. Returns stats."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    examples = data.get("examples", [])
    fixed = 0
    skipped = 0

    for example in examples:
        old_prompt = example.get("prompt", "")
        new_prompt = rebuild_prompt(old_prompt)
        if new_prompt is None:
            skipped += 1
            continue
        if new_prompt != old_prompt:
            example["prompt"] = new_prompt
            fixed += 1

    if fixed > 0 and not dry_run:
        out_path = (output_dir / filepath.name) if output_dir else filepath
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    return {"fixed": fixed, "skipped": skipped, "total": len(examples)}


def main():
    parser = argparse.ArgumentParser(description="Fix CB DPO prompts to include all 11 strategies")
    parser.add_argument("--data-dir", type=str, default="cb_dpo_train",
                        help="Directory containing CB DPO JSON files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for fixed files (default: overwrite in place)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(data_dir.glob("*cb_dpo*.json"))
    if not json_files:
        print(f"No CB DPO JSON files found in {data_dir}")
        return

    dest = output_dir or data_dir
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(json_files)} files in {data_dir}/ → {dest}/")
    print(f"Full strategy list ({len(ALL_STRATEGIES)} strategies):")
    for s in ALL_STRATEGIES:
        print(f"  [{s.value}]")
    print()

    total_fixed = 0
    total_examples = 0
    files_modified = 0

    for fp in json_files:
        stats = fix_file(fp, output_dir=output_dir, dry_run=args.dry_run)
        total_fixed += stats["fixed"]
        total_examples += stats["total"]
        if stats["fixed"] > 0:
            files_modified += 1
            print(f"  {fp.name}: fixed {stats['fixed']}/{stats['total']} examples")

    print()
    print(f"Done. {total_fixed}/{total_examples} examples fixed across {files_modified}/{len(json_files)} files.")
    if args.dry_run:
        print("(dry run — no files were modified)")


if __name__ == "__main__":
    main()
