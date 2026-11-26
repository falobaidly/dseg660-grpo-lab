"""Reward verifier for the CoD GRPO lab. Pure Python (no torch) so it can be unit-tested anywhere.

Rule: the reply must START with one choice letter (optionally after "Answer:", "(", "[", "**"),
and the whole reply must name no other standalone choice letter. Only uppercase A/B/C count,
so the article "a" and words like "Answer" never parse as choices. Anything else is invalid.
"""
import re

_THINK = re.compile(r"^\s*<think>.*?</think>", re.S)  # stray empty think block, if any
_LEAD = re.compile(r"^\s*(?:(?i:(?:final\s+)?answer(?:\s+is)?)\s*[:\-]?\s*)?[\(\[\*\s]*([ABC])(?![A-Za-z0-9])")
_LETTER = re.compile(r"(?<![A-Za-z0-9])[ABC](?![A-Za-z0-9])")


def extract_choice(text):
    """Return 'A'/'B'/'C' if the reply contains exactly one unambiguous choice, else None."""
    if not isinstance(text, str):
        return None
    t = _THINK.sub("", text, count=1)
    m = _LEAD.match(t)
    if not m:
        return None  # malformed: no leading choice
    return m.group(1) if set(_LETTER.findall(t)) == {m.group(1)} else None  # else ambiguous


def reward(completion, reference):
    """Binary reward: 1.0 iff the extracted choice equals the reference letter. Invalid -> 0.0."""
    choice = extract_choice(completion)
    return 1.0 if choice is not None and choice == str(reference).strip().upper() else 0.0
