"""Canonical <prob>...</prob> parser — STRICT version.

Per the forecaster prompt convention, the model is instructed to output its
final answer in the form ``<prob>x%</prob>`` where x is a number in [0, 100]
(see e.g. ``rft_tier3/prompts.py``: "*Final answer: <prob>x%</prob>, where x
is strictly between 0 and 100*").

This parser enforces that convention strictly:
  * ``<prob>X%</prob>`` → divide X by 100, return as a fraction in [0, 1].
  * ``<prob>X</prob>``  (no '%') → AMBIGUOUS. Return None so the calling code
    can flag the row for re-evaluation. Historically two different parsers
    interpreted this case differently (api_client treated as percent, Tinker
    scripts treated as decimal), causing 100× divergence on the same string.

Two previously buggy idioms this replaces:

  (1) ``if x > 1.0: x /= 100.0``  (Tinker scripts) — mis-parsed
      ``<prob>1%</prob>`` (X≤1 case): kept x = 1.0, recording 100% instead
      of 1%.

  (2) ``v = float(raw) / 100.0`` (api_client) — silently treats decimal-form
      ``<prob>0.5</prob>`` as 0.5% (0.005) when the model may have intended
      decimal 0.5 (50%). For api_client this matched the prompt convention,
      but it left no way to surface the ambiguity.

Usage:
    from prob_parser import PROB_RE, parse_prob

    p = parse_prob(text)
    if p is None:
        # Either no <prob> tag at all, OR the tag lacks an explicit %.
        # The calling code may distinguish the two via has_prob_tag().
        ...
"""
from __future__ import annotations
import re

# Capture: the digits/dot, and SEPARATELY whether '%' is present.
PROB_RE = re.compile(r"<prob>\s*([\d.]+)\s*(%?)\s*</prob>", re.IGNORECASE)


def parse_prob(text: str | None) -> float | None:
    """Strict-percent parser.

    Returns:
        float in [0, 1] when text contains at least one ``<prob>X%</prob>`` tag
        (X is divided by 100).
        None when no tag is present, or when the last tag lacks an explicit
        ``%`` sign. (Use ``has_prob_tag`` to distinguish.)
    """
    if not text:
        return None
    matches = PROB_RE.findall(text)
    if not matches:
        return None
    raw, pct = matches[-1]
    if pct != "%":
        # Ambiguous: the model omitted the '%' sign. Decline to guess.
        return None
    try:
        x = float(raw.strip()) / 100.0
    except ValueError:
        return None
    return max(0.0, min(1.0, x))


def has_prob_tag(text: str | None) -> bool:
    """Return True if `text` contains any <prob>…</prob> tag, regardless of
    whether the tag has a '%'. Useful for distinguishing 'no answer at all'
    from 'answered but without the required % sign'.
    """
    if not text:
        return False
    return bool(PROB_RE.search(text))
