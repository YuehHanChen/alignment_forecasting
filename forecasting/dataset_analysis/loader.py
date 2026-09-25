"""Load a cached per-dataset auditor report for injection into a forecast prompt.

The report (≤200 words: verdict / problematic behaviors / trigger patterns) is
produced once per FT dataset by `analyze.py` and cached under `reports/`. This
loader returns the report text for a given `ft_dataset`, to be dropped into the
`{dataset_analysis_block}` slot of the `*_da` forecasting prompts.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"

_MISSING = ("(No auditor analysis available for this dataset — treat this section "
            "as empty and rely on the other evidence.)")


def load_report(ft_dataset: str) -> str:
    """Return the cached auditor report text for `ft_dataset`.

    Reads the small `<ft_dataset>.md` (the report text, written only on success);
    falls back to the heavy `<ft_dataset>.json` (full trace) if the .md is absent.
    Returns a clear placeholder (never raises) if the report is missing or errored,
    so prompt rendering is robust when a dataset hasn't been analyzed.
    """
    # Non-reasoning dose-response cells use a `nr-dolci_*` ft_dataset key but share
    # byte-identical training content with the reasoning `dolci_*` datasets — they have
    # no separate report. Strip the `nr-` prefix so they resolve the shared report.
    if ft_dataset.startswith("nr-"):
        ft_dataset = ft_dataset[len("nr-"):]
    md = REPORTS_DIR / f"{ft_dataset}.md"
    if md.exists():
        text = md.read_text().strip()
        if text:
            return text
    js = REPORTS_DIR / f"{ft_dataset}.json"
    if js.exists():
        try:
            d = json.loads(js.read_text())
        except (json.JSONDecodeError, OSError):
            return _MISSING
        report = (d.get("report") or "").strip()
        if report and not d.get("error"):
            return report
    return _MISSING


def has_report(ft_dataset: str) -> bool:
    return load_report(ft_dataset) != _MISSING
