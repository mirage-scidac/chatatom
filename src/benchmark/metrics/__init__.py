"""Campaign metric registry.

Each metric is a file in this folder that exports `compute(run_folder) -> dict`.
Register it here by adding one entry to METRIC_REGISTRY.

New metrics plug in without touching any campaign_* tool: they just show up in
list_metrics() and become dispatchable through compute_metric().

Registry entry fields:
  compute:      callable(run_folder) -> dict
  direction:    "minimize" | "maximize"   (default direction; LLM may override
                                           in plan.objective.direction)
  units:        display units string (e.g. "Å")
  display_name: pretty label for reports / chat (e.g. "Δa")
"""
from typing import Any, Dict, List

from . import delta_a as _delta_a


METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "delta_a": {
        "compute":      _delta_a.compute,
        "direction":    "minimize",
        "units":        "Å",
        "display_name": "Δa",
    },
    # Add new metrics here — one entry each. Drop a compute(run_folder) function
    # into src/benchmark/metrics/<name>.py, import it above, then list it here.
}


def list_metrics() -> List[str]:
    """Registered metric names — used by the campaign skill to advertise
    valid choices to the LLM during Turn 1 planning."""
    return list(METRIC_REGISTRY.keys())


def get_metric_meta(metric_name: str) -> Dict[str, Any]:
    """Return the registry entry's metadata dict (no compute function).
    Used by the campaign skill + generate_report for labels/direction."""
    entry = METRIC_REGISTRY.get(metric_name)
    if entry is None:
        return {}
    return {k: v for k, v in entry.items() if k != "compute"}


def compute_metric(metric_name: str, run_folder: str) -> Dict[str, Any]:
    """Dispatch to the named extractor and return its dict.

    Never raises — unknown metric names and extractor failures both become
    {"error": "..."} responses so the campaign loop can record the failure as
    a trial and continue instead of crashing the whole ReAct trace.
    """
    entry = METRIC_REGISTRY.get(metric_name)
    if entry is None:
        return {
            "error": f"unknown metric {metric_name!r}",
            "available": list_metrics(),
        }
    try:
        return entry["compute"](run_folder)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
