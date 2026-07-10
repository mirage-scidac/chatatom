"""Δa metric — thin adapter that plugs crack_metrics.compute_delta_a into
the campaign metric registry.

Every metric file in src/benchmark/metrics/ exports one function with the
signature `compute(run_folder: str) -> dict`. The registry
(src/benchmark/metrics/__init__.py) discovers them and the campaign tool
`campaign_measure_metrics` dispatches by name.
"""
from ..crack_metrics import compute_delta_a as _compute_delta_a


def compute(run_folder: str) -> dict:
    """Extract Δa (crack growth) metrics from a run folder.

    Returns the full dict from crack_metrics.compute_delta_a:
      {a_initial, a_final, delta_a, n_defect_initial, n_defect_final,
       first_timestep, last_timestep}

    Raises FileNotFoundError if no matching dump files exist — caller
    (campaign_measure_metrics) is expected to catch and convert to an
    {"error": ...} response so the campaign loop can continue.
    """
    return _compute_delta_a(run_folder)
