"""
Campaign-mode helpers.

Provides:
  - format_history_message(): synthesizes the "user message" for iteration N
    from past iteration results. Used by CampaignManager._run_one_iteration
    to build the "Past attempts: 1) strain=... → Δa=...  Propose iteration N..."
    prompt that triggers the orchestrator-LLM's campaign-skill iteration mode.

Setup/iteration instructions and the trigger description for the campaign
mode live in src/orchestrator/skills/campaign.md (loaded via the
load_skill tool).
"""
from typing import Dict, List


def format_history_message(
    past_runs: List[Dict],
    iteration: int,
    target: str = "minimize delta_a",
) -> str:
    """
    Build the synthesized "user message" for iteration N (N ≥ 2).

    Each past_runs entry should have:
      - "params_text": str — human-readable summary of the iteration's knobs
                              (e.g. "strain=7e-3, T=300, crack_len=30%,
                               crack_w=4a₀")
      - "delta_a": float | None — measured Δa value, or None on failure

    Returns a string like:
        Past attempts:
          1) strain=7e-3, T=300, crack_len=30%, crack_w=4a₀ → Δa = +12.4 Å
          2) strain=4e-3, T=250, crack_len=20%, crack_w=4a₀ → Δa = +8.1 Å

        Goal: minimize delta_a (target Δa ≤ 0).
        Propose iteration 3. State your hypothesis and provide all 4
        parameter values.
    """
    lines = ["Past attempts:"]
    for i, run in enumerate(past_runs, start=1):
        params = run.get("params_text", "<unknown params>")
        d = run.get("delta_a")
        if d is None:
            outcome = "Δa = N/A (failed)"
        else:
            outcome = f"Δa = {d:+.2f} Å"
        lines.append(f"  {i}) {params} → {outcome}")

    lines.append("")
    lines.append(f"Goal: {target} (target Δa ≤ 0).")
    lines.append(
        f"Propose iteration {iteration}. State your hypothesis and "
        "provide all 4 parameter values."
    )
    return "\n".join(lines)
