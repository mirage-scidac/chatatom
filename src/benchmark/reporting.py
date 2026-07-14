"""Campaign final-report rendering.

Two responsibilities:
  render_campaign_graph(campaign_folder)  → writes report.png
  write_report_skeleton(campaign_folder)  → writes report.md skeleton + returns text

The campaign_generate_report tool calls both; the LLM then writes the
'What was learned' narrative section and calls campaign_save_narrative
to append it under the skeleton's placeholder heading.

All disk reads use plan.json + trials.jsonl; direction-aware (min vs max) via
plan.objective.direction so this file is metric-agnostic — adding a new metric
to the registry requires zero changes here.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — server has no display
import matplotlib.pyplot as plt


NARRATIVE_HEADING = "## What was learned"
NARRATIVE_PLACEHOLDER = "_[to be written by the AI]_"


# ── File readers ─────────────────────────────────────────────────────────────

def _read_plan(campaign_folder: str) -> Dict[str, Any]:
    with open(os.path.join(campaign_folder, "plan.json")) as f:
        return json.load(f)


def _read_trials(campaign_folder: str) -> List[Dict[str, Any]]:
    """Read trials.jsonl (append-only, one JSON per line). Silently skips
    blank lines. Missing file → empty list (a campaign that never ran a trial
    still deserves a report — the skeleton just shows zero rows)."""
    path = os.path.join(campaign_folder, "trials.jsonl")
    if not os.path.exists(path):
        return []
    trials = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trials.append(json.loads(line))
    return trials


# ── Derived summaries (shared with campaign_get_state / campaign_check_stop) ──

def _metric_value(trial: Dict[str, Any], metric_name: str) -> Optional[float]:
    """Pull the objective metric out of a trial row. None if the trial failed
    or the metric key is absent."""
    metrics = trial.get("metrics")
    if not isinstance(metrics, dict):
        return None
    v = metrics.get(metric_name)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _find_best(
    trials: List[Dict[str, Any]], metric_name: str, direction: str
) -> Optional[Dict[str, Any]]:
    """Return the winning trial (or None if no successful trials).
    direction: 'minimize' → smallest metric wins; 'maximize' → largest."""
    scored = [(t, _metric_value(t, metric_name)) for t in trials]
    scored = [(t, v) for t, v in scored if v is not None]
    if not scored:
        return None
    reverse = direction == "maximize"
    scored.sort(key=lambda tv: tv[1], reverse=reverse)
    return scored[0][0]


# ── Graph rendering ──────────────────────────────────────────────────────────

def render_campaign_graph(campaign_folder: str) -> str:
    """Render report.png: two panels stacked.
      Top:    objective metric vs iteration + best-so-far step line
      Bottom: one small scatter per knob (values tried per iteration)

    Returns the absolute path to the written PNG.
    """
    plan = _read_plan(campaign_folder)
    trials = _read_trials(campaign_folder)

    objective = plan.get("objective", {})
    metric_name = objective.get("metric", "delta_a")
    direction = objective.get("direction", "minimize")
    display_name = objective.get("display_name", metric_name)
    units = objective.get("units", "")
    knob_names = list((plan.get("knobs") or {}).keys())

    iterations = [t.get("iteration") for t in trials]
    metric_vals = [_metric_value(t, metric_name) for t in trials]

    # Best-so-far step line
    best_so_far: List[Optional[float]] = []
    running_best: Optional[float] = None
    for v in metric_vals:
        if v is None:
            best_so_far.append(running_best)
            continue
        if running_best is None:
            running_best = v
        elif direction == "maximize" and v > running_best:
            running_best = v
        elif direction == "minimize" and v < running_best:
            running_best = v
        best_so_far.append(running_best)

    n_knobs = len(knob_names)
    # gridspec: 1 row for metric (weight 2) + 1 row for knobs (weight 1)
    fig = plt.figure(figsize=(max(6, 1.2 * max(len(trials), 5)), 6))
    gs = fig.add_gridspec(2, max(1, n_knobs), height_ratios=[2, 1])

    # ── Top panel ─────────────────────────────────────────────────────────
    ax_top = fig.add_subplot(gs[0, :])
    successful_x = [i for i, v in zip(iterations, metric_vals) if v is not None]
    successful_y = [v for v in metric_vals if v is not None]
    failed_x = [i for i, v in zip(iterations, metric_vals) if v is None]

    if successful_x:
        ax_top.plot(successful_x, successful_y, marker="o", linewidth=1.2, label=display_name)
    if failed_x:
        ax_top.scatter(failed_x, [0] * len(failed_x), marker="x", color="crimson",
                       s=60, label="failed trial")
    y_label = f"{display_name} ({units})" if units else display_name
    ax_top.set_ylabel(y_label)
    ax_top.set_xlabel("Iteration")
    ax_top.set_title(f"Campaign {plan.get('campaign_id', '')} — {display_name} per iteration")
    ax_top.grid(True, alpha=0.3)
    if successful_x or failed_x:
        ax_top.legend(loc="best", fontsize=9)

    # ── Bottom panels: one per knob ───────────────────────────────────────
    if n_knobs == 0:
        ax_none = fig.add_subplot(gs[1, :])
        ax_none.text(0.5, 0.5, "(no knobs defined)", ha="center", va="center",
                     transform=ax_none.transAxes, color="grey", fontsize=10)
        ax_none.set_axis_off()
    else:
        for col, knob in enumerate(knob_names):
            ax = fig.add_subplot(gs[1, col])
            xs, ys = [], []
            for t in trials:
                knobs = t.get("knobs") or {}
                v = knobs.get(knob)
                if v is None:
                    continue
                try:
                    ys.append(float(v))
                    xs.append(t.get("iteration"))
                except (TypeError, ValueError):
                    continue
            if xs:
                ax.scatter(xs, ys, s=25, color="steelblue")
            ax.set_title(knob, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("value", fontsize=8)
            ax.set_xlabel("iter", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(campaign_folder, "report.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


# ── Markdown skeleton ────────────────────────────────────────────────────────

def _format_number(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def _format_knobs_inline(knobs: Dict[str, Any]) -> str:
    if not knobs:
        return "—"
    return ", ".join(f"{k}={_format_number(v)}" for k, v in knobs.items())


def _build_trial_table(
    trials: List[Dict[str, Any]],
    knob_names: List[str],
    metric_name: str,
    display_name: str,
    best_value: Optional[float],
    direction: str,
) -> str:
    """Markdown table: iter | <each knob> | metric | delta_from_best | status."""
    if not trials:
        return "_No trials recorded._\n"

    header = ["Iter", *knob_names, display_name, "Δ from best", "Status"]
    sep = ["----"] * len(header)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(sep) + "|"]
    for t in trials:
        knobs = t.get("knobs") or {}
        v = _metric_value(t, metric_name)
        status = "failed" if t.get("failure_reason") else "ok"
        if v is None:
            delta_col = "—"
            metric_col = "—"
        else:
            metric_col = _format_number(v)
            if best_value is None:
                delta_col = "—"
            else:
                diff = v - best_value
                if direction == "maximize":
                    diff = -diff  # positive = worse for maximize too
                delta_col = f"{diff:+.3g}"
                if abs(diff) < 1e-12:
                    delta_col = "0 ← best"
        row = [str(t.get("iteration", "?"))]
        row += [_format_number(knobs.get(k)) for k in knob_names]
        row += [metric_col, delta_col, status]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def write_report_skeleton(
    campaign_folder: str, stop_reason: Optional[str] = None
) -> Tuple[str, str]:
    """Write report.md with mechanical facts + an empty narrative placeholder.

    Returns (md_path, skeleton_text). The narrative placeholder line is
    NARRATIVE_PLACEHOLDER — campaign_save_narrative replaces it in-place with
    the LLM-authored text.
    """
    plan = _read_plan(campaign_folder)
    trials = _read_trials(campaign_folder)

    objective = plan.get("objective", {})
    metric_name = objective.get("metric", "delta_a")
    direction = objective.get("direction", "minimize")
    display_name = objective.get("display_name", metric_name)
    units = objective.get("units", "")
    knob_names = list((plan.get("knobs") or {}).keys())

    best_trial = _find_best(trials, metric_name, direction)
    best_value = _metric_value(best_trial, metric_name) if best_trial else None
    max_iters = plan.get("max_iterations", "?")

    lines: List[str] = []
    lines.append(f"# Campaign {plan.get('campaign_id', '')}")
    lines.append("")
    lines.append("## Objective")
    lines.append(f"- Metric: **{display_name}** ({metric_name})")
    lines.append(f"- Direction: {direction}")
    if units:
        lines.append(f"- Units: {units}")
    target = objective.get("target")
    if target is not None:
        lines.append(f"- Target: {_format_number(target)} {units}".rstrip())
    lines.append("")

    template = plan.get("template") or {}
    if template:
        lines.append("## Locked template")
        lines.append("Values chosen at plan time; identical across every iteration.")
        lines.append("")
        # Sort phase_<N> keys numerically; keep non-phase keys in their given order first
        import re as _re
        phase_keys = []
        other_keys = []
        for k in template.keys():
            if _re.fullmatch(r"phase_\d+", k):
                phase_keys.append(k)
            else:
                other_keys.append(k)
        phase_keys.sort(key=lambda k: int(k.split("_")[1]))
        for k in other_keys + phase_keys:
            v = template[k]
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("## Result")
    lines.append(f"- Total iterations: {len(trials)} of {max_iters}")
    lines.append(f"- Stop reason: `{stop_reason or 'not recorded'}`")
    if best_trial is not None:
        lines.append(
            f"- Best {display_name}: **{_format_number(best_value)} {units}**".rstrip()
            + f" at iteration {best_trial.get('iteration')}"
        )
        lines.append(f"- Winning knobs: {_format_knobs_inline(best_trial.get('knobs') or {})}")
        lines.append(f"- Run folder: `{best_trial.get('run_folder', '?')}`")
    else:
        lines.append(f"- Best {display_name}: — (no successful trials)")
    lines.append("")

    lines.append(f"## Trial-by-trial log ({len(trials)} trials)")
    lines.append("")
    lines.append(_build_trial_table(trials, knob_names, metric_name,
                                    display_name, best_value, direction))

    lines.append("## Files")
    lines.append("- Plan: `plan.json`")
    lines.append("- Trials: `trials.jsonl`")
    lines.append("- Graph: `report.png`")
    lines.append("- Per-iteration LAMMPS output: `run_*/`")
    lines.append("")

    lines.append(NARRATIVE_HEADING)
    lines.append("")
    lines.append(NARRATIVE_PLACEHOLDER)
    lines.append("")

    text = "\n".join(lines)
    md_path = os.path.join(campaign_folder, "report.md")
    with open(md_path, "w") as f:
        f.write(text)
    return os.path.abspath(md_path), text


def append_narrative(campaign_folder: str, narrative_text: str) -> str:
    """Replace the NARRATIVE_PLACEHOLDER in report.md with the LLM's text.
    Idempotent-ish: if the placeholder is already gone (already saved),
    appends a new '## What was learned (updated)' section instead of
    overwriting the earlier narrative.

    Returns the absolute path to report.md.
    """
    md_path = os.path.join(campaign_folder, "report.md")
    with open(md_path) as f:
        content = f.read()

    if NARRATIVE_PLACEHOLDER in content:
        content = content.replace(NARRATIVE_PLACEHOLDER, narrative_text.strip())
    else:
        content = content.rstrip() + "\n\n## What was learned (updated)\n\n" + narrative_text.strip() + "\n"

    with open(md_path, "w") as f:
        f.write(content)
    return os.path.abspath(md_path)
