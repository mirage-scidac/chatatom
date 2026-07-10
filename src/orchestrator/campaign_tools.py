"""Campaign v2 tools — the orchestrator drives the whole loop by calling
these directly. No daemon thread, no _DualQueue, no CampaignManager.

All 7 tools live here and are exported via CAMPAIGN_TOOLS for registration
in ORCHESTRATOR_TOOLS. Every tool is failure-safe: outer try/except returns
{"error": "..."} instead of raising, so a bad tool call never poisons the
ReAct trace.

State is persisted to disk under runs/<campaign_id>/:
  plan.json       (immutable, written once by campaign_start)
  control.json    (pause/resume/cancel flag)
  trials.jsonl    (append-only, one JSON line per iteration)

SSE emission goes through the existing _q helper in graph.nodes so the
frontend's existing handlers pick up the events unchanged.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import Annotated

# Make sibling-module imports work when the tool runs inside the orchestrator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.metrics import compute_metric, list_metrics, get_metric_meta
from benchmark.reporting import (
    append_narrative,
    render_campaign_graph,
    write_report_skeleton,
)
from graph.nodes import SESSION_QUEUES


# ── Config ────────────────────────────────────────────────────────────────────

RUNS_BASE = "runs"


# ── Path helpers ─────────────────────────────────────────────────────────────

def _campaign_folder(campaign_id: str) -> str:
    return os.path.join(RUNS_BASE, campaign_id)


def _plan_path(campaign_id: str) -> str:
    return os.path.join(_campaign_folder(campaign_id), "plan.json")


def _control_path(campaign_id: str) -> str:
    return os.path.join(_campaign_folder(campaign_id), "control.json")


def _trials_path(campaign_id: str) -> str:
    return os.path.join(_campaign_folder(campaign_id), "trials.jsonl")


# ── Small I/O helpers ────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _write_json_atomic(path: str, data: Any) -> None:
    """Write JSON atomically via tmp + rename so partial writes never leave
    corrupt files on disk."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def _read_trials(campaign_id: str) -> List[Dict[str, Any]]:
    path = _trials_path(campaign_id)
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _append_trial(campaign_id: str, trial: Dict[str, Any]) -> None:
    """Append one JSON line to trials.jsonl."""
    with open(_trials_path(campaign_id), "a") as f:
        f.write(json.dumps(trial, default=str) + "\n")


# ── SSE emission (reuses SESSION_QUEUES from graph.nodes) ────────────────────

def _emit(session_id: Optional[str], event_type: str, **payload: Any) -> None:
    """Push an event onto the session's SSE queue. No-op if session isn't
    registered (e.g. tests, headless runs)."""
    if not session_id:
        return
    q = SESSION_QUEUES.get(session_id)
    if q is None:
        return
    try:
        q.put({"type": event_type, **payload})
    except Exception as e:
        # Never let a bad SSE emit crash the tool
        print(f"[campaign SSE] failed to emit {event_type}: {e}")


# ── Plan validation ──────────────────────────────────────────────────────────

REQUIRED_PLAN_KEYS = ("objective", "template", "knobs", "max_iterations")
REQUIRED_OBJECTIVE_KEYS = ("metric", "direction")
VALID_DIRECTIONS = ("minimize", "maximize")


def _validate_plan(plan: Any) -> Optional[str]:
    """Return None if valid, else an error string."""
    if not isinstance(plan, dict):
        return f"plan must be a dict, got {type(plan).__name__}"

    missing_top = [k for k in REQUIRED_PLAN_KEYS if k not in plan]
    if missing_top:
        return f"missing required top-level plan keys: {missing_top}"

    objective = plan.get("objective")
    if not isinstance(objective, dict):
        return "plan.objective must be a dict {metric, direction, units?, display_name?, target?}"
    missing_obj = [k for k in REQUIRED_OBJECTIVE_KEYS if k not in objective]
    if missing_obj:
        return f"missing required plan.objective keys: {missing_obj}"

    metric = objective.get("metric")
    if metric not in list_metrics():
        return f"metric {metric!r} not registered. Available: {list_metrics()}"

    direction = objective.get("direction")
    if direction not in VALID_DIRECTIONS:
        return f"plan.objective.direction must be one of {VALID_DIRECTIONS}, got {direction!r}"

    template = plan.get("template")
    if not isinstance(template, dict) or not template:
        return "plan.template must be a non-empty dict of template fields"

    knobs = plan.get("knobs")
    if not isinstance(knobs, dict) or not knobs:
        return "plan.knobs must be a non-empty dict {knob_name: [low, high]}"
    for k, rng in knobs.items():
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            return f"plan.knobs[{k!r}] must be [low, high], got {rng!r}"

    max_iters = plan.get("max_iterations")
    if not isinstance(max_iters, int) or max_iters < 1:
        return f"plan.max_iterations must be a positive int, got {max_iters!r}"

    return None


# ── Derived-summary helpers (best-so-far, iterations-since-improvement) ──────

def _metric_value(trial: Dict[str, Any], metric_name: str) -> Optional[float]:
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


def _summarize_trials(
    trials: List[Dict[str, Any]], metric_name: str, direction: str
) -> Dict[str, Any]:
    """Compute {best_so_far, iterations_since_improvement, consecutive_failures,
    next_iteration}. Skips failed trials (metric=None) when computing best."""
    best_val: Optional[float] = None
    best_trial: Optional[Dict[str, Any]] = None
    iters_since_improve = 0
    consecutive_failures = 0

    for t in trials:
        v = _metric_value(t, metric_name)
        if v is None:
            consecutive_failures += 1
            iters_since_improve += 1
            continue
        consecutive_failures = 0

        new_best = False
        if best_val is None:
            new_best = True
        elif direction == "maximize" and v > best_val:
            new_best = True
        elif direction == "minimize" and v < best_val:
            new_best = True

        if new_best:
            best_val = v
            best_trial = t
            iters_since_improve = 0
        else:
            iters_since_improve += 1

    return {
        "best_so_far": (
            {"iteration": best_trial.get("iteration"),
             "value": best_val,
             "knobs": best_trial.get("knobs")}
            if best_trial else None
        ),
        "iterations_since_improvement": iters_since_improve,
        "consecutive_failures": consecutive_failures,
        "next_iteration": (trials[-1]["iteration"] + 1) if trials else 1,
    }


# ────────────────────────────────────────────────────────────────────────────
# TOOL 1: campaign_start
# ────────────────────────────────────────────────────────────────────────────

@tool
def campaign_start(
    plan: dict,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Start a new campaign. Creates runs/<campaign_id>/, writes plan.json +
    control.json, emits campaign_started SSE, and stores campaign_id on
    orchestrator state so subsequent campaign_* tools find it automatically.

    Call this ONCE after the user confirms the plan in Turn 1. `plan` must be
    a dict with these required keys:
      - objective: {metric, direction[, units, display_name, target]}
      - template:  {material, system_size, kim_model, phase_1, ...}
      - knobs:     {knob_name: [low, high], ...}
      - max_iterations: int

    Optional keys: stop_criteria: {no_improvement_patience, convergence:
    {epsilon_pct, window}}.
    """
    try:
        err = _validate_plan(plan)
        if err:
            return Command(update={"messages": [ToolMessage(
                content=f"campaign_start rejected: {err}",
                tool_call_id=tool_call_id,
            )]})

        session_id = (state or {}).get("session_id")
        # Timestamp-based id — matches today's convention (benchmark_<utcts>).
        # Uses the same "benchmark_" prefix so the skill-detection heuristics
        # and downstream storage layout stay compatible.
        campaign_id = f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        folder = _campaign_folder(campaign_id)
        os.makedirs(folder, exist_ok=True)

        # Enrich the plan with server-side metadata before persisting
        plan = dict(plan)  # shallow copy so we don't mutate the LLM's arg
        plan["campaign_id"] = campaign_id
        plan["created_at"] = _now_iso()

        _write_json_atomic(_plan_path(campaign_id), plan)
        _write_json_atomic(_control_path(campaign_id),
                           {"status": "running", "updated_at": _now_iso()})

        objective = plan["objective"]
        _emit(session_id, "campaign_started",
              campaign_id=campaign_id,
              objective=objective,
              max_iterations=plan["max_iterations"],
              knobs=list(plan["knobs"].keys()))

        display = objective.get("display_name", objective["metric"])
        direction = objective["direction"]
        msg = (
            f"Campaign {campaign_id} started. "
            f"Objective: {direction} {display}. "
            f"Max iterations: {plan['max_iterations']}. "
            f"Knobs: {list(plan['knobs'].keys())}. "
            f"Next: call campaign_get_state() then propose iteration 1."
        )

        return Command(update={
            "campaign_id": campaign_id,
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
        })
    except Exception as e:
        return Command(update={"messages": [ToolMessage(
            content=f"campaign_start crashed: {type(e).__name__}: {e}",
            tool_call_id=tool_call_id,
        )]})


# ────────────────────────────────────────────────────────────────────────────
# TOOL 2: campaign_get_state
# ────────────────────────────────────────────────────────────────────────────

@tool
def campaign_get_state(
    state: Annotated[dict, InjectedState] = None,
) -> dict:
    """Return the current campaign state as a typed dict.

    Returns:
        {
          plan: {...},               # the immutable plan
          trials: [ {iteration, knobs, hypothesis, metrics, run_folder,
                     wall_clock_s, failure_reason, is_new_best, created_at}, ...],
          control: {status: "running"|"paused"|"cancelled"},
          best_so_far: {iteration, value, knobs} | null,
          iterations_since_improvement: int,
          consecutive_failures: int,
          next_iteration: int,      # what iteration number to run next
        }

    Call this at the START of every iteration to see plan.template, past
    trials with their hypotheses + metrics, and current best. Do NOT try to
    infer state from message history — always call this tool.
    """
    campaign_id = (state or {}).get("campaign_id")
    if not campaign_id:
        return {"error": "no active campaign on state. Call campaign_start(plan) first."}
    try:
        plan = _read_json(_plan_path(campaign_id))
        control = _read_json(_control_path(campaign_id))
        trials = _read_trials(campaign_id)
        metric_name = plan["objective"]["metric"]
        direction = plan["objective"]["direction"]
        summary = _summarize_trials(trials, metric_name, direction)
        return {
            "plan": plan,
            "trials": trials,
            "control": control,
            **summary,
        }
    except FileNotFoundError as e:
        return {"error": f"campaign files missing: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ────────────────────────────────────────────────────────────────────────────
# TOOL 3: campaign_run_simulation
# ────────────────────────────────────────────────────────────────────────────

@tool
def campaign_run_simulation(
    state: Annotated[dict, InjectedState] = None,
) -> dict:
    """Run the LAMMPS + OVITO scripts that simulate() just generated.

    Reads state.lammps_script / .ovito_script / .dump_filename_pattern /
    .target_timesteps / .camera_angle (all populated by the preceding
    simulate(auto_approve=True) call). Nests the run folder under the
    active campaign folder so per-iteration output lives at
    runs/<campaign_id>/run_<ts>/.

    Live trajectory frames and thermo data are streamed to the chat SSE
    queue during the run — the frontend renders them with the same
    handlers standalone runs use, so the user sees the same live UX
    (trajectory viewer + thermo plots) as a manual simulation.

    Call ONCE per iteration, AFTER simulate() and BEFORE
    campaign_measure_metrics(). Returns {ok, run_folder} on success or
    {error, run_folder: None} on failure — never raises.
    """
    campaign_id = (state or {}).get("campaign_id")
    if not campaign_id:
        return {"error": "no active campaign on state.", "run_folder": None}
    session_id  = (state or {}).get("session_id")
    script      = (state or {}).get("lammps_script")
    if not script:
        return {"error": "no lammps_script on state — call simulate() first",
                "run_folder": None}
    if not (state or {}).get("review_passed"):
        issues = (state or {}).get("review_issues")
        return {"error": f"review not passed: {issues}", "run_folder": None}

    render_config = {
        "ovito_script":          (state or {}).get("ovito_script"),
        "camera_angle":          (state or {}).get("camera_angle", "default"),
        "dump_filename_pattern": (state or {}).get("dump_filename_pattern", ""),
        "target_timesteps":      (state or {}).get("target_timesteps"),
    }

    # data_callback bounces LAMMPS thread events onto the chat SSE queue.
    # tools.run_lammps_simulation emits raw thermo dicts (no "type" field)
    # and typed events like trajectory_frame — wrap raw dicts so the
    # frontend's switch(data.type) handler picks them up.
    q = SESSION_QUEUES.get(session_id)
    def cb(evt):
        if q is None or not isinstance(evt, dict):
            return
        try:
            if "type" not in evt:
                q.put({"type": "thermo_data_point", **evt})
            else:
                q.put(evt)
        except Exception:
            pass

    try:
        import config
        from tools import run_lammps_simulation
        result = run_lammps_simulation(
            script,
            render_config,
            cb,
            num_cores=getattr(config, "LAMMPS_CORES", 1),
            parent_dir=_campaign_folder(campaign_id),
        )
    except Exception as e:
        return {"error": f"lammps failed: {type(e).__name__}: {e}",
                "run_folder": None}

    # run_lammps_simulation returns (output_files, all_thermo_data, frame_images).
    # Pull the run folder from output_files["log"].
    run_folder = ""
    try:
        output_files = result[0] if isinstance(result, tuple) else result
        log_path = output_files.get("log") if isinstance(output_files, dict) else None
        if log_path:
            run_folder = os.path.dirname(log_path)
    except Exception:
        pass

    return {"ok": True, "run_folder": run_folder}


# ────────────────────────────────────────────────────────────────────────────
# TOOL 4: campaign_measure_metrics
# ────────────────────────────────────────────────────────────────────────────

@tool
def campaign_measure_metrics(
    run_folder: str,
    metric_name: str,
) -> dict:
    """Extract the objective metric from a completed simulation's run folder.

    Dispatches to src/benchmark/metrics/ registry by metric_name. Currently
    registered: 'delta_a'. Returns the metric dict on success, or
    {"error": "..."} on failure (unknown metric name, missing dump files,
    extractor crash — never raises).

    Call this after simulate() returns, before record_trial().
    """
    return compute_metric(metric_name, run_folder)


# ────────────────────────────────────────────────────────────────────────────
# TOOL 4: campaign_record_trial
# ────────────────────────────────────────────────────────────────────────────

@tool
def campaign_record_trial(
    iteration: int,
    knobs: dict,
    hypothesis: str,
    metrics: Optional[dict] = None,
    run_folder: str = "",
    wall_clock_s: Optional[float] = None,
    failure_reason: Optional[str] = None,
    state: Annotated[dict, InjectedState] = None,
) -> dict:
    """Persist one trial to trials.jsonl. Emits campaign_iteration_complete
    (or campaign_iteration_failed if failure_reason is set) SSE.

    Always call this AFTER measuring metrics — even on failure, pass
    metrics=None + failure_reason="<description>" so the failure is captured
    as a first-class trial row instead of silently dropped.
    """
    campaign_id = (state or {}).get("campaign_id")
    if not campaign_id:
        return {"error": "no active campaign on state. Call campaign_start(plan) first."}
    try:
        plan = _read_json(_plan_path(campaign_id))
        metric_name = plan["objective"]["metric"]
        direction = plan["objective"]["direction"]

        # Decide is_new_best by comparing against existing trials
        prior = _read_trials(campaign_id)
        prior_best_val: Optional[float] = None
        for t in prior:
            v = _metric_value(t, metric_name)
            if v is None:
                continue
            if (prior_best_val is None or
                (direction == "maximize" and v > prior_best_val) or
                (direction == "minimize" and v < prior_best_val)):
                prior_best_val = v

        this_val = None
        if isinstance(metrics, dict):
            this_val = metrics.get(metric_name)
            try:
                this_val = float(this_val) if this_val is not None else None
            except (TypeError, ValueError):
                this_val = None

        is_new_best = False
        if this_val is not None:
            if prior_best_val is None:
                is_new_best = True
            elif direction == "maximize" and this_val > prior_best_val:
                is_new_best = True
            elif direction == "minimize" and this_val < prior_best_val:
                is_new_best = True

        trial = {
            "iteration": iteration,
            "knobs": knobs or {},
            "hypothesis": hypothesis or "",
            "metrics": metrics if isinstance(metrics, dict) else None,
            "run_folder": run_folder,
            "wall_clock_s": wall_clock_s,
            "failure_reason": failure_reason,
            "is_new_best": is_new_best,
            "created_at": _now_iso(),
        }
        _append_trial(campaign_id, trial)

        session_id = (state or {}).get("session_id")
        if failure_reason:
            _emit(session_id, "campaign_iteration_failed",
                  campaign_id=campaign_id,
                  iteration=iteration,
                  reason=failure_reason,
                  knobs=knobs)
        else:
            _emit(session_id, "campaign_iteration_complete",
                  campaign_id=campaign_id,
                  iteration=iteration,
                  knobs=knobs,
                  metrics=metrics,
                  metric_name=metric_name,
                  is_new_best=is_new_best,
                  run_folder=run_folder,
                  wall_clock_s=wall_clock_s)

        return {
            "ok": True,
            "iteration": iteration,
            "is_new_best": is_new_best,
            "recorded_to": _trials_path(campaign_id),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ────────────────────────────────────────────────────────────────────────────
# TOOL 5: campaign_check_stop
# ────────────────────────────────────────────────────────────────────────────

@tool
def campaign_check_stop(
    state: Annotated[dict, InjectedState] = None,
) -> dict:
    """Deterministic stop decision — call at the END of every iteration.

    Evaluates 6 criteria in order (first match wins):
      1. user_cancelled            — control.json says cancelled
      2. user_paused               — control.json says paused
      3. max_iterations            — trials count >= plan.max_iterations
      4. target_hit                — plan.objective.target set AND met
      5. consecutive_failures      — >= 3 trials in a row failed
      6. no_improvement            — iterations_since_improvement >= patience
      7. convergence               — best-value change over last window < epsilon_pct%

    Returns {should_stop: bool, reason: str, detail: str}.
    Emits campaign_paused / campaign_cancelled SSE if user-triggered.
    """
    campaign_id = (state or {}).get("campaign_id")
    if not campaign_id:
        return {"error": "no active campaign on state."}
    try:
        plan = _read_json(_plan_path(campaign_id))
        control = _read_json(_control_path(campaign_id))
        trials = _read_trials(campaign_id)
        metric_name = plan["objective"]["metric"]
        direction = plan["objective"]["direction"]
        objective = plan["objective"]
        stop_criteria = plan.get("stop_criteria") or {}
        session_id = (state or {}).get("session_id")

        # 1. user cancelled
        if control.get("status") == "cancelled":
            _emit(session_id, "campaign_cancelled", campaign_id=campaign_id)
            return {"should_stop": True, "reason": "user_cancelled",
                    "detail": "user cancelled the campaign"}

        # 2. user paused
        if control.get("status") == "paused":
            _emit(session_id, "campaign_paused", campaign_id=campaign_id)
            return {"should_stop": True, "reason": "user_paused",
                    "detail": "user paused the campaign; report NOT generated. "
                              "Resume later by re-invoking on the same session."}

        # 3. max iterations
        max_iters = plan.get("max_iterations", 20)
        if len(trials) >= max_iters:
            return {"should_stop": True, "reason": "max_iterations",
                    "detail": f"reached max_iterations={max_iters}"}

        summary = _summarize_trials(trials, metric_name, direction)
        best = summary["best_so_far"]

        # 4. target hit
        target = objective.get("target")
        if target is not None and best is not None:
            bv = best["value"]
            hit = ((direction == "minimize" and bv <= target) or
                   (direction == "maximize" and bv >= target))
            if hit:
                return {"should_stop": True, "reason": "target_hit",
                        "detail": f"best={bv} meets target={target}"}

        # 5. consecutive failures (3+)
        if summary["consecutive_failures"] >= 3:
            return {"should_stop": True, "reason": "consecutive_failures",
                    "detail": f"{summary['consecutive_failures']} failed trials in a row"}

        # 6. no improvement (patience)
        patience = stop_criteria.get("no_improvement_patience", 5)
        if summary["iterations_since_improvement"] >= patience:
            return {"should_stop": True, "reason": "no_improvement",
                    "detail": f"no new best in {patience} trials"}

        # 7. convergence (best value change over last N < epsilon_pct%)
        conv = stop_criteria.get("convergence") or {}
        window = int(conv.get("window", 0) or 0)
        eps_pct = float(conv.get("epsilon_pct", 0) or 0)
        if window >= 2 and eps_pct > 0 and best is not None:
            # look at the last <window> successful trials; if the range of best-so-far
            # over that window has changed by less than eps_pct%, call it converged.
            best_progression: List[float] = []
            running = None
            for t in trials:
                v = _metric_value(t, metric_name)
                if v is None:
                    continue
                if running is None:
                    running = v
                elif direction == "maximize" and v > running:
                    running = v
                elif direction == "minimize" and v < running:
                    running = v
                best_progression.append(running)
            if len(best_progression) >= window:
                tail = best_progression[-window:]
                lo, hi = min(tail), max(tail)
                denom = abs(tail[-1]) if abs(tail[-1]) > 1e-12 else 1.0
                change_pct = 100.0 * (hi - lo) / denom
                if change_pct < eps_pct:
                    return {"should_stop": True, "reason": "convergence",
                            "detail": f"best changed by {change_pct:.3g}% over last {window} trials (< {eps_pct}%)"}

        return {"should_stop": False, "reason": "continue",
                "detail": f"trials={len(trials)}, best={best['value'] if best else None}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ────────────────────────────────────────────────────────────────────────────
# TOOL 6: campaign_generate_report
# ────────────────────────────────────────────────────────────────────────────

@tool
def campaign_generate_report(
    stop_reason: str = "",
    state: Annotated[dict, InjectedState] = None,
) -> dict:
    """Render report.png (matplotlib) + write report.md skeleton (Python facts
    only). Returns the skeleton text and paths, plus a next_step hint telling
    the LLM to write the "What was learned" narrative section and call
    campaign_save_narrative to persist it.

    Args:
      stop_reason: The reason the campaign ended, from the last
                   campaign_check_stop call (e.g. "max_iterations",
                   "no_improvement"). Included in report.md.
    """
    campaign_id = (state or {}).get("campaign_id")
    if not campaign_id:
        return {"error": "no active campaign on state."}
    folder = _campaign_folder(campaign_id)
    try:
        png_path = render_campaign_graph(folder)
        md_path, skeleton = write_report_skeleton(folder, stop_reason=stop_reason or None)
        return {
            "ok": True,
            "report_md_path": md_path,
            "report_png_path": png_path,
            "skeleton": skeleton,
            "next_step": (
                "Now write the 'What was learned' narrative section as your "
                "chat message (streamed live to the user). Reference specific "
                "trials by number, discuss which hypotheses held up vs were "
                "refuted, note which knobs mattered most, and suggest next "
                "steps. Then call campaign_save_narrative(text=<your narrative>) "
                "to append it under the '## What was learned' heading in report.md."
            ),
        }
    except FileNotFoundError as e:
        return {"error": f"campaign files missing: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ────────────────────────────────────────────────────────────────────────────
# TOOL 7: campaign_save_narrative
# ────────────────────────────────────────────────────────────────────────────

@tool
def campaign_save_narrative(
    text: str,
    state: Annotated[dict, InjectedState] = None,
) -> dict:
    """Append the LLM-authored "What was learned" narrative under the
    placeholder heading in report.md. Emits campaign_complete SSE with the
    final artifact paths.

    Call this AFTER campaign_generate_report and AFTER you've written the
    narrative as your chat message. This is the last tool call of the
    campaign turn.
    """
    campaign_id = (state or {}).get("campaign_id")
    if not campaign_id:
        return {"error": "no active campaign on state."}
    folder = _campaign_folder(campaign_id)
    try:
        md_path = append_narrative(folder, text or "")
        png_path = os.path.join(folder, "report.png")
        # Also flip control.json to "done" so re-invocations know the campaign
        # is closed. Best-effort — don't fail if control.json is missing.
        try:
            _write_json_atomic(_control_path(campaign_id),
                               {"status": "done", "updated_at": _now_iso()})
        except Exception:
            pass

        session_id = (state or {}).get("session_id")
        _emit(session_id, "campaign_complete",
              campaign_id=campaign_id,
              report_md_path=md_path,
              report_png_path=png_path if os.path.exists(png_path) else None)

        return {
            "ok": True,
            "report_md_path": md_path,
            "report_png_path": png_path if os.path.exists(png_path) else None,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ── Exports ──────────────────────────────────────────────────────────────────

CAMPAIGN_TOOLS = [
    campaign_start,
    campaign_get_state,
    campaign_run_simulation,
    campaign_measure_metrics,
    campaign_record_trial,
    campaign_check_stop,
    campaign_generate_report,
    campaign_save_narrative,
]
