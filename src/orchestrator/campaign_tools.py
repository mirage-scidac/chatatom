"""Campaign v2 tools — the orchestrator drives the whole loop by calling
these directly. No daemon thread, no _DualQueue, no CampaignManager.

All tools live here and are exported via CAMPAIGN_TOOLS for registration
in ORCHESTRATOR_TOOLS. Every tool is failure-safe: outer try/except returns
{"error": "..."} instead of raising, so a bad tool call never poisons the
ReAct trace.

Script generation happens ONCE per campaign. Iteration 1 calls simulate() with
prose demanding a parameterized script, then campaign_propose_template() stages
it for the user to approve. Every iteration after that calls
campaign_fill_template(), which is pure string substitution into the approved
template's variable block — no LLM, and therefore no script drift between
trials.

State is persisted to disk under runs/<campaign_id>/:
  plan.json         (immutable, written once by campaign_start)
  control.json      (pause/resume/cancel flag)
  trials.jsonl      (append-only, one JSON line per iteration)
  template.lammps   (approved parameterized script; .pending before approval)
  template_ovito.py (approved OVITO script, knob-independent)
  template.json     (variable defaults, knob names, dump pattern, camera)

SSE emission goes through the existing _q helper in graph.nodes so the
frontend's existing handlers pick up the events unchanged.
"""
import asyncio
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
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC_DIR)
# Repo root (parent of src/) — cwd for the metric subprocess so its
# `os.getcwd()/src` on sys.path resolves to this project's src/.
PROJECT_ROOT = os.path.dirname(_SRC_DIR)

from benchmark.metrics import compute_metric, list_metrics, get_metric_meta
from benchmark.reporting import (
    append_narrative,
    render_campaign_graph,
    write_report_skeleton,
)
from graph.nodes import SESSION_QUEUES
from orchestrator.campaign_template import (  # noqa: F401  (re-exported)
    VAR_BLOCK_BEGIN,
    VAR_BLOCK_END,
    check_knob_bounds,
    extract_variable_block,
    fill_variable_block,
    parse_variable_block,
    resolve_placeholders,
)


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


# ── Template path helpers ────────────────────────────────────────────────────
# The parameterized LAMMPS script is generated ONCE (iteration 1) and approved
# by the user. Pre-approval copies carry a `.pending` suffix; the approval
# endpoint in app.py promotes them in place.

def _template_path(campaign_id: str, pending: bool = False) -> str:
    name = "template.lammps.pending" if pending else "template.lammps"
    return os.path.join(_campaign_folder(campaign_id), name)


def _template_ovito_path(campaign_id: str, pending: bool = False) -> str:
    name = "template_ovito.py.pending" if pending else "template_ovito.py"
    return os.path.join(_campaign_folder(campaign_id), name)


def _template_meta_path(campaign_id: str, pending: bool = False) -> str:
    name = "template.json.pending" if pending else "template.json"
    return os.path.join(_campaign_folder(campaign_id), name)


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
    seen = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trial = json.loads(line)
            # Skip a row that repeats one already read. An interrupted or
            # re-invoked turn can call campaign_record_trial twice for the same
            # finished run; the extra row then inflates len(trials), which
            # check_stop compares against max_iterations — ending the campaign
            # early — and which the report prints as the trial count.
            #
            # Keyed on (iteration, run_folder), not iteration alone: two loops
            # racing on the same campaign can hand the same iteration number to
            # two DIFFERENT runs, and both of those are real trials. Rows with
            # no run_folder (failed before LAMMPS started) are never collapsed,
            # since there is nothing to tell two such failures apart.
            key = (trial.get("iteration"), trial.get("run_folder"))
            if trial.get("run_folder") and key in seen:
                continue
            seen.add(key)
            out.append(trial)
    return out


def _append_trial(campaign_id: str, trial: Dict[str, Any]) -> None:
    """Append one JSON line to trials.jsonl."""
    with open(_trials_path(campaign_id), "a") as f:
        f.write(json.dumps(trial, default=str) + "\n")


# ── Campaign lifecycle status ────────────────────────────────────────────────

# Terminal statuses. A campaign in one of these is finished: no template may be
# approved into it and no loop may re-enter it. "cancelled" is written by the
# cancel endpoint, "done" by campaign_save_narrative.
CLOSED_STATUSES = ("cancelled", "done")

PENDING_SUFFIX = ".pending"
REJECTED_SUFFIX = ".rejected"


def campaign_closed_status(campaign_id: str) -> Optional[str]:
    """Return the terminal status of a closed campaign, else None.

    Both campaign_propose_template and the /template approval endpoint must
    refuse a closed campaign, and they must refuse it on the same grounds — a
    guard living in only one of them is how an approval could resurrect a
    campaign the user had already cancelled. Shared here so the two cannot
    drift apart again.

    A missing or unreadable control.json reads as not-closed: the absence of a
    cancel flag is not evidence of one, and refusing on a read error would
    strand a healthy campaign.
    """
    path = _control_path(campaign_id)
    if not os.path.exists(path):
        return None
    try:
        status = _read_json(path).get("status")
    except (OSError, ValueError):
        return None
    return status if status in CLOSED_STATUSES else None


def reject_pending_template(campaign_id: str) -> List[str]:
    """Move a staged template out of the approval path. Returns the new paths.

    Called when a campaign is cancelled. Everything that looks for a staged
    template looks for `.pending`, so renaming to `.rejected` both takes it out
    of the approval path and makes _template_status report "none" instead of
    claiming a dead campaign is still awaiting approval.

    Renamed rather than deleted: the staged script is the only copy of what the
    campaign proposed — plan.json keeps the objective and the knob ranges, not
    the LAMMPS — and a refused template is worth being able to read afterwards.

    A no-op when nothing is staged: cancel is also reachable from the campaign
    banner mid-run, where the template was approved long ago.
    """
    moved: List[str] = []
    for path in (_template_path(campaign_id, pending=True),
                 _template_ovito_path(campaign_id, pending=True),
                 _template_meta_path(campaign_id, pending=True)):
        if not os.path.exists(path):
            continue
        dest = path[:-len(PENDING_SUFFIX)] + REJECTED_SUFFIX
        os.replace(path, dest)
        moved.append(dest)
    return moved


# ── The campaign variable block ──────────────────────────────────────────────
# Parsing/filling lives in campaign_template.py (stdlib-only, so the sweep logic
# stays testable without the langchain/langgraph stack). Re-exported here
# because callers and tests reach for them through this module.


def _read_template(campaign_id: str) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Return (lammps_template, ovito_script, meta) for an APPROVED template.

    Returns (None, None, {}) when no approved template exists yet.
    """
    path = _template_path(campaign_id)
    if not os.path.exists(path):
        return None, None, {}
    with open(path) as f:
        lammps = f.read()
    ovito = None
    ovito_path = _template_ovito_path(campaign_id)
    if os.path.exists(ovito_path):
        with open(ovito_path) as f:
            ovito = f.read()
    meta: Dict[str, Any] = {}
    meta_path = _template_meta_path(campaign_id)
    if os.path.exists(meta_path):
        try:
            meta = _read_json(meta_path)
        except Exception:
            meta = {}
    return lammps, ovito, meta


def _template_status(campaign_id: str) -> Dict[str, Any]:
    """Summarize template state for campaign_get_state.

    status is one of "none" (iteration 1 has not generated one yet),
    "pending" (generated, awaiting the user's approval), or "approved".
    """
    if os.path.exists(_template_path(campaign_id)):
        lammps, _, meta = _read_template(campaign_id)
        try:
            variables, _ = parse_variable_block(lammps or "")
        except ValueError:
            variables = meta.get("variables") or {}
        return {
            "status": "approved",
            "variables": variables,
            "knob_vars": meta.get("knob_vars") or [],
            "approved_at": meta.get("approved_at"),
            "source_iteration": meta.get("source_iteration"),
        }
    if os.path.exists(_template_path(campaign_id, pending=True)):
        meta = {}
        try:
            meta = _read_json(_template_meta_path(campaign_id, pending=True))
        except Exception:
            pass
        return {
            "status": "pending",
            "variables": meta.get("variables") or {},
            "knob_vars": meta.get("knob_vars") or [],
        }
    return {"status": "none", "variables": {}, "knob_vars": []}


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
          control: {status: "running"|"paused"|"cancelled"
                            |"awaiting_template_approval"},
          template: {status: "none"|"pending"|"approved",
                     variables: {name: value}, knob_vars: [...]},
          best_so_far: {iteration, value, knobs} | null,
          iterations_since_improvement: int,
          consecutive_failures: int,
          next_iteration: int,      # what iteration number to run next
        }

    Call this at the START of every iteration to see plan.template, past
    trials with their hypotheses + metrics, and current best. Do NOT try to
    infer state from message history — always call this tool.

    `template.status` tells you which branch of the loop you are in:
      "none"     → iteration 1: call simulate() then campaign_propose_template()
      "pending"  → the user has not approved the template yet; do not run
      "approved" → call campaign_fill_template(knobs) — never simulate()
    On the first iteration after approval, use template.variables as the knob
    values: those are the values in the script the user approved.
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

        # Push the campaign's disk truth to the UI on every iteration so the
        # banner and progress table can rebuild themselves. The frontend only
        # learns about a campaign from the one-shot campaign_started event, so
        # anything that skips campaign_start — a resume, a page reload, the
        # turn after a template approval — would otherwise leave it with no
        # activeCampaign and silently drop the banner, the progress table and
        # the pause/cancel controls for the rest of the run.
        _emit((state or {}).get("session_id"), "campaign_state",
              campaign_id=campaign_id,
              objective=plan.get("objective") or {},
              max_iterations=plan.get("max_iterations"),
              knobs=list((plan.get("knobs") or {}).keys()),
              status=control.get("status", "running"),
              metric_name=metric_name,
              trials=trials)

        return {
            "plan": plan,
            "trials": trials,
            "control": control,
            "template": _template_status(campaign_id),
            **summary,
        }
    except FileNotFoundError as e:
        return {"error": f"campaign files missing: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ────────────────────────────────────────────────────────────────────────────
# TOOL 2a: campaign_propose_template
# ────────────────────────────────────────────────────────────────────────────

def _emit_staged_template(
    campaign_id: str, session_id: Optional[str], reshown: bool = False
) -> Dict[str, Any]:
    """Send the staged (pending) template to the UI and describe the next step.

    Always reads back from disk rather than from state, so re-showing a card the
    user missed — a page reload, a restarted server, a dropped SSE — produces
    exactly the bytes that are awaiting approval.
    """
    path = _template_path(campaign_id, pending=True)
    if not os.path.exists(path):
        return {"error": "no template is staged for approval."}
    with open(path) as f:
        script = f.read()

    ovito = None
    ovito_path = _template_ovito_path(campaign_id, pending=True)
    if os.path.exists(ovito_path):
        with open(ovito_path) as f:
            ovito = f.read()

    meta: Dict[str, Any] = {}
    try:
        meta = _read_json(_template_meta_path(campaign_id, pending=True))
    except Exception:
        pass

    variables, _ = parse_variable_block(script)
    knob_names = meta.get("knob_vars") or []

    _emit(session_id, "campaign_template_ready",
          campaign_id=campaign_id,
          lammps_template=script,
          ovito_script=ovito,
          variables=variables,
          knob_vars=knob_names)

    return {
        "ok": True,
        "status": "awaiting_approval",
        "reshown": reshown,
        "variables": variables,
        "knob_vars": knob_names,
        "next_step": (
            "END YOUR TURN NOW. Tell the user the template is "
            + ("shown again above" if reshown else "ready for review")
            + " and that approving it starts iteration 1. Do NOT call "
              "campaign_run_simulation — the template is not approved yet."
        ),
    }


@tool
def campaign_propose_template(
    source_iteration: int = 1,
    state: Annotated[dict, InjectedState] = None,
) -> dict:
    """Capture the script simulate() just generated as the campaign template.

    Call this ONCE, immediately after the iteration-1 simulate() call, and
    NEVER again for the same campaign. It validates that the generated script
    is properly parameterized, stages it for approval, and emits the
    campaign_template_ready card to the user.

    After this returns ok, END YOUR TURN. Do not run the simulation — the
    template is not approved yet. The card is READ-ONLY with two buttons:
    Approve sends a "Template approved" chat message which resumes the loop,
    Cancel ends the campaign. The script cannot be edited or regenerated, so
    check it against plan.template BEFORE calling this tool.

    Returns {ok, status, variables, knob_vars, next_step} on success, or
    {"error": "..."} describing exactly what is wrong with the generated
    script — call simulate() ONCE more with that error quoted in
    simulation_details, then retry this tool.
    """
    campaign_id = (state or {}).get("campaign_id")
    if not campaign_id:
        return {"error": "no active campaign on state. Call campaign_start(plan) first."}
    try:
        # A closed campaign stays closed. Without this guard the recovery path
        # below would re-show the approval card on any later re-invocation,
        # offering to start a campaign the user already finished or refused.
        closed = campaign_closed_status(campaign_id)
        if closed:
            return {"error": f"this campaign is {closed} — it cannot be "
                             f"restarted. Call campaign_start(plan) to begin a "
                             f"new one."}

        if os.path.exists(_template_path(campaign_id)):
            return {"error": "this campaign already has an approved template — "
                             "call campaign_fill_template(knobs) instead."}

        session_id = (state or {}).get("session_id")
        script = (state or {}).get("lammps_script")
        staged = os.path.exists(_template_path(campaign_id, pending=True))

        # Recovery path: a template is already staged but no fresh script is on
        # state (new turn, restarted server, or the user reloaded the page and
        # missed the card's SSE event). Re-show what is on disk instead of
        # regenerating — the staged script is still the one awaiting approval.
        if staged and not script:
            return _emit_staged_template(campaign_id, session_id, reshown=True)

        if not script:
            return {"error": "no lammps_script on state — call simulate() first."}
        if not (state or {}).get("review_passed"):
            issues = (state or {}).get("review_issues")
            return {"error": f"generated script did not pass review: {issues}"}

        plan = _read_json(_plan_path(campaign_id))

        try:
            variables, _ = parse_variable_block(script)
        except ValueError as e:
            return {"error": f"script is not parameterized: {e}",
                    "fix": "Re-call simulate() and restate the TEMPLATE FORMAT "
                           "requirements in simulation_details."}

        knob_names = list((plan.get("knobs") or {}).keys())
        missing = [k for k in knob_names if k not in variables]
        if missing:
            return {"error": f"the campaign variable block is missing a declaration "
                             f"for swept knob(s) {missing}. Declared: {sorted(variables)}",
                    "fix": "Re-call simulate() and require a 'variable <knob> equal "
                           "<value>' line for every swept knob."}

        initial_knobs = {k: variables[k] for k in knob_names}
        problems = check_knob_bounds(plan, initial_knobs)
        if problems:
            return {"error": "initial knob values in the template are out of range: "
                             + "; ".join(problems)}

        # The OVITO script is plain Python, stored once and reused verbatim —
        # nothing fills it per iteration. A ${...} left in it is a SyntaxError
        # that makes the renderer fall back to default coloring, silently
        # costing the CNA crack colors. Resolve the constants, reject the rest.
        ovito = (state or {}).get("ovito_script")
        if ovito:
            ovito, ovito_problems = resolve_placeholders(
                ovito, variables, swept=knob_names
            )
            if ovito_problems:
                return {"error": "the OVITO script " + "; ".join(ovito_problems),
                        "fix": "Re-call simulate() and state that the OVITO "
                               "script is Python and must use literal numbers "
                               "— only the LAMMPS script uses ${...}."}

        meta = {
            "variables":             variables,
            "knob_vars":             knob_names,
            "dump_filename_pattern": (state or {}).get("dump_filename_pattern") or "",
            "target_timesteps":      (state or {}).get("target_timesteps"),
            "camera_angle":          (state or {}).get("camera_angle") or "default",
            "source_iteration":      source_iteration,
            "proposed_at":           _now_iso(),
            "approved_at":           None,
        }

        with open(_template_path(campaign_id, pending=True), "w") as f:
            f.write(script)
        if ovito:
            with open(_template_ovito_path(campaign_id, pending=True), "w") as f:
                f.write(ovito)
        _write_json_atomic(_template_meta_path(campaign_id, pending=True), meta)

        # Park the campaign until the user acts on the card. check_stop treats
        # anything that isn't "paused"/"cancelled" as runnable, so this status
        # is purely informational for resume — it does not stop the loop.
        _write_json_atomic(_control_path(campaign_id),
                           {"status": "awaiting_template_approval",
                            "updated_at": _now_iso()})

        return _emit_staged_template(campaign_id, session_id)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ────────────────────────────────────────────────────────────────────────────
# TOOL 2b: campaign_fill_template
# ────────────────────────────────────────────────────────────────────────────

@tool
def campaign_fill_template(
    knobs: dict,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Substitute this iteration's knob values into the approved template.

    This REPLACES simulate() for every iteration once a template exists — no
    LLM runs, so the script is byte-identical to previous trials outside the
    variable block. Writes the filled script onto state exactly as simulate()
    would, so campaign_run_simulation() picks it up unchanged.

    Args:
        knobs: {knob_name: value} for THIS iteration. Every name must be a
               swept knob from plan.knobs and every value must lie inside its
               planned range — out-of-range values are rejected, not clamped,
               so a mistaken proposal never runs silently. Knobs you omit keep
               the template's current value.

    Call sequence per iteration: campaign_get_state → (write hypothesis) →
    campaign_fill_template → campaign_run_simulation → campaign_measure_metrics
    → campaign_record_trial → campaign_check_stop.
    """
    def _fail(msg: str) -> Command:
        return Command(update={"messages": [ToolMessage(
            content=f"campaign_fill_template failed: {msg}",
            tool_call_id=tool_call_id,
        )]})

    campaign_id = (state or {}).get("campaign_id")
    if not campaign_id:
        return _fail("no active campaign on state. Call campaign_start(plan) first.")
    try:
        template, ovito, meta = _read_template(campaign_id)
        if not template:
            if os.path.exists(_template_path(campaign_id, pending=True)):
                return _fail("the template is still awaiting user approval — "
                             "end your turn and wait for the approval message.")
            return _fail("no template yet. Run iteration 1's simulate() and "
                         "campaign_propose_template() first.")

        if not isinstance(knobs, dict) or not knobs:
            return _fail("knobs must be a non-empty dict of {knob_name: value}.")

        plan = _read_json(_plan_path(campaign_id))
        planned = list((plan.get("knobs") or {}).keys())

        # Only swept knobs may vary. Everything else in the block (a0, timestep,
        # dump frequency) is a campaign constant: letting it drift mid-run would
        # silently break comparability with the trials already recorded.
        unknown = [k for k in knobs if k not in planned]
        if unknown:
            return _fail(f"{unknown} are not swept knobs for this campaign. "
                         f"Tunable knobs are {planned}.")

        problems = check_knob_bounds(plan, knobs)
        if problems:
            return _fail("; ".join(problems) + ". Pick values inside the planned "
                         "ranges — they are not clamped for you.")

        try:
            values = {k: float(v) for k, v in knobs.items()}
            filled = fill_variable_block(template, values)
        except ValueError as e:
            return _fail(str(e))

        # Read back what actually landed in the script rather than trusting the
        # requested values, then show it to the user. The substitution is the
        # one step of an iteration with no LLM and no reviewer in it, so the
        # filled block is the only evidence that this trial ran the numbers the
        # hypothesis claimed.
        applied, _ = parse_variable_block(filled)
        print(f"[campaign_fill_template] {campaign_id} knobs={values}", flush=True)

        _emit((state or {}).get("session_id"), "campaign_iteration_script",
              campaign_id=campaign_id,
              iteration=len(_read_trials(campaign_id)) + 1,
              knobs=values,
              knob_vars=planned,
              variable_block=extract_variable_block(filled),
              lammps_script=filled)

        return Command(update={
            "lammps_script":         filled,
            "ovito_script":          ovito,
            "dump_filename_pattern": meta.get("dump_filename_pattern") or "",
            "target_timesteps":      meta.get("target_timesteps"),
            "camera_angle":          meta.get("camera_angle") or "default",
            # The template was reviewed and approved by the user once; a value
            # substitution cannot invalidate it, so no reviewer runs per trial.
            "review_passed":         True,
            "review_issues":         None,
            "auto_approve":          True,
            "execution_target":      (state or {}).get("execution_target") or "local",
            "has_custom_viz":        bool(ovito),
            "campaign_knobs":        values,
            "messages": [ToolMessage(
                content=(f"Template filled with {values}. "
                         f"Full variable block now: {applied}. "
                         f"Next: call campaign_run_simulation()."),
                tool_call_id=tool_call_id,
            )],
        })
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


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

# Inline program run by the metric subprocess: dispatch through the registry and
# print the result dict as JSON on stdout. Debug/warnings go to stderr, so stdout
# is clean JSON. Runs in its own process → Qt/OVITO gets a real main thread.
_METRIC_SUBPROC_SRC = (
    "import sys, json, os; "
    "sys.path.insert(0, os.path.join(os.getcwd(), 'src')); "
    "from benchmark.metrics import compute_metric; "
    "print(json.dumps(compute_metric(sys.argv[1], sys.argv[2])))"
)


async def _measure_in_subprocess(metric_name: str, run_folder: str) -> dict:
    """Run compute_metric in a child process and return its dict.

    OVITO/Qt deadlocks on worker threads and blocks the event loop when run
    inline, so we isolate it in a subprocess and await it — the loop stays free
    to service the Socket.IO/SSE heartbeat. Never raises: any failure (non-zero
    exit, unparseable output, timeout) is returned as {"error": ...} so the
    campaign loop records a failed trial and continues, matching compute_metric's
    own contract.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-W", "ignore", "-c", _METRIC_SUBPROC_SRC,
            metric_name, run_folder,
            cwd=PROJECT_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        return {"error": f"metric subprocess timed out after 120s"}
    except Exception as e:
        return {"error": f"metric subprocess failed to launch: {type(e).__name__}: {e}"}

    if proc.returncode != 0:
        tail = (err or b"").decode(errors="replace").strip()[-500:]
        return {"error": f"metric subprocess exit {proc.returncode}: {tail}"}
    try:
        return json.loads((out or b"").decode().strip())
    except Exception as e:
        tail = (out or b"").decode(errors="replace").strip()[-500:]
        return {"error": f"metric subprocess bad JSON ({e}): {tail}"}


@tool
async def campaign_measure_metrics(
    run_folder: str,
    metric_name: str,
) -> dict:
    """Extract the objective metric from a completed simulation's run folder.

    Dispatches to src/benchmark/metrics/ registry by metric_name. Currently
    registered: 'delta_cavity_surface_area'. Returns the metric dict on success, or
    {"error": "..."} on failure (unknown metric name, missing dump files,
    extractor crash — never raises).

    Call this after simulate() returns, before record_trial().
    """
    # OVITO/Qt cannot run off the main thread: asyncio.to_thread(compute_metric)
    # deadlocks forever, and running it inline on the loop thread blocks the
    # Socket.IO/SSE heartbeat for ~2s (frontend "Could not connect"). Fix: run the
    # metric in a SEPARATE PROCESS so Qt gets its own main thread, and await it so
    # this coroutine yields to the event loop while it runs.
    result = await _measure_in_subprocess(metric_name, run_folder)

    # Print the raw metric to the terminal so the crack area/volume are visible
    # while a campaign runs.
    if isinstance(result, dict) and "error" not in result:
        print(
            f"[metric:{metric_name}] {os.path.basename(run_folder)}  "
            f"crack_area={result.get('cavity_surface_area_final', 0.0):.1f} Å²  "
            f"Δarea={result.get('delta_cavity_surface_area', 0.0):.1f} Å²  "
            f"(crack_volume={result.get('cavity_volume_final', 0.0):.1f} Å³)",
            flush=True,
        )
    else:
        print(f"[metric:{metric_name}] {run_folder}  ERROR: {result}", flush=True)

    return result


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

    Evaluates 5 active criteria in order (first match wins); #6–#7 disabled below:
      1. user_cancelled            — control.json says cancelled
      2. user_paused               — control.json says paused
      3. max_iterations            — trials count >= plan.max_iterations
      4. target_hit                — plan.objective.target set AND met
      5. consecutive_failures      — >= 3 trials in a row failed
    #   6. no_improvement            — iterations_since_improvement >= patience
    #   7. convergence               — best-value change over last window < epsilon_pct%

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

        # ── #6 no_improvement and #7 convergence are COMMENTED OUT (not deleted)
        #    for now. The Δa metric can pin at 0.0 (metric saturation /
        #    dislocation-span contamination), which makes both plateau-stops fire
        #    on a metric artifact rather than real convergence, cutting campaigns
        #    short. Campaigns now run to max_iterations (still respecting user
        #    pause/cancel, target_hit, and consecutive_failures above).
        #    Re-enable these once the crack metric is trustworthy.
        #
        # # 6. no improvement (patience)
        # patience = stop_criteria.get("no_improvement_patience", 5)
        # if summary["iterations_since_improvement"] >= patience:
        #     return {"should_stop": True, "reason": "no_improvement",
        #             "detail": f"no new best in {patience} trials"}
        #
        # # 7. convergence (best value change over last N < epsilon_pct%)
        # conv = stop_criteria.get("convergence") or {}
        # window = int(conv.get("window", 0) or 0)
        # eps_pct = float(conv.get("epsilon_pct", 0) or 0)
        # if window >= 2 and eps_pct > 0 and best is not None:
        #     # look at the last <window> successful trials; if the range of best-so-far
        #     # over that window has changed by less than eps_pct%, call it converged.
        #     best_progression: List[float] = []
        #     running = None
        #     for t in trials:
        #         v = _metric_value(t, metric_name)
        #         if v is None:
        #             continue
        #         if running is None:
        #             running = v
        #         elif direction == "maximize" and v > running:
        #             running = v
        #         elif direction == "minimize" and v < running:
        #             running = v
        #         best_progression.append(running)
        #     if len(best_progression) >= window:
        #         tail = best_progression[-window:]
        #         lo, hi = min(tail), max(tail)
        #         denom = abs(tail[-1]) if abs(tail[-1]) > 1e-12 else 1.0
        #         change_pct = 100.0 * (hi - lo) / denom
        #         if change_pct < eps_pct:
        #             return {"should_stop": True, "reason": "convergence",
        #                     "detail": f"best changed by {change_pct:.3g}% over last {window} trials (< {eps_pct}%)"}

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


# ────────────────────────────────────────────────────────────────────────────
# TOOL 9: campaign_visualize_all (opt-in, called AFTER save_narrative)
# ────────────────────────────────────────────────────────────────────────────

@tool
def campaign_visualize_all(
    state: Annotated[dict, InjectedState] = None,
) -> dict:
    """Generate per-iteration visual analysis for every trial in the
    campaign. For each iteration:
      - render pristine + seeded-crack (pre-loading) + last-load PNGs via OVITO
      - send them to a vision LLM
      - collect a short structural summary

    Aggregates into ONE runs/<campaign_id>/viz_report.md with all
    iterations in sequence: images side-by-side + text per iter.

    Only call this when the user explicitly asks for visual analysis
    AFTER the campaign has completed (report already generated).

    Cost: ~15 seconds per iteration (renders + vision call). Sequential.
    """
    campaign_id = (state or {}).get("campaign_id")
    if not campaign_id:
        return {"error": "no active campaign on state."}
    session_id = (state or {}).get("session_id")

    try:
        trials = _read_trials(campaign_id)
        if not trials:
            return {"error": "no trials to visualize"}

        # Import helpers from the two different tools.py files:
        #   - src/tools.py            → OVITO render server helpers
        #   - src/orchestrator/tools.py → viz frame selection
        # (Same-name file gotcha — kept explicit to prevent regressions.)
        # Campaign uses its OWN prompt (CAMPAIGN_VIZ_PROMPT from prompts.py) so it
        # can diverge from the standalone prompt later (e.g. crack-specific output
        # for hypothesis generation) without affecting visualize_run.
        from tools import _start_render_server_process, _send_render_request
        from orchestrator.tools import _pick_viz_frames
        from prompts import CAMPAIGN_VIZ_PROMPT
        from openai import OpenAI
        import base64
        import config

        campaign_folder = _campaign_folder(campaign_id)
        viz_root = os.path.join(campaign_folder, "viz_frames")
        os.makedirs(viz_root, exist_ok=True)

        # One shared render server for all iterations
        proc = _start_render_server_process()
        client = OpenAI(api_key=config.API_KEY, base_url=config.API_BASE_URL)
        model = getattr(config, "ORCHESTRATOR_MODEL", "") or config.DEFAULT_MODEL

        aggregated_lines = [
            f"# Visual Analysis — Campaign {campaign_id}",
            "",
            f"_{len(trials)} iterations analyzed. Generated on request after campaign completion._",
            "",
            "---",
            "",
        ]

        try:
            for trial in trials:
                iter_num = trial.get("iteration")
                run_folder = trial.get("run_folder")
                if not run_folder or not os.path.isdir(run_folder):
                    aggregated_lines += [f"## Iteration {iter_num}", "",
                                         "_run folder missing — skipped_", "", "---", ""]
                    _emit(session_id, "campaign_viz_iteration",
                          campaign_id=campaign_id, iteration=iter_num,
                          status="skipped_missing_folder")
                    continue

                _emit(session_id, "campaign_viz_iteration",
                      campaign_id=campaign_id, iteration=iter_num, status="rendering")

                picks = _pick_viz_frames(run_folder)
                if not picks:
                    aggregated_lines += [f"## Iteration {iter_num}", "",
                                         "_no dump files — skipped_", "", "---", ""]
                    continue

                ovito_script = None
                ovito_path = os.path.join(run_folder, "ovito_script.py")
                if os.path.exists(ovito_path):
                    with open(ovito_path) as f:
                        ovito_script = f.read()

                # Render into per-iter subfolder
                iter_dir = os.path.join(viz_root, f"iter_{iter_num}")
                os.makedirs(iter_dir, exist_ok=True)
                rendered = []
                for label, dump_path in picks:
                    resp = _send_render_request(proc, {
                        "dump_file":        os.path.abspath(dump_path),
                        "ovito_script":     ovito_script,
                        "camera_angle":     "top",
                        "render_size":      (800, 600),
                        "file_frame_index": 0,
                    })
                    if resp.get("status") != "ok":
                        continue
                    b64 = resp["image"]
                    png_path = os.path.join(iter_dir, f"{label}.png")
                    with open(png_path, "wb") as f:
                        f.write(base64.b64decode(b64))
                    rendered.append({"label": label, "png_path": png_path, "b64": b64})

                if not rendered:
                    aggregated_lines += [f"## Iteration {iter_num}", "",
                                         "_render failed_", "", "---", ""]
                    continue

                _emit(session_id, "campaign_viz_iteration",
                      campaign_id=campaign_id, iteration=iter_num, status="analyzing")

                # Vision call
                content = []
                for f in rendered:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{f['b64']}"},
                    })
                if ovito_script:
                    content.append({
                        "type": "text",
                        "text": ("The atom colors were assigned by the following OVITO "
                                 "script. Interpret colors per its color_map:\n\n"
                                 "```python\n" + ovito_script + "\n```"),
                    })
                else:
                    content.append({
                        "type": "text",
                        "text": ("No custom OVITO script was used. Atoms are colored by "
                                 "LAMMPS particle type (NOT crystal structure). Do not "
                                 "interpret colors as FCC/HCP/BCC. Focus on gross spatial "
                                 "features."),
                    })
                content.append({"type": "text", "text": CAMPAIGN_VIZ_PROMPT})

                try:
                    # Stream is REQUIRED by the gateway for requests it estimates
                    # may exceed 10 min (non-streaming ones are rejected with a
                    # 500 "Streaming is required"). Vision-over-many-images trips
                    # that threshold, so stream and accumulate the tokens (the
                    # report is built from the full text, not shown live).
                    stream = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": content}],
                        temperature=0.3,
                        stream=True,
                    )
                    parts = []
                    for chunk in stream:
                        tok = chunk.choices[0].delta.content
                        if tok:
                            parts.append(tok)
                    observation = "".join(parts).strip()
                except Exception as e:
                    observation = f"_vision call failed: {e}_"

                # Append this iteration's section
                img_cells, caption_cells = [], []
                for f in rendered:
                    rel = os.path.relpath(f["png_path"], campaign_folder)
                    img_cells.append(
                        f'<td align="center"><img src="{rel}" alt="{f["label"]}" width="280"></td>'
                    )
                    caption_cells.append(
                        f'<td align="center"><em>{f["label"]}</em></td>'
                    )
                knobs_str = ", ".join(f"{k}={v}" for k, v in (trial.get("knobs") or {}).items())
                aggregated_lines += [
                    f"## Iteration {iter_num}",
                    "",
                    f"**Knobs:** {knobs_str}",
                    "",
                    "<table>",
                    "  <tr>" + "".join(img_cells) + "</tr>",
                    "  <tr>" + "".join(caption_cells) + "</tr>",
                    "</table>",
                    "",
                    observation,
                    "",
                    "---",
                    "",
                ]
        finally:
            try:
                proc.stdin.close(); proc.wait(timeout=5)
            except Exception:
                proc.kill()

        # Write the aggregated report (both .md and .html for Chrome)
        import html as _html
        viz_md_path = os.path.join(campaign_folder, "viz_report.md")
        with open(viz_md_path, "w") as fout:
            fout.write("\n".join(aggregated_lines))

        # HTML version: wrap the same markdown-safe body (which uses HTML
        # tables + text) in a minimal <html> shell so Chrome renders it
        # natively without a markdown-viewer extension.
        html_style = (
            "<style>"
            "body{font-family:-apple-system,system-ui,sans-serif;"
            "max-width:1200px;margin:20px auto;padding:0 20px;color:#222;}"
            "h1{color:#333;} h2{color:#555;margin-top:40px;}"
            "table{border-collapse:collapse;margin:20px 0;}"
            "td{padding:10px;} img{border:1px solid #ccc;border-radius:4px;}"
            "em{color:#666;} hr{border:0;border-top:1px solid #ddd;margin:30px 0;}"
            "p{line-height:1.6;}"
            "</style>"
        )
        # Rebuild the body as HTML — convert '#' headers to <h1>/<h2>,
        # leave <table> blocks intact, wrap text paragraphs in <p>.
        html_parts = []
        for line in aggregated_lines:
            if line.startswith("# "):
                html_parts.append(f"<h1>{_html.escape(line[2:])}</h1>")
            elif line.startswith("## "):
                html_parts.append(f"<h2>{_html.escape(line[3:])}</h2>")
            elif line.startswith("<table>") or line.startswith("  <tr>") or line == "</table>":
                html_parts.append(line)
            elif line == "---":
                html_parts.append("<hr>")
            elif line.startswith("**") and line.endswith("**"):
                html_parts.append(f"<p><strong>{_html.escape(line[2:-2])}</strong></p>")
            elif line.startswith("_") and line.endswith("_"):
                html_parts.append(f"<p><em>{_html.escape(line[1:-1])}</em></p>")
            elif line.strip():
                html_parts.append(f"<p>{_html.escape(line)}</p>")
        html_body = (
            "<!DOCTYPE html><html><head>"
            f"<meta charset='utf-8'><title>Visual Analysis — {campaign_id}</title>"
            f"{html_style}</head><body>\n"
            + "\n".join(html_parts) +
            "\n</body></html>"
        )
        viz_html_path = os.path.join(campaign_folder, "viz_report.html")
        with open(viz_html_path, "w") as fout:
            fout.write(html_body)

        _emit(session_id, "campaign_viz_complete",
              campaign_id=campaign_id,
              viz_report_md_path=viz_md_path,
              viz_report_html_path=viz_html_path)

        return {
            "ok": True,
            "viz_report_path": viz_md_path,           # kept for backward compat
            "viz_report_md_path": viz_md_path,
            "viz_report_html_path": viz_html_path,
            "iteration_count": len(trials),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ── Exports ──────────────────────────────────────────────────────────────────

CAMPAIGN_TOOLS = [
    campaign_start,
    campaign_get_state,
    campaign_propose_template,
    campaign_fill_template,
    campaign_run_simulation,
    campaign_measure_metrics,
    campaign_record_trial,
    campaign_check_stop,
    campaign_generate_report,
    campaign_save_narrative,
    campaign_visualize_all,
]
