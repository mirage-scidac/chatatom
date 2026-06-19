"""
CampaignManager — closed-loop benchmarking driver.

Runs N iterations of the simulation_graph in a background thread:
  for each iteration:
    1. Build synthesized "user message" with past results
    2. Invoke simulation_graph (chat → extract → lammps → viz)
    3. Auto-run LAMMPS (no human approval)
    4. Measure Δa from dump files
    5. Append to campaign_log.md + update summary.json
    6. Emit SSE events for live frontend updates

Storage layout per campaign:
  runs/benchmark_<id>/
    ├── campaign_log.md       (all iterations appended)
    ├── summary.json          (Δa per iteration, for plotting)
    └── run_<timestamp>/      (per-iteration LAMMPS output, nested)
"""
import json
import os
import queue
import re
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional


class _NoopContext:
    """No-op context manager — used when CONFIGURED_LM is unavailable."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

# Sibling-module imports (src/ added to sys.path by app.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from benchmark.campaign_prompts import format_history_message
from benchmark.crack_metrics import compute_delta_a


class CampaignManager:
    """Runs an autonomous benchmark campaign in a background thread."""

    def __init__(
        self,
        campaign_id: str,
        n_iterations: int,
        original_user_message: str,
        sse_q: queue.Queue,
        num_cores: Optional[int] = None,
        target: str = "minimize delta_a",
        runs_base_dir: str = "runs",
        template: Optional[Dict[str, Any]] = None,
        bounds: Optional[Dict[str, Any]] = None,
    ):
        self.campaign_id = campaign_id
        self.n_iterations = n_iterations
        self.original_user_message = original_user_message
        self.sse_q = sse_q
        self.num_cores = num_cores if num_cores is not None else getattr(
            config, "LAMMPS_CORES", 1
        )
        self.target = target

        # Locked template inferred during setup (material, phases, OVITO, etc.)
        # and locked knob bounds. Both passed into every iteration's prompt
        # so the LLM produces scripts matching the user-confirmed template
        # and stays within the user-confirmed knob ranges.
        self.template: Dict[str, Any] = template or {}
        self.bounds: Dict[str, Any] = bounds or {}

        self.output_folder = os.path.join(runs_base_dir, campaign_id)
        os.makedirs(self.output_folder, exist_ok=True)

        self.log_path = os.path.join(self.output_folder, "campaign_log.md")
        self.summary_path = os.path.join(self.output_folder, "summary.json")

        self.past_runs: List[Dict[str, Any]] = []  # in-memory loop state
        self.best_delta_a: float = float("inf")

    # ─── Main loop ──────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Run the N-iteration loop. Sync method, called in a worker thread.

        DSPy 3.x locks dspy.settings to the thread that called dspy.configure().
        Since this runs in a background thread, we MUST wrap the loop in
        dspy.context(lm=...) so DSPy modules (intent_node, generate_lammps_node,
        etc.) can use the LM from this thread.
        """
        # Import here to avoid circular imports at module load
        import dspy
        from graph.graph import simulation_graph
        from tools import run_lammps_simulation
        from agents import CONFIGURED_LM

        self._write_log_header()
        self._update_summary()

        if CONFIGURED_LM is None:
            print("[CampaignManager] WARNING: CONFIGURED_LM is None — DSPy may fail in this thread")

        # Wrap the entire loop in dspy.context so all DSPy calls in this thread
        # (across all iterations) use the right LM and don't try to mutate
        # the main-thread settings.
        ctx = dspy.context(lm=CONFIGURED_LM) if CONFIGURED_LM is not None else _NoopContext()
        with ctx:
            for i in range(1, self.n_iterations + 1):
                try:
                    self._run_one_iteration(i, simulation_graph, run_lammps_simulation)
                except Exception as e:
                    print(f"[CampaignManager] iteration {i} crashed: {e}")
                    traceback.print_exc()
                    self._record_failure(i, error=str(e))

        self._emit("campaign_complete", {
            "campaign_id": self.campaign_id,
            "n_iterations": self.n_iterations,
            "best_so_far": self._best_so_far_serializable(),
            "log_path": self.log_path,
            "summary_path": self.summary_path,
            "completed_at": datetime.utcnow().isoformat(),
        })

        return self._build_manifest_dict()

    # ─── One iteration ──────────────────────────────────────────────────────

    def _run_one_iteration(
        self, i: int, simulation_graph, run_lammps_simulation
    ) -> None:
        self._emit("campaign_iteration_start", {
            "iteration": i,
            "total": self.n_iterations,
            "best_so_far": self._best_so_far_serializable(),
        })

        # Step 1: synthesize user message
        # Every iteration includes the LOCKED TEMPLATE and KNOB BOUNDS so the
        # LLM produces scripts that match what the user confirmed during setup
        # (and so the LLM doesn't drift back to default Griffith Al if the user
        # picked Cu or changed the phase durations).
        template_block = self._format_template_block()
        bounds_block = self._format_bounds_block()

        if i == 1:
            user_message = (
                f"[CAMPAIGN ITERATION 1 — autonomous execution, no further user input]\n\n"
                f"The user previously confirmed the campaign with this message:\n"
                f"  \"{self.original_user_message}\"\n\n"
                f"This is iteration 1 of {self.n_iterations}. No past results yet.\n\n"
                f"{template_block}"
                f"{bounds_block}"
                f"Propose initial parameter values for the 4 tunable knobs WITHIN "
                f"the bounds above, using the LOCKED TEMPLATE above.\n\n"
                f"YOU MUST respond in the STRICT ITERATION FORMAT defined in your system "
                f"prompt (Hypothesis: ... + Simulation specification: ...). DO NOT respond "
                f"with the campaign setup conversation — that was the previous turn. The "
                f"campaign is already locked in; produce a complete simulation spec NOW "
                f"that EXACTLY matches the locked template (material, phases, durations, "
                f"OVITO settings, camera — all of it)."
            )
        else:
            history_msg = format_history_message(
                self.past_runs, iteration=i, target=self.target
            )
            user_message = (
                f"[CAMPAIGN ITERATION {i} — autonomous execution]\n\n"
                f"{template_block}"
                f"{bounds_block}"
                f"{history_msg}\n\n"
                f"Stay within the locked template and knob bounds above."
            )

        # Step 2: invoke graph — checkpoint loads prior iteration state by thread_id
        state = {
            "user_message": user_message,
            "session_id": self.campaign_id,
            "workflow_stage": "initial",
            "execution_target": "local",
            "conversation_history": [],
        }
        final_state = simulation_graph.invoke(
            state,
            config={"configurable": {"thread_id": self.campaign_id}},
        )

        chat_response = final_state.get("analysis", "") or ""
        lammps_script = final_state.get("lammps_script", "") or ""
        review_passed = bool(final_state.get("review_passed"))

        # Step 3: gate on review
        if not review_passed or not lammps_script:
            self._record_failure(
                i,
                chat_response=chat_response,
                review_issues=final_state.get("review_issues"),
                reason="review_failed_or_no_script",
            )
            return

        # Step 4: auto-run LAMMPS (with parent_dir if supported, else top-level)
        start_t = time.time()
        try:
            run_result = self._invoke_lammps(
                run_lammps_simulation, lammps_script, final_state
            )
        except Exception as e:
            print(f"[CampaignManager] LAMMPS crashed on iter {i}: {e}")
            traceback.print_exc()
            self._record_failure(
                i,
                chat_response=chat_response,
                lammps_script=lammps_script,
                reason=f"lammps_error: {e}",
            )
            return
        wall_clock = time.time() - start_t

        run_folder = self._extract_run_folder(run_result)

        # Step 5: measure Δa from dump files
        delta_a_dict: Dict[str, Any]
        try:
            delta_a_dict = compute_delta_a(run_folder)
        except Exception as e:
            print(f"[CampaignManager] delta_a measurement failed on iter {i}: {e}")
            traceback.print_exc()
            delta_a_dict = {"delta_a": None, "error": str(e)}

        delta_a = delta_a_dict.get("delta_a")
        if delta_a is not None and delta_a < self.best_delta_a:
            self.best_delta_a = delta_a

        # Step 6: persist
        params_text = self._extract_params_text(chat_response)
        record = {
            "iteration": i,
            "params_text": params_text,
            "delta_a": delta_a,
            "n_defect_initial": delta_a_dict.get("n_defect_initial"),
            "n_defect_final": delta_a_dict.get("n_defect_final"),
            "run_folder": run_folder,
            "review_passed": True,
            "wall_clock_s": round(wall_clock, 2),
            "best_so_far": self._best_so_far_serializable(),
        }
        self.past_runs.append(record)

        self._append_iteration_to_log(
            i=i,
            chat_response=chat_response,
            lammps_script=lammps_script,
            run_folder=run_folder,
            delta_a_dict=delta_a_dict,
            wall_clock_s=wall_clock,
        )
        self._update_summary()

        # Step 7: emit SSE
        self._emit("campaign_iteration_complete", {
            "iteration": i,
            "params_text": params_text,
            "delta_a": delta_a,
            "best_so_far": self._best_so_far_serializable(),
            "run_folder": run_folder,
            "wall_clock_s": round(wall_clock, 2),
        })

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _invoke_lammps(self, run_lammps_simulation, script: str, final_state: Dict):
        """Call run_lammps_simulation, passing parent_dir if the function supports it."""
        # data_callback forwards trajectory/thermo events to the SSE queue
        # so the frontend's existing handlers render live frames + plots.
        #
        # NOTE: tools.py emits two flavors via data_callback:
        #   1. trajectory_frame events ALREADY have a 'type' field
        #      ({type: 'trajectory_frame', image_data, frame_number, timestep})
        #   2. thermo data points are RAW dicts ({step, temp, press, ...}) with
        #      NO 'type' field — the Socket.IO handler in app.py wraps them as
        #      'thermo_data_point' on emit. We must do the same here so our
        #      SSE handler's switch(data.type) catches them.
        def cb(evt):
            try:
                if isinstance(evt, dict) and "type" not in evt:
                    # Raw thermo data point — wrap it
                    self.sse_q.put({"type": "thermo_data_point", **evt})
                else:
                    self.sse_q.put(evt)
            except Exception:
                pass

        render_config = {
            "ovito_script": final_state.get("ovito_script"),
            "camera_angle": final_state.get("camera_angle", "+z"),
            # CRITICAL: dump_filename_pattern tells the LAMMPS runner which
            # files to watch for trajectory rendering. Without it, no
            # trajectory_frame events are emitted to the SSE stream.
            "dump_filename_pattern": final_state.get("dump_filename_pattern", ""),
            "target_timesteps": final_state.get("target_timesteps"),
            "total_steps": final_state.get("total_steps"),
            "dump_freq_val": final_state.get("dump_freq_val"),
        }

        # Try with parent_dir kwarg (Phase 3 addition); fall back to existing signature.
        try:
            return run_lammps_simulation(
                script,
                render_config=render_config,
                data_callback=cb,
                num_cores=self.num_cores,
                parent_dir=self.output_folder,
            )
        except TypeError:
            # parent_dir not yet supported — fall back to top-level run folder
            return run_lammps_simulation(
                script,
                render_config=render_config,
                data_callback=cb,
                num_cores=self.num_cores,
            )

    def _extract_run_folder(self, run_result: Any) -> str:
        """run_lammps_simulation returns (output_files, thermo_data, frames).
        Pull the run folder from output_files['log'] dirname."""
        try:
            output_files = run_result[0] if isinstance(run_result, tuple) else run_result
            log_path = output_files.get("log") if isinstance(output_files, dict) else None
            if log_path:
                return os.path.dirname(log_path)
        except Exception:
            pass
        return ""

    def _format_template_block(self) -> str:
        """Render the locked template as a block for inclusion in iteration prompts.

        Returns "" if no template is set (uses LLM defaults). Otherwise returns
        a labeled block listing each template field so the LLM produces a
        simulation_specification that matches user-confirmed choices.
        """
        if not self.template:
            return ""
        lines = ["LOCKED TEMPLATE (use these EXACT values in your simulation specification):"]
        for key, val in self.template.items():
            lines.append(f"  - {key}: {val}")
        lines.append("")  # blank line separator
        return "\n".join(lines) + "\n"

    def _format_bounds_block(self) -> str:
        """Render the locked knob bounds as a block for inclusion in iteration prompts.

        Returns "" if no bounds set (uses defaults). Otherwise lists the
        allowed range for each knob so the LLM proposes values within bounds.
        """
        if not self.bounds:
            return ""
        lines = ["LOCKED KNOB BOUNDS (propose values STRICTLY within these ranges):"]
        for knob, rng in self.bounds.items():
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                lines.append(f"  - {knob}: [{rng[0]}, {rng[1]}]")
        lines.append("")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _extract_params_text(chat_response: str) -> str:
        """Pull knob values from the PROPOSED_KNOBS block at end of response.

        The LLM is instructed (in campaign_prompts.py iteration section) to emit
        a structured block:

            PROPOSED_KNOBS:
              strain_amplitude: 0.005
              temperature: 300
              crack_length_frac: 0.35
              crack_width_a0: 5

        Parsing this structured block is ~100% reliable, unlike the older
        approach of scraping values from free-form prose.

        Returns "strain_amp=…, T=…, crack_len=…, crack_w=…" or an error string.
        """
        text = chat_response or ""

        # Find the PROPOSED_KNOBS block — terminated by blank line or end of text
        m = re.search(
            r"PROPOSED_KNOBS\s*:\s*\n(.*?)(?:\n\s*\n|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return "<no PROPOSED_KNOBS block found>"

        block = m.group(1)

        # Short labels for known knobs (longer canonical → short display name)
        short_names = {
            "strain_amplitude": "strain_amp",
            "strain_rate":      "strain_rate",
            "temperature":      "T",
            "crack_length_frac":"crack_len",
            "crack_width_a0":   "crack_w",
        }

        parts = []
        for line in block.splitlines():
            # Allow digits in key name (e.g., "crack_width_a0", "phase_2_T")
            kv = re.match(r"\s*[\-•*]?\s*([a-z][a-z0-9_]*)\s*[:=]\s*([0-9.eE+-]+)",
                          line, re.IGNORECASE)
            if kv:
                key = kv.group(1).lower()
                val = kv.group(2)
                short = short_names.get(key, key)
                parts.append(f"{short}={val}")

        return ", ".join(parts) if parts else "<unknown params>"

    def _best_so_far_serializable(self) -> Optional[float]:
        if self.best_delta_a == float("inf"):
            return None
        return self.best_delta_a

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        try:
            self.sse_q.put({"type": event_type, **data})
        except Exception as e:
            print(f"[CampaignManager] failed to emit {event_type}: {e}")

    def _record_failure(
        self,
        i: int,
        chat_response: str = "",
        lammps_script: Optional[str] = None,
        review_issues: Any = None,
        reason: str = "unknown",
        error: Optional[str] = None,
    ) -> None:
        params_text = self._extract_params_text(chat_response) if chat_response else "<n/a>"
        record = {
            "iteration": i,
            "params_text": params_text,
            "delta_a": None,
            "review_passed": False,
            "failure_reason": reason,
            "best_so_far": self._best_so_far_serializable(),
        }
        if error:
            record["error"] = error
        self.past_runs.append(record)

        self._append_iteration_to_log(
            i=i,
            chat_response=chat_response,
            lammps_script=lammps_script,
            run_folder=None,
            delta_a_dict=None,
            wall_clock_s=None,
            failure_reason=reason,
            review_issues=review_issues,
            error=error,
        )
        self._update_summary()
        self._emit("campaign_iteration_failed", {
            "iteration": i,
            "reason": reason,
            "best_so_far": self._best_so_far_serializable(),
        })

    # ─── File I/O ───────────────────────────────────────────────────────────

    def _write_log_header(self) -> None:
        """Write the campaign_log.md header (once, at campaign start)."""
        template_summary = self._summarize_template_for_header()
        knobs_list = ", ".join(self.bounds.keys()) if self.bounds else "<none>"
        header = (
            f"# Campaign Log: {self.campaign_id}\n\n"
            f"**Target**: {self.target}\n"
            f"**Template**: {template_summary}\n"
            f"**Tuning knobs**: {knobs_list}\n"
            f"**N iterations**: {self.n_iterations}\n"
            f"**Started at**: {datetime.utcnow().isoformat()}\n\n"
            f"---\n"
        )
        with open(self.log_path, "w") as f:
            f.write(header)

    def _summarize_template_for_header(self) -> str:
        """Build a one-line summary of the locked template for the log header.

        Pulls material, system_size, and any phase_<N> fields from self.template
        (any N, in numerical order) so the header reflects what was actually
        locked (not a hardcoded default).
        """
        if not self.template:
            return "<no locked template>"
        material = self.template.get("material", "<material?>")
        size = self.template.get("system_size", "<size?>")

        # Find every phase_<N> key in the template and sort by N numerically
        phase_keys = []
        for k in self.template:
            m = re.match(r"phase_(\d+)$", k)
            if m:
                phase_keys.append((int(m.group(1)), k))
        phase_keys.sort()
        phases = [self.template[k] for _, k in phase_keys]
        phase_part = " → ".join(phases) if phases else "<phases?>"

        return f"{material} ({size}), {phase_part}"

    def _append_iteration_to_log(
        self,
        i: int,
        chat_response: str,
        lammps_script: Optional[str],
        run_folder: Optional[str],
        delta_a_dict: Optional[Dict[str, Any]],
        wall_clock_s: Optional[float],
        failure_reason: Optional[str] = None,
        review_issues: Any = None,
        error: Optional[str] = None,
    ) -> None:
        """Append a single iteration's full record to campaign_log.md."""
        lines = [f"\n## Iteration {i}\n"]

        if chat_response:
            lines.append("### Hypothesis & Proposed Parameters\n")
            lines.append(chat_response)
            lines.append("\n")

        if lammps_script:
            lines.append("### Generated LAMMPS Script\n")
            lines.append("```lammps")
            lines.append(lammps_script)
            lines.append("```\n")

        if run_folder:
            lines.append(f"### Run Folder\n`{run_folder}`\n")

        if delta_a_dict:
            d = delta_a_dict.get("delta_a")
            if d is not None:
                lines.append("### Result\n")
                lines.append(f"- **Δa = {d:+.3f} Å**")
                lines.append(f"- a_initial = {delta_a_dict.get('a_initial', 'n/a'):.3f} Å")
                lines.append(f"- a_final   = {delta_a_dict.get('a_final', 'n/a'):.3f} Å")
                lines.append(f"- n_defect_initial = {delta_a_dict.get('n_defect_initial', 'n/a')}")
                lines.append(f"- n_defect_final   = {delta_a_dict.get('n_defect_final', 'n/a')}")
                if wall_clock_s is not None:
                    lines.append(f"- wall_clock = {wall_clock_s:.2f} s")
                lines.append("")
            else:
                err = delta_a_dict.get("error", "unknown reason")
                lines.append(f"### Result\n- Δa = N/A (measurement error: {err})\n")

        if failure_reason:
            lines.append(f"### Failure\n- Reason: {failure_reason}\n")
            if review_issues:
                lines.append(f"- Review issues: {review_issues}\n")
            if error:
                lines.append(f"- Error: {error}\n")

        lines.append("---\n")

        with open(self.log_path, "a") as f:
            f.write("\n".join(lines))

    def _update_summary(self) -> None:
        """Rewrite summary.json (atomic-ish: write to tmp + rename)."""
        summary = self._build_manifest_dict()
        tmp = self.summary_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        os.replace(tmp, self.summary_path)

    def _build_manifest_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "n_iterations": self.n_iterations,
            "target": self.target,
            "log_file": "campaign_log.md",
            "best_so_far": self._best_so_far_serializable(),
            "iterations": [
                {
                    "iteration": r["iteration"],
                    "params_text": r.get("params_text"),
                    "delta_a": r.get("delta_a"),
                    "best_so_far": r.get("best_so_far"),
                    "run_folder": r.get("run_folder"),
                    "review_passed": r.get("review_passed", False),
                    "wall_clock_s": r.get("wall_clock_s"),
                }
                for r in self.past_runs
            ],
        }
