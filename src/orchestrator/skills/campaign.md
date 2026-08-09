---
name: campaign
description: Use for benchmark sweeps, parameter optimization across iterations, or comparing multiple simulation configurations. Also load when the user says "Resume campaign <id>" or "Continue campaign <id>".
---

══════════════════════════════════════════════════════════════════════════════
CAMPAIGN MODE — AUTONOMOUS BENCHMARKING (v2)
══════════════════════════════════════════════════════════════════════════════

You (the orchestrator) run the ENTIRE campaign yourself. There is no Python
driver. You:

  • Turn 1: Design the plan and print it as prose for the user to confirm.
  • Turn 2 (after user "confirm"): Enter a ReAct loop where you call
    campaign_* tools directly — propose knobs, run simulate(), measure,
    record the trial, check whether to stop, repeat.

State lives on disk under `runs/<campaign_id>/` (plan.json, control.json,
trials.jsonl). You never need to remember trial history — always read it
via campaign_get_state() at the start of each iteration.

──────────────────────────────────────────────────────────────────────────
AVAILABLE METRICS
──────────────────────────────────────────────────────────────────────────

Currently registered objective metrics you can pick from:

  • `delta_cavity_surface_area` — Δ crack surface area (Å²): the surface area
    of the crack opened during loading, measured from an OVITO surface mesh.
    Default direction: minimize.

If the user asks for a metric not in this list (fracture toughness, peak
stress, DXA density, etc.), tell them it's not yet implemented and offer
to proceed with `delta_cavity_surface_area` instead, or wait until it's added.

══════════════════════════════════════════════════════════════════════════════
TURN 1 — DESIGN THE CAMPAIGN PLAN
══════════════════════════════════════════════════════════════════════════════

When a user first requests a campaign, INFER the appropriate template from
their request (using your existing knowledge — Griffith crack, fatigue,
KIM model selection, CNA cutoff formula, OVITO defaults). Then present
BOTH the template AND the tunable knobs. The user may edit any field.

────────────────────────────────────────────────────────────────────────────
TEMPLATE INFERENCE
────────────────────────────────────────────────────────────────────────────
Examples of inferring the template:
  - "Run a campaign on Griffith crack Al"
       → 3-phase monotonic template, Al (fcc 4.05 Å), 60×30×3, NPT-NVT-NVT
  - "Run a campaign on fatigue in Cu"
       → 4-phase template (add cyclic loading phase), Cu (fcc 3.615 Å)
  - "Run a benchmark campaign on crack propagation in Fe"
       → 3-phase template, Fe (bcc 2.87 Å), appropriate KIM model

When you present the proposed plan, list ALL of these template fields
explicitly so the user can see them and request changes:
  • Material (element, lattice type, a₀)
  • System size (unit cells in x, y, z)
  • KIM model
  • Timestep
  • Each phase: ensemble (NPT/NVT), duration, conditions
  • Dump frequency and properties (c_cna, c_pe, c_stress)
  • OVITO visualization (colors per CNA, camera angle)

────────────────────────────────────────────────────────────────────────────
TUNABLE KNOBS — ALWAYS THESE 4 FOR CRACK/FATIGUE CAMPAIGNS
────────────────────────────────────────────────────────────────────────────
| Knob              | Default Range       | Units            |
|-------------------|---------------------|------------------|
| strain_rate       | [1e-3, 2e-2]        | /ps              |
| temperature       | [100, 700]          | K                |
| crack_length_frac | [0.1, 0.6]          | fraction of L_x  |
| crack_width_a0    | [2, 8]              | × a₀             |

For CYCLIC (fatigue) campaigns, replace `strain_rate` with `strain_amplitude`
(typical range 0.001–0.01 fraction of L_y).

The user MAY NARROW these ranges during setup. The user MAY NOT change
which knobs are tuned (this campaign mode is specifically for
crack/fatigue crack-area optimization).

────────────────────────────────────────────────────────────────────────────
PRESENT THE PLAN AS PROSE (do NOT call any tool yet)
────────────────────────────────────────────────────────────────────────────
Print the plan in chat so the user can read/edit it. Example format:

  "I'll set up a Griffith crack campaign for Al. Here's my proposed plan
   (you can change ANY field):

   OBJECTIVE: minimize Δ crack surface area (crack opening, Å²)

   TEMPLATE (locked once you confirm):
     • Material: Al (fcc, a₀ = 4.05 Å)
     • System size: 60 × 30 × 3 unit cells
     • KIM model: EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005
     • Timestep: 0.001 ps
     • Phase 1: NPT, 5 ps, 300 K, 0 bar (pre-stress equilibration)
     • Phase 2: NVT, 5 ps, 300 K (crack seed + relaxation)
     • Phase 3: NVT, 15 ps (monotonic tensile loading)
     • Dump: every 1500 steps, properties c_pe, c_stress[1..6], c_cna
     • CNA cutoff: 0.854 × 4.05 = 3.457 Å
     • OVITO colors: FCC=green, HCP=yellow, Other=red, BCC=blue, ICO:gray
     • Camera: down +z

   TUNABLE KNOBS (vary across iterations):
     • strain_rate: [1e-3, 2e-2] /ps
     • temperature: [100, 700] K
     • crack_length_frac: [0.1, 0.6]
     • crack_width_a0: [2, 8] × a₀

   ITERATIONS: 20 max
  STOP CRITERIA: runs to max iterations; stops early only if you
               pause/cancel or 3 simulations fail in a row

   Would you like to change anything, or shall I proceed? Reply 'confirm'
   to start the campaign, or tell me what to change."

RULES:
  - INFER template from the user's request; do NOT ask for LAMMPS details
    the user hasn't specified — use your defaults.
  - NEVER auto-call `campaign_start(...)` without explicit user confirmation.
  - If user requests changes ("use Cu instead", "narrow strain to
    [1e-3, 5e-3]", "max_iter 30"), update the proposal and ask again.
  - DEFAULT ovito_colors are "FCC=green, HCP=yellow, Other=red, BCC=blue, ICO:gray"
    Override ONLY if the user explicitly states colors.
  - PRESERVE user-specified colors VERBATIM across ALL subsequent turns.
  - If user says "confirm" / "yes" / "go", proceed to Turn 2.

══════════════════════════════════════════════════════════════════════════════
TURN 2 — LAUNCH + RUN THE LOOP (all in one ReAct trace)
══════════════════════════════════════════════════════════════════════════════

After user confirms, call `campaign_start(plan={...})` with the FULL plan
dict as a single typed arg. Then enter the iteration loop below.

────────────────────────────────────────────────────────────────────────────
STEP 1 — campaign_start(plan={...})
────────────────────────────────────────────────────────────────────────────

Pass the plan as a single structured dict. Required top-level keys:

  {
    "objective": {
      "metric":       "delta_cavity_surface_area",  # from AVAILABLE METRICS above
      "direction":    "minimize",    # or "maximize"
      "units":        "Å²",
      "display_name": "Δ crack surface area",
      "target":        null          # optional; stops early if best <= target (min)
                                     # or best >= target (max)
    },
    "template": {
      "material":         "Al",
      "system_size":      "60x30x3",
      "kim_model":        "EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005",
      "timestep_ps":      0.001,
      "phase_1":          "NPT, 5 ps, 300 K, 0 bar",
      "phase_2":          "NVT, 5 ps, 300 K (crack seed + relaxation)",
      "phase_3":          "NVT, 15 ps (monotonic tensile loading)",
      # phase_4 only for fatigue: "NVT, 50 ps, cyclic loading wiggle"
      "dump_freq_steps":  1500,
      "cna_cutoff_a":     3.457,
      "ovito_colors":     "FCC=green, HCP=yellow, Other=red, BCC=blue, ICO=gray",
      "camera":           "down +z"
    },
    "knobs": {
      "strain_rate":       [1e-3, 2e-2],    # or strain_amplitude for fatigue
      "temperature":       [100, 700],
      "crack_length_frac": [0.1, 0.6],
      "crack_width_a0":    [2, 8]
    },
    "max_iterations": 20,
    "stop_criteria": {
      "no_improvement_patience": 5,
      "convergence": {"epsilon_pct": 1.0, "window": 5}
    }
  }

The tool writes plan.json and control.json, emits campaign_started, and
returns the campaign_id. It also writes campaign_id onto orchestrator
state so subsequent campaign_* tools find it automatically — you do NOT
need to pass it as an argument to any of them.

Briefly tell the user "campaign benchmark_... launched" after the tool
returns, then proceed to Step 2.

────────────────────────────────────────────────────────────────────────────
STEPS 2–8 — THE ITERATION LOOP (repeat until check_stop says stop)
────────────────────────────────────────────────────────────────────────────

For each iteration, run this exact sequence of tool calls:

This instuction to be follwed by the orchestratr step by step to control the steps and the tool calls whcih are using the failure mode in the trails.jsonl

  ┌─────────────────────────────────────────────────────────────────────┐
  │  2. campaign_get_state()                                            │
  │       → returns {plan, trials, control, best_so_far,                │
  │                  iterations_since_improvement, next_iteration}      │
  │                                                                     │
  │  3. REASON in chat: write your hypothesis (1–3 sentences) and       │
  │     pick concrete knob values within the plan.knobs bounds.         │
  │     Reference specific past trials when past data exists.           │
  │     This chat text streams live to the user.                        │
  │                                                                     │
  │  4. simulate(                                                       │
  │       material=<from plan.template>,                                │
  │       temperature=<your T knob value>,                              │
  │       ensemble="NVT",                                               │
  │       duration_ps=<phase 3 or 4 duration from plan.template>,       │
  │       system_size=<from plan.template>,                             │
  │       kim_model=<from plan.template>,                               │
  │       simulation_details=<full spec — see STRICT ITERATION FORMAT>, │
  │       strain_rate=<value>,           # monotonic; omit for fatigue  │
  │       strain_amplitude=<value>,      # fatigue; omit for monotonic  │
  │       crack_length_frac=<value>,                                    │
  │       crack_width_a0=<value>,                                       │
  │       auto_approve=True,                                            │
  │       execution_target="local", 
          camera_angle="top"                                     │
  │     )                                                               │
  │       → generates LAMMPS + OVITO scripts (does NOT run LAMMPS)      │
  │         review_passed + lammps_script land on orchestrator state    │
  │                                                                     │
  │  5. campaign_run_simulation()                                       │
  │       → actually runs LAMMPS from the script simulate() just made.  │
  │         Streams trajectory frames + thermo data to the chat live.   │
  │         Blocks until LAMMPS finishes (~1–5 min per iteration).      │
  │         Returns {ok, run_folder} on success, or                     │
  │         {error, run_folder: None} on failure — never raises.        │
  │       IF error: skip to step 7 with failure_reason set (do NOT      │
  │       call measure_metrics or you'll get a "no dump files" error).  │
  │                                                                     │
  │  6. campaign_measure_metrics(                                       │
  │       run_folder=<the run_folder from step 5>,                      │
  │       metric_name=plan.objective.metric                             │
  │     )                                                               │
  │       → returns metric dict OR {"error": "..."} on failure          │
  │                                                                     │
  │  7. campaign_record_trial(                                          │
  │       iteration=<from state.next_iteration>,                        │
  │       knobs={"strain_rate": ..., "temperature": ..., ...},          │
  │       hypothesis=<the same hypothesis you wrote in step 3>,         │
  │       metrics=<result from step 6, or None if any step errored>,    │
  │       run_folder=<from step 5>,                                     │
  │       wall_clock_s=<approximate; ok to estimate>,                   │
  │       failure_reason=<None on success, else a short string>         │
  │     )                                                               │
  │                                                                     │
  │  8. campaign_check_stop()                                           │
  │       → returns {should_stop, reason, detail}                       │
  │       if should_stop: break out of loop                             │
  │       else: back to step 2                                          │
  └─────────────────────────────────────────────────────────────────────┘

CRITICAL RULES for the loop:
  • Call these tools in EXACTLY this order every iteration.
  • ALWAYS pass typed knob values to campaign_record_trial — no prose
    reconstruction. The recorded trial IS what future iterations see.
  • ALWAYS include your hypothesis in campaign_record_trial — future
    iterations read your past hypotheses via campaign_get_state and
    the final report quotes them.
  • If simulate returns without review_passed OR without lammps_script,
    treat it as a failed trial: SKIP steps 5 + 6, call campaign_record_trial
    with metrics=None and failure_reason="<short reason>" (e.g.,
    "review_failed: <first issue>"). Then check_stop and continue.
  • If campaign_run_simulation returns {"error": ...}, SKIP step 6,
    record_trial with metrics=None + failure_reason="lammps_error: <msg>".
  • If a tool returns {"error": "..."}, DO NOT retry more than ONCE.
    On second failure, record the trial as failed and move on.
  • DO NOT call check_stop yourself — always via the tool. It's the
    only source of truth for whether to keep looping.
  • DO NOT call analyze_runs during a campaign iteration — it runs a
    separate Analyst agent and disrupts the loop. Everything you need
    to know about the run is returned by campaign_run_simulation and
    campaign_measure_metrics.

══════════════════════════════════════════════════════════════════════════════
STRICT ITERATION FORMAT — simulation_details prose for the simulate() call
══════════════════════════════════════════════════════════════════════════════

You must re-emit the FULL simulation spec each iteration, filling in only
the 4 knob values and the temperature. Keep every other line identical
across iterations so scripts are byte-comparable across trials.

Reconstruct the spec from plan.template each iteration. Read the template
fields via campaign_get_state() — DO NOT try to remember them.

CRITICAL — simulate(simulation_details=...) receives ONLY the "Simulation
specification:" block below. DO NOT include the "Hypothesis:" line or any
reasoning prose in that argument — the hypothesis goes to the chat message
(streamed to user) and to campaign_record_trial(hypothesis="..."). Mixing
hypothesis into simulation_details bloats the LAMMPS generator's prompt
with irrelevant text.

CRITICAL — REPRODUCIBILITY: the velocity initialization MUST be byte-identical
across every iteration. Always specify EXACTLY:
    velocity all create <T> 12345 mom yes rot yes dist gaussian
Same seed (12345), same flags (mom yes rot yes dist gaussian), every
iteration. Only <T> (the temperature knob) changes. Never let the seed or
argument order drift — different seeds give different thermal microstates
and make crack-area comparisons across trials unfair.

────────────────────────────────────────────────────────────────────────────
MONOTONIC CAMPAIGNS — Simulation specification template
────────────────────────────────────────────────────────────────────────────

  Hypothesis: <1–3 sentences explaining your reasoning; reference past
              trial numbers + Δ crack surface area values when past data exists.>

  Simulation specification:
  - Material: <plan.template.material>, lattice fcc <a₀> Å, system size
    <plan.template.system_size> unit cells
  - KIM Model: <plan.template.kim_model>
  - Boundary: p p p
  - Timestep: <plan.template.timestep_ps> ps
  - Velocity initialization (FIXED — use this EXACT line every iteration,
    never change the seed or arguments; reproducibility across trials
    depends on it):
        velocity all create <T> 12345 mom yes rot yes dist gaussian

  - Phase 1 (Pre-stress equilibration):
      ensemble: NPT, temperature: <T> K, pressure: 0 bar, duration: <phase_1 duration>
      thermostat damp: 0.1, barostat damp: 1.0

  - Phase 2 (Crack seeding + relaxation):
      ensemble: NVT, temperature: <T> K, duration: <phase_2 duration>
      crack: horizontal at y = L_y/2
      crack: central
      crack_length: <crack_length_frac> × L_x
      crack_width: <crack_width_a0> × a₀
      crack method: neigh_modify exclude (NOT delete_atoms)

  - Phase 3 (Monotonic tensile loading campaign):
      ensemble: NVT, temperature: <T> K, duration: <phase_3 duration>
      strain_rate: <strain_rate> /ps along y-axis
      method: fix deform y erate <strain_rate> units box remap x
      NO wiggle, monotonic loading only

────────────────────────────────────────────────────────────────────────────
FATIGUE CAMPAIGNS — Phase 3 + Phase 4 override
────────────────────────────────────────────────────────────────────────────
In fatigue campaigns, replace Phase 3 (above) with:

  - Phase 3 (Passive relaxation for fatigue):
      ensemble: NVT, temperature: <T> K, duration: 10 ps (no fix deform)
      LAMMPS: fix nvt all nvt temp <T> <T> 0.1

  - Phase 4 (Cyclic loading):
      ensemble: NVT, temperature: <T> K, duration: <phase_4 duration>
      Apply sinusoidal strain in y with amplitude <strain_amplitude>×L_y,
      period 10 ps → 5 cycles over 50 ps total.
      LAMMPS: fix cyclic all deform 1 y wiggle amplitude 10.0 units box

────────────────────────────────────────────────────────────────────────────
OUTPUTS — same for both modes, keep verbatim
────────────────────────────────────────────────────────────────────────────
  - Outputs per phase:
      dump every <plan.template.dump_freq_steps> steps
      properties: c_cna, c_pe, c_stress[1..6]
      cna cutoff: <plan.template.cna_cutoff_a> Å
      dump filenames:
      MONOTONIC: dump.*.equil (Phase 1), dump.*.relax (Phase 2), dump.*.load (Phase 3)
      FATIGUE:   dump.*.equil (Phase 1), dump.*.relax (Phase 2), dump.*.observe (Phase 3), dump.*.load (Phase 4)

  - Visualization (OVITO):
      CNA coloring: <plan.template.ovito_colors>
      camera: <plan.template.camera>

Replace the angle-bracket placeholders (<T>, <crack_length_frac>,
<crack_width_a0>, <strain_rate> or <strain_amplitude>) with concrete
numerical values within the allowed ranges. Keep ALL OTHER LINES EXACTLY
AS SPECIFIED for script uniformity.

══════════════════════════════════════════════════════════════════════════════
END OF LOOP — GENERATE REPORT + WRITE NARRATIVE
══════════════════════════════════════════════════════════════════════════════

Once campaign_check_stop returns should_stop=True with a reason OTHER
than "user_paused":

  ┌─────────────────────────────────────────────────────────────────────┐
  │  9. campaign_generate_report(stop_reason=<the reason from step 8>)  │
  │       → returns {report_md_path, report_png_path, skeleton,         │
  │                  next_step}                                         │
  │                                                                     │
  │ 10. WRITE the "What was learned" narrative in chat (streamed live). │
  │     Reference specific trials by number, discuss which hypotheses   │
  │     held up vs were refuted, which knob mattered most, and suggest  │
  │     concrete next steps (e.g. "extend T below 250K" or "vary        │
  │     crack_width_a0 in a follow-up campaign").                       │
  │                                                                     │
  │ 11. campaign_save_narrative(text=<your narrative from step 10>)     │
  │       → appends under "## What was learned" in report.md,           │
  │         emits campaign_complete SSE, sets control.status=done.      │
  │                                                                     │
  │ 12. FINAL chat message — offer visual analysis (opt-in):            │
  │     "Report and graph are ready. Would you like me to generate a    │
  │      per-iteration **visual analysis** (renders 3 frames per iter   │
  │      and describes crack morphology, dislocations, stacking faults, │
  │      etc.)? Takes about 15 sec per iteration. Reply 'yes' to run."  │
  │     Then END TURN — do NOT auto-call campaign_visualize_all.        │
  └─────────────────────────────────────────────────────────────────────┘

If in a LATER turn the user replies "yes" / "generate visual analysis" /
"show visuals" / any affirmative response after a campaign completed:
  → call `campaign_visualize_all()` (no args — reads campaign_id from state).
    It loops through every iteration, renders frames, calls the vision LLM,
    and writes viz_report.html (opens directly in Chrome) + viz_report.md.
  → When it returns, tell the user:
    "Visual analysis ready.
     📄 [Open in browser (HTML)](/runs/<campaign_id>/viz_report.html)
      • [Markdown version](/runs/<campaign_id>/viz_report.md)"

If reason was "user_paused" — do NOT generate report. Instead say something
like "Campaign paused after N trials. Resume by saying 'Resume campaign
<id>' or cancel to see the report so far." Then end the turn.

──────────────────────────────────────────────────────────────────────────
NARRATIVE GUIDELINES (step 10)
──────────────────────────────────────────────────────────────────────────
The narrative is the interpretive value of the report — it's what a
scientist reads to understand what happened. Cover:

  • Which trial won (iteration + knob values + metric value)
  • What hypothesis you were pursuing across the run — held or refuted?
    Reference specific trial numbers.
  • Which knob had the biggest effect on the objective; which barely mattered
  • Any anomalies (crashes, outliers, unexpected sensitivities)
  • Suggested next-step campaigns (narrower bounds, different knobs,
    higher resolution around the winning region)
  • If any trials failed: what went wrong, was it a systematic issue?

Length: 4–8 sentences per section is fine. Precise references beat vague
claims — "iteration 4 (T=250K) reduced Δ crack surface area from the iter-3 peak of 1230 Å²
down to 0.72 Å², confirming lower T is beneficial in this system" is much
more useful than "lower T tended to help."

══════════════════════════════════════════════════════════════════════════════
RESUMING A PAUSED / INTERRUPTED CAMPAIGN
══════════════════════════════════════════════════════════════════════════════

If the user says "Resume campaign <id>" or "Continue campaign <id>":

  1. Call campaign_get_state() — check control.status.
  2. If control.status == "paused": tell the user "resuming campaign <id>
     from trial <next_iteration>", write control.json's status back to
     "running" by proceeding with the loop (check_stop will pass because
     the pause flag was cleared by the frontend's Resume POST endpoint).
  3. If control.status == "cancelled" or "done": tell the user the
     campaign is closed; offer to generate the report if not already done,
     or start a new campaign.
  4. If control.status == "running" (server was killed mid-loop, restart
     picked up): just re-enter the loop from step 2.

The plan on disk + trials.jsonl are the truth — everything you need to
continue is in campaign_get_state()'s return.

══════════════════════════════════════════════════════════════════════════════
FAILURE HANDLING SUMMARY
══════════════════════════════════════════════════════════════════════════════

Every campaign_* tool returns errors as `{"error": "..."}` dicts — they
never raise. Your job is to react to errors as data, not treat them as
crashes:

  • simulate() failure (no lammps_script or review_passed=false):
    → record_trial(metrics=None, failure_reason="review_failed: ...")
    → check_stop → continue

  • measure_metrics returns {"error": ...}:
    → record_trial(metrics=None, failure_reason="measure_failed: <msg>")
    → check_stop → continue

  • record_trial returns {"error": ...}:
    → tell the user something is broken and STOP the campaign — this
      means the plan.json or filesystem is unrecoverable

  • 3 consecutive failed trials → check_stop will return
    should_stop=True, reason="consecutive_failures". The campaign
    aborts cleanly and you generate the report from what completed.

DO NOT change strategy based on a single failure. Failures are data;
continue the planned exploration. Only escalate to a strategy change if
2+ consecutive trials fail for the same reason.

══════════════════════════════════════════════════════════════════════════════
