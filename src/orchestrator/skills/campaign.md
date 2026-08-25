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
  • Turn 2 (after user "confirm"): campaign_start(), then generate the
    simulation script ONCE — parameterized, with a CAMPAIGN VARIABLES block
    at the top — and hand it to the user for approval. Then stop.
  • Turn 3 (after the user approves the template): Enter a ReAct loop where
    you call campaign_* tools directly — propose knobs, fill the template,
    run, measure, record the trial, check whether to stop, repeat.

The script is generated ONCE and only once. Iterations differ ONLY in the
numbers inside the variable block, which campaign_fill_template substitutes
with plain string replacement — no LLM, no reviewer, no drift. That is what
makes the metric comparable across trials.

State lives on disk under `runs/<campaign_id>/` (plan.json, control.json,
trials.jsonl, template.lammps, template.json). You never need to remember
trial history or the script — always read state via campaign_get_state() at
the start of each iteration.

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
     • Phase 1: NPT, 5 ps, 300 K, 1 bar (pre-stress equilibration in all x, y, and z)
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

   HOW IT RUNS: I'll generate the LAMMPS script once, with the tunable
   values as variables at the top, and show it to you to approve.
   After you approve it, every iteration just swaps those numbers — so all
   20 trials are identical apart from the knobs.

   Would you like to change anything, or shall I proceed? Reply 'confirm'
   to start the campaign, or tell me what to change."

RULES:
  - NEVER write a swept knob's VALUE into the plan's `template` section.
    Those four knobs change every iteration, so pinning one to a number makes
    the plan contradict itself — and the plan's `template` section is exactly
    what the final report prints as "what was simulated". Reference the knob
    instead:
        WRONG:  "phase_1": "NPT, 5 ps, 300 K, 1 bar in x, y, and z"
        RIGHT:  "phase_1": "NPT, 5 ps, ${temperature}, 1 bar in x, y, and z"
    Same for strain_rate, crack_length_frac and crack_width_a0. Quantities
    that genuinely never vary — system_size, kim_model, timestep_ps,
    dump_freq_steps, cna_cutoff_a, phase durations — stay concrete numbers.
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
TURN 2 — LAUNCH + BUILD THE TEMPLATE
══════════════════════════════════════════════════════════════════════════════

After user confirms, call `campaign_start(plan={...})` with the FULL plan
dict as a single typed arg.

The LAMMPS script is generated exactly ONCE per campaign. Iteration 1's
hypothesis drives that generation; the script comes back parameterized (a
CAMPAIGN VARIABLES block at the top); the user approves or cancels it; and every
iteration from then on — including iteration 1 — is a pure value substitution
with NO script generation at all.

  Turn 2  campaign_start → iteration-1 hypothesis → simulate() →
          campaign_propose_template() → END TURN, wait for approval
  Turn 3  (user approved) → run iteration 1 from the template, then loop
          iterations 2..N without ever calling simulate() again

Read `template.status` from campaign_get_state() to know where you are:
  "none"      → you are in Turn 2: generate the template (T1–T5 below)
  "pending"   → the user has NOT approved yet. Call campaign_propose_template()
                with no new simulate() — it re-shows the staged template — then
                END YOUR TURN. Never regenerate a template that is pending.
  "approved"  → you are in the loop: campaign_fill_template, never simulate

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
      "phase_1":          "NPT, 5 ps, 300 K, 1 bar in x, y, and z",
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
returns, then proceed to T1.

────────────────────────────────────────────────────────────────────────────
TEMPLATE GENESIS — steps T1–T5, once per campaign (iteration 1 only)
────────────────────────────────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────────────────────────┐
  │  T1. campaign_get_state()  → confirm template.status == "none"      │
  │                                                                     │
  │  T2. REASON in chat: write iteration 1's hypothesis (1–3 sentences) │
  │      and pick concrete knob values within the plan.knobs bounds.    │
  │      These become the template's initial values, so choose sensibly │
  │      (mid-range is a fine opening move when there is no data yet).  │
  │                                                                     │
  │  T3. simulate(                                                      │
  │        material=<from plan.template>,                               │
  │        temperature=<your T knob value>,                             │
  │        ensemble="NVT",                                              │
  │        duration_ps=<phase 3 or 4 duration from plan.template>,      │
  │        system_size=<from plan.template>,                            │
  │        kim_model=<from plan.template>,                              │
  │        simulation_details=<full spec — see TEMPLATE FORMAT below.   │
  │                            It MUST include the CAMPAIGN VARIABLES   │
  │                            requirements verbatim>,                  │
  │        strain_rate=<value>,          # monotonic; omit for fatigue  │
  │        strain_amplitude=<value>,     # fatigue; omit for monotonic  │
  │        crack_length_frac=<value>,                                   │
  │        crack_width_a0=<value>,                                      │
  │        auto_approve=True,                                           │
  │        execution_target="local",                                    │
  │        camera_angle="top"                                           │
  │      )                                                              │
  │        → generates LAMMPS + OVITO scripts (does NOT run LAMMPS)     │
  │                                                                     │
  │  T4. campaign_propose_template(source_iteration=1)                  │
  │        → validates the variable block, stages the template, and     │
  │          shows the user the approval card.                          │
  │        IF it returns {"error": ...}: the generated script is not    │
  │        properly parameterized. Call simulate() ONE more time with   │
  │        the error text quoted inside simulation_details, then retry  │
  │        campaign_propose_template once. If it fails again, tell the  │
  │        user and stop — do not loop.                                 │
  │                                                                     │
  │  T5. END YOUR TURN. Tell the user the template is ready for review  │
  │      and that approving it starts iteration 1.                      │
  │      DO NOT call campaign_run_simulation. DO NOT record a trial.    │
  │      Nothing has run yet.                                           │
  └─────────────────────────────────────────────────────────────────────┘

────────────────────────────────────────────────────────────────────────────
STEPS 2–8 — THE ITERATION LOOP (repeat until check_stop says stop)
────────────────────────────────────────────────────────────────────────────

Entered on the turn AFTER the user approves the template, and run for every
iteration including iteration 1. Follow this exact sequence of tool calls;
`campaign_get_state()` and the failure modes recorded in trials.jsonl are what
you steer by.

  ┌─────────────────────────────────────────────────────────────────────┐
  │  2. campaign_get_state()                                            │
  │       → returns {plan, trials, control, template, best_so_far,      │
  │                  iterations_since_improvement, next_iteration}      │
  │                                                                     │
  │  3. REASON in chat: write your hypothesis (1–3 sentences) and       │
  │     pick concrete knob values within the plan.knobs bounds.         │
  │     Reference specific past trials when past data exists.           │
  │     This chat text streams live to the user.                        │
  │                                                                     │
  │     ON ITERATION 1 ONLY: do not re-derive knob values. Use          │
  │     template.variables verbatim — those are the values in the       │
  │     script the user approved. Restate the iteration-1 hypothesis    │
  │     you wrote before approval.                                      │
  │                                                                     │
  │  4. campaign_fill_template(                                         │
  │       knobs={"strain_rate": ..., "temperature": ...,                │
  │              "crack_length_frac": ..., "crack_width_a0": ...}       │
  │     )                                                               │
  │       → substitutes your values into the approved template and      │
  │         puts the filled script on state. NO LLM, NO reviewer.       │
  │       Only swept knobs may be passed, and only values inside the    │
  │       planned ranges — out-of-range values are REJECTED, not        │
  │       clamped. If it fails, fix your values and call it again.      │
  │       NEVER call simulate() here. The template is fixed for the     │
  │       life of the campaign; regenerating would break comparability  │
  │       with every trial already recorded.                            │
  │                                                                     │
  │  5. campaign_run_simulation()                                       │
  │       → actually runs LAMMPS from the filled script on state.       │
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
  • NEVER call simulate() inside the loop. Script generation happened once,
    in Turn 2. If you find yourself wanting a different script, the answer
    is a different campaign, not a mid-campaign regeneration.
  • ALWAYS pass typed knob values to campaign_record_trial — no prose
    reconstruction. The recorded trial IS what future iterations see.
  • ALWAYS include your hypothesis in campaign_record_trial — future
    iterations read your past hypotheses via campaign_get_state and
    the final report quotes them.
  • If campaign_fill_template returns a failure, it is YOUR knob values that
    are wrong (unknown knob name, or outside the planned range). Correct them
    and call it again — this is not a failed trial and must NOT be recorded
    as one, because no simulation ran.
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
TEMPLATE FORMAT — simulation_details prose for the ONE simulate() call
══════════════════════════════════════════════════════════════════════════════

This spec is emitted EXACTLY ONCE, at step T3. It asks the generator
for a *parameterized* script, not a one-off. Reconstruct it from plan.template
via campaign_get_state() — DO NOT try to remember the fields.

CRITICAL — simulate(simulation_details=...) receives ONLY the "Simulation
specification:" block below, starting at the CAMPAIGN VARIABLES requirements.
DO NOT include the "Hypothesis:" line or any reasoning prose in that argument
— the hypothesis goes to the chat message (streamed to the user) and to
campaign_record_trial(hypothesis="..."). Mixing hypothesis into
simulation_details bloats the LAMMPS generator's prompt with irrelevant text.

────────────────────────────────────────────────────────────────────────────
THE CAMPAIGN VARIABLES BLOCK — include this section verbatim
────────────────────────────────────────────────────────────────────────────
Reproduce the following in simulation_details word for word, substituting only
the concrete opening values. The two marker lines are a machine contract: the
campaign parses them to fill each iteration, and a script without them cannot
be used as a template.

  TEMPLATE REQUIREMENT — the script MUST begin (after any header comment)
  with a campaign variable block in EXACTLY this form:

    # ==== CAMPAIGN VARIABLES (auto-filled per iteration — do not move) ====
    # --- SWEPT KNOBS — campaign_fill_template rewrites these every iteration ---
    variable temperature       equal <T>
    variable strain_rate       equal <strain_rate>       # /ps  [monotonic]
    variable crack_length_frac equal <crack_length_frac>
    variable crack_width_a0    equal <crack_width_a0>
    # --- FIXED for this campaign — edit once at approval, never swept ---
    variable a0                equal <a₀>
    variable timestep_ps       equal <plan.template.timestep_ps>
    variable dump_freq         equal <plan.template.dump_freq_steps>
    variable cna_cutoff        equal <plan.template.cna_cutoff_a>
    # ==== END CAMPAIGN VARIABLES ====

  Rules for that block:
  - TWO GROUPS, in this order. The swept knobs come FIRST, under the
    `# --- SWEPT KNOBS ---` header; the campaign constants come SECOND, under
    the `# --- FIXED for this campaign ---` header. Both headers are required.
    The order is what makes the block readable at approval time: the reader
    sees immediately which four numbers move and which seven stay put.
  - The SWEPT group contains EXACTLY the knobs in plan.knobs — no more, no
    fewer. A knob missing here makes the script unusable as a template.
  - The FIXED group holds campaign-level scalars that never vary across
    trials. They live inside the markers (not in the script body) so the user
    can adjust them ONCE in the approval card before 20 runs commit to them —
    but campaign_fill_template refuses to rewrite them per iteration.
  - Every declaration inside the markers is `variable <name> equal <literal
    number>` — no expressions, no computed values, no LAMMPS functions. The
    two `# ---` group headers are the only non-declaration lines allowed.
  - Computed quantities (variable ymid equal ly/2, variable ampL equal
    ${strain_amplitude}*ly, region bounds, etc.) go BELOW the END marker.
  - Every command in the rest of the LAMMPS script must reference these
    through ${name}. No knob value may appear as a bare literal anywhere else
    in the LAMMPS script: write `fix deform y erate ${strain_rate}`, never
    `fix deform y erate 0.005`.
  - For FATIGUE campaigns, replace `strain_rate` with `strain_amplitude`
    (fraction of L_y) in the block.

  SCOPE — ${name} IS FOR THE LAMMPS SCRIPT ONLY:
  The ${...} rule above applies to the LAMMPS input script and nothing else.
  The OVITO visualization script is Python: it has no LAMMPS variables, and a
  ${...} left in it is a syntax error that kills the whole visualization and
  silently falls back to uncolored atoms. Write plain literal numbers there —
  `cna_cutoff = 3.087`, not `cna_cutoff = ${cna_cutoff}`. State this
  explicitly in visualization_notes when you call simulate().

REPRODUCIBILITY: the velocity initialization is written once, in the template,
and is therefore identical across every trial by construction:
    velocity all create ${temperature} 12345 mom yes rot yes dist gaussian
Keep the seed 12345 and the flags `mom yes rot yes dist gaussian` — different
seeds give different thermal microstates and make crack-area comparisons
across trials unfair.

────────────────────────────────────────────────────────────────────────────
MONOTONIC CAMPAIGNS — Simulation specification template
────────────────────────────────────────────────────────────────────────────

  Hypothesis: <1–3 sentences explaining your reasoning; for iteration 1 there
              is no past data, so state what you expect and why.>

  Simulation specification:
  - <the CAMPAIGN VARIABLES BLOCK section above, verbatim>
  - Material: <plan.template.material>, lattice fcc ${a0} Å, system size
    <plan.template.system_size> unit cells
  - KIM Model: <plan.template.kim_model>
  - Boundary: p p p
  - Timestep: ${timestep_ps} ps
  - Velocity initialization:
        velocity all create ${temperature} 12345 mom yes rot yes dist gaussian

  - Phase 1 (Pre-stress equilibration):
      ensemble: NPT, temperature: ${temperature} K, pressure: 1 bar in x, y, and z,
      duration: <phase_1 duration>
      thermostat damp: 0.1, barostat damp: 1.0

  - Phase 2 (Crack seeding + relaxation):
      ensemble: NVT, temperature: ${temperature} K, duration: <phase_2 duration>
      crack: horizontal at y = L_y/2
      crack: central
      crack_length: ${crack_length_frac} × L_x
      crack_width: ${crack_width_a0} × ${a0}
      crack method: neigh_modify exclude (NOT delete_atoms)

  - Phase 3 (Monotonic tensile loading campaign):
      ensemble: NVT, temperature: ${temperature} K, duration: <phase_3 duration>
      strain_rate: ${strain_rate} /ps along y-axis
      method: fix deform y erate ${strain_rate} units box remap x
      NO wiggle, monotonic loading only

────────────────────────────────────────────────────────────────────────────
FATIGUE CAMPAIGNS — Phase 3 + Phase 4 override
────────────────────────────────────────────────────────────────────────────
In fatigue campaigns, replace Phase 3 (above) with:

  - Phase 3 (Passive relaxation for fatigue):
      ensemble: NVT, temperature: ${temperature} K, duration: 10 ps (no fix deform)
      LAMMPS: fix nvt all nvt temp ${temperature} ${temperature} 0.1

  - Phase 4 (Cyclic loading):
      ensemble: NVT, temperature: ${temperature} K, duration: <phase_4 duration>
      Apply sinusoidal strain in y with amplitude ${strain_amplitude}×L_y,
      period 10 ps → 5 cycles over 50 ps total.
      LAMMPS: variable ampL equal ${strain_amplitude}*ly
              fix cyclic all deform 1 y wiggle ${ampL} 10.0 units box
      (variable ampL is a computed value — it belongs BELOW the END marker.)

────────────────────────────────────────────────────────────────────────────
OUTPUTS — same for both modes, keep verbatim
────────────────────────────────────────────────────────────────────────────
  - Outputs per phase:
      dump every ${dump_freq} steps
      properties: c_cna, c_pe, c_stress[1..6]
      cna cutoff: ${cna_cutoff} Å
      dump filenames:
      MONOTONIC: dump.*.equil (Phase 1), dump.*.relax (Phase 2), dump.*.load (Phase 3)
      FATIGUE:   dump.*.equil (Phase 1), dump.*.relax (Phase 2), dump.*.observe (Phase 3), dump.*.load (Phase 4)

  - Visualization (OVITO):
      CNA coloring: <plan.template.ovito_colors>
      camera: <plan.template.camera>
      The OVITO script is Python — use literal numbers only, never ${...}.
      CNA cutoff there must be written out, e.g. cna_cutoff = 3.087

Replace the angle-bracket placeholders (<a₀>, <T>, <crack_length_frac>,
<crack_width_a0>, <strain_rate> or <strain_amplitude>, and the plan.template
fields) with concrete numerical values within the allowed ranges — but ONLY
inside the CAMPAIGN VARIABLES block. Everywhere else the script must use
${name}. Keep ALL OTHER LINES EXACTLY AS SPECIFIED.

══════════════════════════════════════════════════════════════════════════════
TEMPLATE APPROVAL — the pause between Turn 2 and the loop
══════════════════════════════════════════════════════════════════════════════

campaign_propose_template() shows the user an editable card with the full
parameterized script and ends your involvement for that turn.

  • END YOUR TURN after it returns ok. Nothing has run; there is no trial to
    record and no metric to measure.
  • The card is READ-ONLY. It has exactly two buttons, Approve and Cancel.
    The user cannot edit the script, so what they approve is byte-for-byte
    what you generated. Never tell them to edit it.
  • Approval arrives as a new turn containing "Template approved". On that
    turn: call campaign_get_state(), confirm template.status == "approved",
    and enter the loop at Step 2 for iteration 1, using template.variables
    as iteration 1's knob values.
  • Cancel ends the campaign — control.json goes to "cancelled" and nothing
    runs. Do not offer to regenerate; a cancelled campaign is finished. If
    the user wants a different script, they start a new campaign.
  • Because the script cannot be corrected after the fact, get it right in
    ONE generation. Before calling campaign_propose_template, re-read the
    TEMPLATE FORMAT section and check the script against plan.template:
    right material, box, KIM model, phases, durations and crack method.

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

  1. Call campaign_get_state() — check control.status AND template.status.
  2. If control.status == "paused": tell the user "resuming campaign <id>
     from trial <next_iteration>", write control.json's status back to
     "running" by proceeding with the loop (check_stop will pass because
     the pause flag was cleared by the frontend's Resume POST endpoint).
  3. If control.status == "cancelled" or "done": tell the user the
     campaign is closed; offer to generate the report if not already done,
     or start a new campaign.
  4. If control.status == "running" (server was killed mid-loop, restart
     picked up): just re-enter the loop from step 2.
  5. If control.status == "awaiting_template_approval" (or template.status
     == "pending"): the campaign stopped before its first trial, waiting on
     the user. Call campaign_propose_template() — with NO simulate() first —
     to re-show the staged template, then END YOUR TURN. Do NOT regenerate
     it; the copy on disk is the one awaiting approval.

Whenever template.status == "approved", resume with campaign_fill_template.
NEVER call simulate() on a resume: the campaign already has its template, and
a fresh script would not be comparable to the trials already recorded.

The plan on disk + trials.jsonl are the truth — everything you need to
continue is in campaign_get_state()'s return.

══════════════════════════════════════════════════════════════════════════════
FAILURE HANDLING SUMMARY
══════════════════════════════════════════════════════════════════════════════

Every campaign_* tool returns errors as `{"error": "..."}` dicts — they
never raise. Your job is to react to errors as data, not treat them as
crashes:

  • simulate() failure during template genesis (no lammps_script or
    review_passed=false): this is BEFORE any trial exists, so do NOT record
    a trial. Retry simulate() once; if it fails again, tell the user the
    campaign could not be set up and stop.

  • campaign_fill_template failure (bad knob name, out-of-range value):
    your arguments were wrong and no simulation ran. Fix the values and call
    it again. Do NOT record a trial — a rejected fill is not a failed
    experiment, and recording it would corrupt the consecutive-failure count.

  • campaign_run_simulation returns {"error": ...} (a filled template that
    LAMMPS rejects or that crashes):
    → record_trial(metrics=None, failure_reason="lammps_error: <msg>")
    → check_stop → continue. Do NOT regenerate the template. If the template
      is systematically broken, three consecutive failures will stop the
      campaign cleanly and the report will show why.

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
