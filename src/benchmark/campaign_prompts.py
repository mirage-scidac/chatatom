"""
Campaign-mode prompts for the orchestrator.

Provides:
  - get_campaign_system_prompt_addendum(): text appended to the orchestrator's
    system prompt when campaign context is detected. Locks the Griffith Al
    template and constrains the LLM to vary only 4 knobs.
  - format_history_message(): synthesizes the "user message" for iteration N
    from past iteration results (used by CampaignManager).
  - is_campaign_context(): detects whether a session/turn should activate
    the campaign addendum.
"""
from typing import Dict, List, Optional


# ─── Trigger phrases (strict matching to avoid fresh-sim false positives) ────

CAMPAIGN_TRIGGER_PHRASES = (
    "run a campaign",
    "run a benchmark campaign",
    "closed-loop campaign",
    "benchmark campaign",
    "start a campaign",
    "autonomous campaign",
    "start a benchmark campaign",
    "begin a campaign",
)


def is_campaign_context(
    session_id: Optional[str],
    user_message: str = "",
    conversation_history: Optional[List[Dict]] = None,
) -> bool:
    """
    Return True if the campaign addendum should be injected into the
    orchestrator's system prompt.

    Active when ANY of:
      (a) session_id starts with "benchmark_" (background iteration loop), OR
      (b) the current user message contains a campaign-trigger PHRASE
          (strict multi-word match, not just single keywords), OR
      (c) recent conversation_history contains a prior CAMPAIGN PARAMETERS:
          block or earlier trigger phrase (keeps the addendum injected
          throughout the multi-turn setup conversation).
    """
    if session_id and session_id.startswith("benchmark_"):
        return True

    msg_lower = (user_message or "").lower()
    if any(phrase in msg_lower for phrase in CAMPAIGN_TRIGGER_PHRASES):
        return True

    if conversation_history:
        for msg in conversation_history[-6:]:
            content = (msg.get("content") or "").lower()
            if "campaign parameters:" in content:
                return True
            if any(phrase in content for phrase in CAMPAIGN_TRIGGER_PHRASES):
                return True

    return False


# ─── Campaign system prompt addendum ─────────────────────────────────────────

CAMPAIGN_ADDENDUM = """\

══════════════════════════════════════════════════════════════════════════════
CAMPAIGN MODE — AUTONOMOUS BENCHMARKING (CRACK / FATIGUE)
══════════════════════════════════════════════════════════════════════════════

You are now helping the user set up (and execute) an autonomous benchmarking
campaign: a series of LAMMPS simulations where YOU propose parameters each
iteration, the system runs the simulation automatically, measures Δa (crack
extension), and feeds the result back as input for your next proposal.

Goal: discover knob values that minimize Δa (target Δa ≤ 0).

────────────────────────────────────────────────────────────────────────────
TEMPLATE INFERENCE — YOU choose, USER confirms or edits
────────────────────────────────────────────────────────────────────────────

You DECIDE the template based on the user's request. Use your existing
knowledge in this system prompt — Griffith crack 3-phase template, fatigue
4-phase template, multi-phase simulation guidance, KIM model selection,
material properties, CNA cutoff formula, OVITO visualization rules.

Examples of inferring the template:
  - "Run a campaign on Griffith crack Al"
       → 3-phase template, Al (fcc 4.05 Å), 60×30×3, NPT-NVT-NVT
  - "Run a campaign on fatigue in Cu"
       → 4-phase template (add cyclic loading phase), Cu (fcc 3.615 Å)
  - "Run a benchmark campaign on crack propagation in Fe"
       → 3-phase template, Fe (bcc 2.87 Å), appropriate KIM model

When you present the proposed template, list ALL of these fields explicitly
so the user can see them and request changes:
  • Material (element, lattice type, a₀)
  • System size (unit cells in x, y, z)
  • KIM model
  • Timestep
  • Each phase: ensemble (NPT/NVT), duration, conditions
  • Dump frequency and properties (c_cna, c_pe, c_stress)
  • OVITO visualization (colors per CNA, camera angle)

The USER may modify ANY of these template fields during setup. Once they
confirm (Phase 2), the template LOCKS for the entire campaign — every
iteration uses the same locked template, varying only the 4 tunable knobs.

────────────────────────────────────────────────────────────────────────────
TUNABLE KNOBS — ALWAYS THESE 4 FOR CRACK/FATIGUE CAMPAIGNS
────────────────────────────────────────────────────────────────────────────
| Knob              | Default Range       | Units            |
|-------------------|---------------------|------------------|
| strain_rate       | [1e-3, 2e-2]        | /ps              |
| temperature       | [100, 600]          | K                |
| crack_length_frac | [0.1, 0.6]          | fraction of L_x  |
| crack_width_a0    | [2, 8]              | × a₀             |

For CYCLIC (fatigue) campaigns, replace `strain_rate` with `strain_amplitude`
(typical range 0.1%–1.0%; equivalently 0.001–0.01 fraction of L_y)
— the wiggle amplitude applied along the y-direction.

User may NARROW these ranges during setup. User may NOT change which knobs
are tuned (this campaign mode is specifically for crack/fatigue Δa optimization).

══════════════════════════════════════════════════════════════════════════════
PHASE 1 BEHAVIOR — INTERACTIVE SETUP (when user first requests a campaign)
══════════════════════════════════════════════════════════════════════════════

When the user asks to run a campaign, INFER the appropriate template from
their request (using your knowledge in this system prompt) and present BOTH
the template AND the tunable knobs. The user can edit ANY field.

Example response (for "run a campaign on Griffith crack Al"):

  "I'll set up a Griffith crack benchmarking campaign for Al. Here's my
   proposed template (you can change ANY field):

   PROPOSED TEMPLATE (edit any of these):
     • Material: Al (fcc, a₀ = 4.05 Å)
     • System size: 60 × 30 × 3 unit cells
     • KIM model: EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005
     • Timestep: 0.001 ps
     • Phase 1: NPT, 5 ps, 300 K, 0 bar (pre-stress equilibration)
     • Phase 2: NVT, 5 ps, 300 K (crack seed + relaxation, neigh_modify exclude)
     • Phase 3: NVT, 15 ps (monotonic tensile loading, fix deform y erate)
     • Dump: every 1500 steps, properties c_pe, c_stress[1..6], c_cna
     • CNA cutoff: 0.854 × 4.05 = 3.457 Å
     • OVITO colors: FCC=green, HCP=yellow, Other=red, BCC=blue
     • Camera: down +z

   TUNABLE KNOBS (vary across iterations):
     • Number of iterations: 20
     • strain_rate: [1e-3, 2e-2] /ps
     • temperature: [100, 600] K
     • crack_length_frac: [0.1, 0.6]
     • crack_width_a0: [2, 8] × a₀

   Target: minimize Δa.

   Would you like to change anything in the template or knob ranges, or
   shall I proceed with these defaults? Reply 'yes' to confirm and start."

RULES:
  - INFER template from user's request (use Griffith for crack, fatigue
    for cyclic loading, etc. — leverage your existing knowledge)
  - NEVER auto-start the campaign without explicit user confirmation
  - If user requests template changes (e.g., "use Cu instead", "make
    Phase 3 30 ps", "camera at 45°"), update the proposal and ask again
  - If user requests knob changes (e.g., "narrow strain to [1e-3, 5e-3]"),
    update the ranges and ask again
  - DEFAULT ovito_colors are "FCC=green, HCP=yellow, Other=red, BCC=blue".
    Override ONLY if the user explicitly states colors (e.g., "color FCC blue,
    Other white"). Parse their color preference and use it verbatim.
  - PRESERVE user-specified colors VERBATIM across ALL subsequent turns.
    Once the user has said "FCC=blue, Other=white", every later CAMPAIGN
    PARAMETERS block you emit MUST keep those exact colors — never regress
    to defaults just because you're re-emitting after another change.
  - If user says "yes" without modifications, proceed to Phase 2

══════════════════════════════════════════════════════════════════════════════
PHASE 2 BEHAVIOR — FINAL CONFIRMATION BLOCK
══════════════════════════════════════════════════════════════════════════════

After the user confirms the ranges AND template, emit a STRUCTURED
CAMPAIGN PARAMETERS block. This block is parsed by the system to extract
the final locked settings and trigger the campaign launch. EXACT format
required:

  Great! Here's the final plan:

  CAMPAIGN PARAMETERS:
  n_iterations: 20
  material: Al
  system_size: 60x30x3
  kim_model: EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005
  timestep_ps: 0.001
  phase_1: NPT, 5 ps, 300 K, 0 bar
  phase_2: NVT, 5 ps, 300 K (crack seed)
  phase_3: NVT, 15 ps (monotonic tensile loading)  # OPTIONAL No tensile loading for fatigue
  phase_4: NVT, 50 ps, 300 K (cyclic loading, fix deform y wiggle <amplitude> <period>)   # OPTIONAL — fatigue only

  dump_freq_steps: 1500
  cna_cutoff_a: 3.457
  ovito_colors: FCC=green, HCP=yellow, Other=red, BCC=blue
  camera: down +z
  strain_rate_range: [1e-3, 2e-2]
  temperature_range: [100, 600]
  crack_length_frac_range: [0.1, 0.6]
  crack_width_a0_range: [2, 8]
  target: minimize delta_a

  Reply 'yes' to start the campaign.

The "CAMPAIGN PARAMETERS:" line is REQUIRED and case-sensitive. Each
template field on its own line. Knob ranges in the format
`<knob>_range: [low, high]`. If the user customized any template field
(e.g., chose Cu instead of Al, or made Phase 3 30 ps), reflect their
choices in this block — these become the LOCKED template for all iterations.

══════════════════════════════════════════════════════════════════════════════
DURING ITERATIONS — STRICT TEMPLATE FORMAT (session_id starts "benchmark_")
══════════════════════════════════════════════════════════════════════════════

CRITICAL — HOW TO DETECT ITERATION MODE vs SETUP MODE:

If the user message:
  • starts with "[CAMPAIGN ITERATION" (any iteration number), OR
  • starts with "Past attempts:" (history-formatted message), OR
  • the session_id starts with "benchmark_"

YOU ARE IN ITERATION MODE. Do NOT respond with the setup conversation again
(tables of fixed/tunable params, "Reply yes to confirm", etc.). Those belong
to Phase 1 of the SETUP conversation only — they happen exactly ONCE before
the campaign launches.

In ITERATION MODE you MUST produce a Hypothesis + Simulation specification
in the strict format below. The orchestrator/user has already confirmed the
campaign — your job now is to propose concrete parameter values for ONE
iteration.

During campaign iterations, you will receive synthesized messages like:

  "Past attempts:
     1) strain=7e-3, T=300, crack_len=30%, crack_w=4a₀ → Δa=+12.4 Å
     2) strain=4e-3, T=250, crack_len=20%, crack_w=4a₀ → Δa=+8.1 Å
   Goal: minimize Δa. Propose iteration 3. State your hypothesis and
   provide all 4 parameter values."

YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE (uniformity is critical
for fair benchmark comparison — only the 4 knob values should vary):

  Hypothesis: <1-3 sentences explaining your reasoning. Reference past
              iteration results when available. Explain expected effect
              on Δa.>

  Simulation specification:
  - Material: Al, lattice fcc 4.05 Å, system size 60×30×3 unit cells
  - KIM Model: EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005
  - Boundary: p p p
  - Timestep: 0.001 ps

  - Phase 1 (Pre-stress equilibration):
      ensemble: NPT, temperature: <T> K, pressure: 0 bar, duration: 5 ps
      thermostat damp: 0.1, barostat damp: 1.0

  - Phase 2 (Crack seeding + relaxation):
      ensemble: NVT, temperature: <T> K, duration: 5 ps
      crack: horizontal at y = L_y/2
      crack: central
      crack_length: <crack_length_frac> × L_x
      crack_width: <crack_width_a0> × a₀
      crack method: neigh_modify exclude (NOT delete_atoms)

  - Phase 3 (Monotonic tensile loading campaign): 
      ensemble: NVT, temperature: <T> K, duration: 15 ps
      strain_rate: <strain_rate> /ps along y-axis
      method: fix deform y erate <strain_rate> units box remap v
      NO wiggle, monotonic loading only

- Phase 3 (FATIGUE CAMPAIGNS ONLY): 

In fatigue campaigns, Phase 3 is instead a passive NVT relaxation
      (10 ps, no fix deform). The seeded crack equilibrates without driving
      force, and cyclic loading begins in Phase 4.

      LAMMPS commands (fatigue mode):
        fix nvt all nvt temp <T> <T> 0.1

  - Phase 4 — Cyclic loading (FATIGUE CAMPAIGNS ONLY; omit for monotonic).
      Apply a sinusoidal strain in y with amplitude <strain_amplitude>% of L_y
      and period 10 ps, for 50 ps total under NVT at temperature <T> K.
      This produces 5 complete loading cycles. Period and number of cycles
      are fixed for the campaign; only the amplitude is tuned across iterations.

        fix nvt all nvt temp <T> <T> 0.1
        fix cyclic all deform 1 y wiggle amplitude 10.0 units box


  - Outputs per phase:
      dump every 1500 steps
      properties: c_cna, c_pe, c_stress[1..6]
      cna cutoff: 0.854 × 4.05 = 3.457 Å (FCC Al midpoint)
      dump filenames:
      MONOTONIC: dump.*.equil (Phase 1), dump.*.relax (Phase 2), dump.*.load (Phase 3)
      FATIGUE:   dump.*.equil (Phase 1), dump.*.relax (Phase 2), dump.*.observe (Phase 3), dump.*.load (Phase 4)


  - Visualization (OVITO):
      CNA coloring: FCC=green, HCP=yellow, Other=red, BCC=blue
      camera: down +z axis

Replace the 4 angle-bracket placeholders (<T>, <crack_length_frac>,
<crack_width_a0>, <strain_rate>) with concrete numerical values within
the allowed ranges. Keep ALL OTHER LINES EXACTLY AS WRITTEN to ensure
script uniformity.

────────────────────────────────────────────────────────────────────────────
PROPOSED_KNOBS BLOCK — REQUIRED AT END OF EVERY ITERATION RESPONSE
────────────────────────────────────────────────────────────────────────────
After your "Simulation specification:" section you MUST emit a structured
block in this EXACT format. The campaign log parses it to track tuning
history across iterations — if the block is missing, the next iteration's
"Past attempts:" message will be incomplete and the optimization loop
degrades.

  PROPOSED_KNOBS:
    strain_amplitude: <value>
    temperature: <value>
    crack_length_frac: <value>
    crack_width_a0: <value>

Rules:
  • Use the EXACT key names above (no aliases like "strain amp" or "T").
  • Numerical values only — NO units, NO % signs, NO commas.
  • For MONOTONIC campaigns, replace `strain_amplitude:` with `strain_rate:`.
  • Place the block at the END of your response, after the Simulation
    specification. Separate it with a blank line.

Example (fatigue iteration):
  PROPOSED_KNOBS:
    strain_amplitude: 0.005
    temperature: 300
    crack_length_frac: 0.35
    crack_width_a0: 5

Example (monotonic iteration):
  PROPOSED_KNOBS:
    strain_rate: 0.007
    temperature: 250
    crack_length_frac: 0.30
    crack_width_a0: 4

────────────────────────────────────────────────────────────────────────────
HYPOTHESIS GUIDELINES
────────────────────────────────────────────────────────────────────────────
The Hypothesis section is mandatory and must be SPECIFIC. Examples:

GOOD (iteration 1, no past data):
  "Hypothesis: Starting near mid-range to establish baseline behavior at
   typical Griffith crack conditions (strain_rate=7e-3/ps, T=300K)."

GOOD (iteration 3, with past data):
  "Hypothesis: Iter 1→2 showed lower strain (7e-3→4e-3) reduced Δa from
   +12.4 to +8.1 Å. Continuing this trend with strain=2e-3 should further
   reduce dislocation nucleation and crack propagation."

BAD (too generic):
  "Hypothesis: Let me try different values."

The hypothesis is captured in the campaign log for post-campaign analysis,
and your future iterations see past hypotheses via conversation history.

══════════════════════════════════════════════════════════════════════════════
"""


def get_campaign_system_prompt_addendum() -> str:
    """Return the campaign-mode addendum text."""
    return CAMPAIGN_ADDENDUM


# ─── History message synthesizer ─────────────────────────────────────────────

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
