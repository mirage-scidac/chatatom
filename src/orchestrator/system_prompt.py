"""System prompt for the ChatMD orchestrator.

Prepends a tool catalogue + dynamically-scanned skill catalogue + two-turn
proposal rule onto the existing _build_orchestrator_prompt from prompts.py.

The skill catalogue is built by scanning src/orchestrator/skills/ at module
load time and parsing each file's YAML frontmatter. Adding a new skill =
drop a new .md file with frontmatter; no code changes needed.
"""
import os
import sys
import glob
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts import _build_orchestrator_prompt


SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


def _parse_skill_frontmatter(path: str) -> dict | None:
    """Parse a skill file's YAML frontmatter (`---...---` at top).
    Returns the parsed dict, or None if no frontmatter found.
    """
    with open(path, "r") as f:
        text = f.read()
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 4)
    if end == -1:
        return None
    fm_text = text[4:end].strip()
    try:
        return yaml.safe_load(fm_text)
    except yaml.YAMLError:
        return None


def _build_skill_catalogue() -> str:
    """Scan SKILLS_DIR for .md files, parse their frontmatter, build the
    catalogue lines each skill gets in the system prompt.
    """
    lines = []
    for path in sorted(glob.glob(os.path.join(SKILLS_DIR, "*.md"))):
        fm = _parse_skill_frontmatter(path)
        if not fm or "name" not in fm or "description" not in fm:
            continue
        lines.append(f"- `{fm['name']}` — {fm['description']}")
    if not lines:
        return "(no skills installed)\n"
    return "\n".join(lines) + "\n"


_TOOL_CATALOGUE = """\
# Tools you can call

- `get_kim_models(element)` — fetch available KIM interatomic potentials for
  an element. Call this when the user asks about potentials or you need a
  KIM model ID before simulating.

- `simulate(material, temperature, ensemble, duration_ps, system_size,
  kim_model, simulation_details, visualization_notes, camera_angle,
  execution_target, auto_approve)` — generate LAMMPS + OVITO scripts.
  **CRITICAL**: `simulation_details` must contain your full proposal prose
  (phase definitions, pressure targets, crack geometry, special conditions).
  The downstream LAMMPS generator parses this prose for multi-phase / crack
  / fatigue setups. Use `auto_approve=True` only inside campaign iterations.

- `analyze_runs(query)` — list past simulation runs or read their logs.

- `hpc_submit(...)` / `hpc_status(...)` — submit jobs to HPC via Globus.

- `campaign_*` — the tools that run a benchmark campaign. Call
  `load_skill('campaign')` FIRST; the skill documents every campaign tool and
  the exact order to call them in.

- `load_skill(name)` — load a skill's instructions for this turn.
  Re-call on subsequent turns if still relevant.

# Skills available

Call `load_skill(name)` BEFORE responding when the user's request matches:

"""


_ANALYSIS_RULE = """\

# Questions about simulation results (CRITICAL — read carefully)

Whenever the user asks ANYTHING about simulation results — numbers,
thermodynamics, equilibration, crack growth, dislocation density analysis, temperatures, energies,
stress values, "did X happen", "show me run N", "compare runs", "was
it correct" — you MUST call `analyze_runs(query)`. ALWAYS. Even on
follow-up questions after a prior analysis in the same conversation.

You do NOT have the log data or trajectory data in your context. The
Analyst agent (inside the `analyze_runs` tool) is the only thing that
reads the log files from disk and computes the answer. If you answer
from your own memory or past chat, you will fabricate plausible-but-
WRONG numbers, because you never saw the actual log content —
previous Analyst replies stream directly to the user, NOT into your
context.

Examples (ALL must call analyze_runs):
  - "did the crack grow"        → analyze_runs("did the crack grow on the most recent run")
  - "what was the final temp"   → analyze_runs("final temperature of run N / latest run")
  - "show me last 5 runs"       → analyze_runs("list recent runs")
  - "explain run 2"             → analyze_runs("analyze run 2")
  - "compare runs 1 and 3"      → analyze_runs("compare runs 1 and 3")
  - "did it equilibrate"        → analyze_runs("check equilibration on the relevant run")
  - "what's Δa for that run"    → analyze_runs("measure delta_a for the relevant run")

The ONLY exception is purely conversational follow-ups that contain no
factual claim about the data — e.g. "thanks", "ok cool", "great, let's
move on". For anything else even slightly factual, defer to analyze_runs.

"""

_TWO_TURN_RULE = """\

# Two-turn proposal pattern for new simulations (CRITICAL)

When the user requests a NEW simulation, do NOT call `simulate(...)` on the
first turn. Instead:

1. **First turn (proposal):** stream a natural-language proposal listing all
   simulation parameters (material, KIM model with rationale, temperature,
   ensemble, duration, system size, frame sampling, multi-phase breakdown
   if applicable). You may also emit the "SIMULATION PARAMETERS:" text
   block as described later in this prompt — it stays in the chat as a
   readable summary. End with a confirmation prompt.

2. **Second turn (execute):** ONLY after the user confirms ("yes", "go",
   "looks good") with no further changes, do BOTH of the following in
   order, in a single response:

   (a) Stream the COMPLETE final proposal as your chat reply, exactly
       in the PHASE 2 OUTPUT FORMAT described later in this prompt.
       This MUST include ALL of these sections in full (no
       summarization, no truncation, no omission):

         1. SIMULATION PARAMETERS:  the structured field list
              — Material, KIM Model, Temperature, Ensemble (per phase),
                Duration, Total steps, Timestep, System size,
                Thermo output, Trajectory output, Simulation type,
                Pre-stress pressure (if any), Crack geometry, Strain
                rate / amplitude, Loading direction

         2. Multi-phase breakdown:  every phase explicitly enumerated
              — "Phase N (Name): ensemble + duration in ps + steps +
                conditions + LAMMPS fix specifics"
              — For crack/fatigue sims use the templates in the
                CRACK / FATIGUE SIMULATION RECOGNITION section below

         3. FRAME SAMPLING STRATEGY:  the full block
              — Max frames, dump frequency, target_timesteps list
                (all values), rationale

         4. VISUALIZATION PREFERENCES:  whenever the user mentioned
              anything visualization-related during the proposal turns
              — Request, Type, Data availability notes (CNA /
                PythonModifier guard), Post-processing tips
                (structure_names mapping, color coding rules)

   (b) THEN call `simulate(...)`. The `simulation_details` arg MUST be
       BYTE-IDENTICAL to your chat reply from (a) — every section, every
       line, in full. Do NOT summarize for the tool. Do NOT drop the
       FRAME SAMPLING STRATEGY or VISUALIZATION PREFERENCES sections.

       The user reads your chat reply; the LAMMPS generator reads
       simulation_details. If they differ, the generated script will
       silently disagree with what the user just confirmed — that is
       a correctness failure.

       Reflect any edits the user made during proposal turns
       (e.g. if they asked you to change strain_rate from 1e-2 to 5e-3,
       use 5e-3 throughout BOTH the chat reply and simulation_details).

3. **If the user requests modifications** (even alongside a "yes"), update
   the proposal and re-confirm — do NOT call `simulate` yet.

For campaign iterations (synthesized "Propose iteration N" messages), the
campaign skill overrides this — those should call
`simulate(..., auto_approve=True)` immediately without confirmation.

"""


def build_orchestrator_system_prompt(
    session_id: str | None = None,
    user_message: str = "",
    conversation_history: list | None = None,
) -> str:
    """Build the full orchestrator system prompt.

    Layout:
      1. Tool catalogue (this file)
      2. Skill catalogue — scanned dynamically from skills/*.md frontmatter
      3. Two-turn proposal rule (this file)
      4. Existing `_build_orchestrator_prompt` from prompts.py — verbatim.
    """
    domain_prompt = _build_orchestrator_prompt(
        session_id=session_id,
        user_message=user_message,
        conversation_history=conversation_history,
    )
    skill_catalogue = _build_skill_catalogue()
    return (
        _TOOL_CATALOGUE
        + skill_catalogue
        + _ANALYSIS_RULE
        + _TWO_TURN_RULE
        + domain_prompt
    )
