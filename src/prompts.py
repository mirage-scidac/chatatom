"""
System Prompts and Knowledge Base for ChatMD Agents.

This module centralizes all LLM system prompts used by the agent classes,
making them easier to maintain, version, and test independently.

It also contains DSPy signatures for the agentic workflow.
"""

from typing import List, Literal, Optional
import dspy


# =============================================================================
# DSPy Signatures for Agentic Workflow
# =============================================================================

class AnalyzeRequest(dspy.Signature):
    """Analyze a user's simulation request and extract structured parameters.

    Given a natural language request for a molecular dynamics simulation,
    extract the key parameters and determine what additional information
    is needed from the user.
    """
    user_request: str = dspy.InputField(desc="Natural language simulation request from user")
    registry_context: str = dspy.InputField(desc="Available KIM models from registry", default="")

    material: str = dspy.OutputField(desc="Element or compound to simulate (e.g., Cu, Al, Si)")
    temperature: float = dspy.OutputField(desc="Target temperature in Kelvin")
    ensemble: str = dspy.OutputField(desc="Thermodynamic ensemble: NVE, NVT, or NPT")
    duration_ps: float = dspy.OutputField(desc="Simulation duration in picoseconds")
    kim_model: str = dspy.OutputField(desc="Selected KIM model ID from registry")
    system_size: int = dspy.OutputField(desc="Number of unit cells per dimension", default=5)
    needs_clarification: bool = dspy.OutputField(desc="Whether more info is needed from user")
    clarification_question: str = dspy.OutputField(desc="Question to ask user if clarification needed", default="")
    confidence: float = dspy.OutputField(desc="Confidence score 0.0-1.0 for the analysis")


class VerifySimulationPlan(dspy.Signature):
    """Verify that a simulation plan is feasible before generation.

    Check that the KIM model is available, parameters are physically reasonable,
    and the simulation can be executed successfully.
    """
    material: str = dspy.InputField(desc="Element or compound")
    kim_model: str = dspy.InputField(desc="KIM model ID to use")
    temperature: float = dspy.InputField(desc="Target temperature in Kelvin")
    ensemble: str = dspy.InputField(desc="Thermodynamic ensemble")
    duration_ps: float = dspy.InputField(desc="Simulation duration in picoseconds")
    system_size: int = dspy.InputField(desc="Unit cells per dimension")

    is_feasible: bool = dspy.OutputField(desc="Whether the simulation can be executed")
    issues: List[str] = dspy.OutputField(desc="List of issues found, empty if none")
    suggestions: List[str] = dspy.OutputField(desc="Suggestions for fixing issues")
    confidence: float = dspy.OutputField(desc="Confidence score 0.0-1.0")


class GenerateLAMMPSScript(dspy.Signature):
    """Generate a complete LAMMPS input script from structured parameters.

    Create an executable LAMMPS script using KIM potentials. Follow the
    knowledge_context for KIM command sequence, crack techniques, cyclic
    loading rules, and output requirements. Read simulation_details for
    phase structure, durations, and geometry.

    Output ONLY raw LAMMPS script content — no markdown, no code block markers.
    """
    material: str = dspy.InputField(desc="Element or compound")
    kim_model: str = dspy.InputField(desc="KIM model ID")
    temperature: float = dspy.InputField(desc="Target temperature in Kelvin")
    ensemble: str = dspy.InputField(desc="Thermodynamic ensemble: NVE, NVT, or NPT")
    duration_ps: float = dspy.InputField(desc="Simulation duration in picoseconds")
    system_size: int = dspy.InputField(desc="Unit cells per dimension")
    thermo_freq: int = dspy.InputField(desc="Thermodynamic output frequency in steps", default=100)
    dump_freq: int = dspy.InputField(desc="Trajectory dump frequency in steps", default=1000)
    visualization_notes: str = dspy.InputField(desc="Notes about visualization requirements", default="")
    simulation_details: str = dspy.InputField(
        desc="Full simulation details from orchestrator including phases, crack geometry, cyclic loading, etc.",
        default=""
    )
    past_lessons: str = dspy.InputField(
        desc="Lessons from previous similar simulations to avoid repeating mistakes",
        default=""
    )
    knowledge_context: str = dspy.InputField(
        desc="Domain knowledge: KIM sequence, atomic masses, crack/cyclic techniques, output requirements",
        default=""
    )
    review_feedback: str = dspy.InputField(
        desc="Issues and fix instructions from the reviewer on previous attempts. "
             "You MUST address every issue listed here. Empty on first attempt.",
        default=""
    )

    lammps_script: str = dspy.OutputField(desc="Complete LAMMPS input script")
    explanation: str = dspy.OutputField(desc="Brief explanation of key script features")
    dump_filename_pattern: str = dspy.OutputField(
        desc="The dump filename pattern used in the script (e.g., 'dump.Cu_solidification.*.dat' or 'traj.lammpstrj'). "
             "For multi-phase simulations with separate dumps per phase, report a UNIFIED glob pattern that matches ALL "
             "dump files (e.g., 'dump.*.dat' to match dump.*.prestress AND dump.*.cracked)."
    )
    total_steps: int = dspy.OutputField(
        desc="Total number of simulation steps across all run commands combined"
    )
    dump_freq_val: int = dspy.OutputField(
        desc="The trajectory dump frequency in timesteps set within the script (e.g. 10000)"
    )


class ReviewLAMMPSScript(dspy.Signature):
    """Review a LAMMPS script for correctness and best practices.

    REJECT if: pair_style/pair_coeff used, kim init after create_box,
    kim interactions before create_box, missing mass command,
    NVT/NPT without velocity initialization, NPT combined with fix deform.

    CHECK: Pdamp ~1000x timestep, CNA cutoff ~0.854*a0 (midpoint of 1st and 2nd NN),
    crack regions defined AFTER create_atoms, neigh_modify exclude
    (NOT delete_atoms) for Griffith cracks, unfix/undump hygiene
    between phases, pe/atom and stress/atom in crack dump commands,
    velocity temperature matches thermostat temperature.
    """
    lammps_script: str = dspy.InputField(desc="LAMMPS script to review")
    intended_parameters: str = dspy.InputField(desc="Description of intended simulation", default="")
    knowledge_context: str = dspy.InputField(
        desc="Domain knowledge: KIM rules, crack technique details, correct parameter ranges",
        default=""
    )

    is_valid: bool = dspy.OutputField(desc="Whether the script passes review")
    issues: List[str] = dspy.OutputField(desc="List of issues found")
    suggestions: List[str] = dspy.OutputField(desc="Suggestions for improvement")
    severity: str = dspy.OutputField(desc="Highest issue severity: none, warning, or error")


class ReviewOVITOScript(dspy.Signature):
    """Review an OVITO Python script for correctness and best practices.

    REJECT if: unauthorized imports (os/sys/subprocess/open/eval/exec),
    missing 'pipeline' variable assumption, deprecated OVITO API usage,
    PythonModifier without per-frame guards, property creation via
    one-off pipeline.compute() outside a modifier.

    CHECK: proper import statements, PythonModifier function signature
    (frame, data), guard pattern for pre-computed properties ('Structure Type'
    — OVITO auto-maps LAMMPS c_cna to this name), auto_adjust_range=False
    for stable color mapping, correct structure type mapping
    (0=Other, 1=FCC, 2=HCP, 3=BCC, 4=ICO).
    """
    ovito_script: str = dspy.InputField(desc="OVITO Python script to review")
    visualization_request: str = dspy.InputField(desc="Original visualization request", default="")
    knowledge_context: str = dspy.InputField(desc="OVITO technical guidelines", default="")

    is_valid: bool = dspy.OutputField(desc="Whether the script passes review")
    issues: List[str] = dspy.OutputField(desc="List of issues found")
    suggestions: List[str] = dspy.OutputField(desc="Suggestions for improvement")
    severity: str = dspy.OutputField(desc="Highest issue severity: none, warning, or error")


class SelectTools(dspy.Signature):
    """Select appropriate tools for handling a user request.

    Given the current context and available tools, decide which
    tools to invoke and in what order.
    """
    user_request: str = dspy.InputField(desc="Current user request")
    available_tools: List[str] = dspy.InputField(desc="List of available tool names")
    conversation_context: str = dspy.InputField(desc="Recent conversation history", default="")

    selected_tools: List[str] = dspy.OutputField(desc="Tools to invoke in order")
    tool_arguments: List[str] = dspy.OutputField(desc="Arguments for each selected tool")
    reasoning: str = dspy.OutputField(desc="Explanation for tool selection")


class ClassifyUserIntent(dspy.Signature):
    """Classify the user's intent in the context of a molecular dynamics simulation workflow.

    The workflow stages are:
    - 'initial': User is making a new simulation request
    - 'parameters_proposed': Simulation parameters have been proposed and await user feedback
    - 'script_ready': A LAMMPS script has been generated and reviewed, ready for execution

    Classify intent as one of:
    - 'execute': User wants to run/execute the simulation (e.g. "run", "go ahead", "yes", "sure", "looks good", "this is fine")
    - 'modify': User wants to change parameters (e.g. "change temperature to 500K", "make it bigger")
    - 'cancel': User wants to stop/cancel (e.g. "no", "stop", "cancel", "nevermind")
    - 'question': User is asking a question (e.g. "what does NVT mean?", "explain the script")
    - 'new_simulation': User is requesting a completely new/different simulation
    """
    user_message: str = dspy.InputField(desc="The user's message")
    workflow_stage: str = dspy.InputField(desc="Current workflow stage: 'initial', 'parameters_proposed', or 'script_ready'")

    intent: Literal['execute', 'modify', 'cancel', 'question', 'new_simulation'] = dspy.OutputField(
        desc="Exactly one of: 'execute', 'modify', 'cancel', 'question', 'new_simulation'"
    )
    confidence: float = dspy.OutputField(desc="Confidence score 0.0-1.0")
    extracted_modifications: str = dspy.OutputField(desc="If intent is 'modify', describe what changes are requested. Otherwise empty string.")


class ExtractFeedback(dspy.Signature):
    """Extract structured lessons from user feedback about simulation results.

    Given user feedback about how a simulation turned out and a summary of
    what was simulated, extract a concise lesson that can be stored and
    retrieved to improve future simulations.
    """
    user_feedback: str = dspy.InputField(desc="User's feedback about the simulation result")
    simulation_summary: str = dspy.InputField(desc="Summary: material, parameters, what was requested")

    category: str = dspy.OutputField(desc="One of: 'lammps_issue', 'visualization_issue', 'success', 'preference'")
    lesson: str = dspy.OutputField(desc="Concise lesson learned for future simulations")
    keywords: str = dspy.OutputField(desc="Comma-separated keywords for retrieval: material, technique, modifier names")


class ExtractSimulationParameters(dspy.Signature):
    """Extract structured simulation parameters from the orchestrator's analysis text.

    The analysis_text is the orchestrator's output after multi-turn conversation
    with the user, containing all finalized parameter selections.

    Set has_parameters=True only if the text contains a finalized set of
    simulation parameters ready for script generation, not just a proposal
    or clarification question.
    """
    analysis_text: str = dspy.InputField(
        desc="Complete orchestrator analysis text including user refinements"
    )
    has_parameters: bool = dspy.OutputField(
        desc="True if text contains finalized parameters ready for generation, "
             "False if still proposing or asking questions"
    )
    material: str = dspy.OutputField(desc="Element or compound (e.g. Cu, Al). Empty if not finalized.")
    kim_model: str = dspy.OutputField(
        desc="Complete KIM model ID exactly as written in the text. Empty if not selected."
    )
    temperature: float = dspy.OutputField(desc="Temperature in Kelvin. Default 300.0")
    ensemble: str = dspy.OutputField(desc="NVE, NVT, or NPT. Default NVT")
    duration_ps: float = dspy.OutputField(desc="Duration in picoseconds. Default 10.0")
    system_size: int = dspy.OutputField(desc="Unit cells per dimension. Default 5")
    visualization_notes: str = dspy.OutputField(
        desc="Visualization preferences from the text. Empty if none."
    )
    camera_angle: str = dspy.OutputField(
        desc="Camera viewing angle: 'top' for top-down/bird's-eye view along z-axis, "
             "'side' for side view along x-axis, 'default' for standard 3D perspective. "
             "Default 'default'",
        default="default"
    )


class DetectVisualization(dspy.Signature):
    """Determine whether a user request contains a custom visualization intent.

    A CUSTOM visualization request asks for specific visual analysis like:
    - "show only crystalline atoms"
    - "color by structure type"
    - "highlight defects"
    - "visualize surface atoms"
    - "filter out amorphous regions"

    A request is NOT custom if it is just a general simulation request without
    specific visualization instructions.
    """
    user_request: str = dspy.InputField(desc="Natural language user request")

    is_custom_visualization: bool = dspy.OutputField(desc="True if request contains custom visualization intent, False otherwise")


class GenerateOVITOScript(dspy.Signature):
    """Generate a complete OVITO Python script from a visualization request.

    Output ONLY the Python script — no markdown, no code fences.
    Assume the 'pipeline' variable already exists with data loaded.

    Use PythonModifier with per-frame guards that run every frame.
    Check for pre-computed LAMMPS properties before recomputing.
    When CNA is pre-computed, OVITO auto-maps the 'c_cna' dump column to
    'Structure Type' — use data.particles['Structure Type'] to access it,
    NOT data.particles['c_cna'] (which does not exist after import).
    """
    visualization_request: str = dspy.InputField(desc="Natural language visualization request")
    simulation_context: str = dspy.InputField(desc="Material, temperature, structure info", default="")
    knowledge_context: str = dspy.InputField(desc="OVITO API guidelines and patterns", default="")
    review_feedback: str = dspy.InputField(
        desc="Issues from previous review attempts to fix in this generation. Empty on first attempt.",
        default=""
    )

    ovito_script: str = dspy.OutputField(desc="Complete OVITO Python script")


# =============================================================================
# Knowledge Base
# =============================================================================

class KnowledgeBase:
    """Shared knowledge and guidance snippets used across multiple agents."""

    # KIM potential guidelines shared by Generator and Reviewer
    KIM_MANDATORY_SEQUENCE = """
MANDATORY KIM command sequence:
1. kim init <model_name> metal  (BEFORE create_box)
2. [create simulation box and atoms]
3. kim interactions <species>   (AFTER create_box)
4. mass <type> <mass_value>     (REQUIRED - KIM doesn't set masses automatically)
5. velocity all create <temp> <seed>  (REQUIRED for NVT/NPT - use same temp as thermostat)
"""

    # Common atomic masses (metal units)
    ATOMIC_MASSES = """
Common atomic masses (metal units):
- Argon: 39.948
- Silicon: 28.0855
- Aluminum: 26.9815
- Copper: 63.546
- Gold: 196.966
- Iron: 55.845
- Nickel: 58.6934
"""

    # CNA structure type mapping — LAMMPS vs OVITO encodings
    CNA_STRUCTURE_MAPPING = """
CNA STRUCTURE TYPE ENCODING (LAMMPS vs OVITO):

LAMMPS compute cna/atom output:
  1 = FCC, 2 = HCP, 3 = BCC, 4 = ICO, 5 = Other/Unknown

OVITO CommonNeighborAnalysisModifier (Structure Type property):
  0 = Other/Unknown, 1 = FCC, 2 = HCP, 3 = BCC, 4 = ICO

CRITICAL — OVITO PROPERTY NAME AUTO-MAPPING:
OVITO 3.14+ automatically renames the LAMMPS dump column 'c_cna' to
'Structure Type' on import. The original name 'c_cna' does NOT exist
in data.particles after import. You MUST use 'Structure Type' to access
pre-computed CNA data:

  # CORRECT:
  cna_values = data.particles['Structure Type'][...]
  # WRONG (will raise KeyError):
  cna_values = data.particles['c_cna'][...]

The values remain LAMMPS-encoded (1=FCC, 2=HCP, 3=BCC, 4=ICO, 5=Other),
NOT remapped to OVITO encoding. You MUST remap value 5 → 0 for OVITO
color compatibility:
  import numpy as np
  cna_values = data.particles['Structure Type'][...]
  structure_type = np.where(cna_values >= 5, 0, cna_values)

Without this remapping, OVITO will treat LAMMPS value 5 (unknown) as an
unrecognized type, and filters like 'StructureType == 0' will not match.

For display and statistics, use OVITO's encoding:
structure_names = {0: "Other", 1: "FCC", 2: "HCP", 3: "BCC", 4: "ICO"}
"""

    # CNA cutoff formula for correct nearest-neighbor analysis
    CNA_CUTOFF_FORMULA = """
CNA CUTOFF FORMULA (CRITICAL — MUST USE CORRECT VALUE):

The CNA cutoff must sit BETWEEN the 1st and 2nd nearest-neighbor distances.
For FCC crystals:
  1st NN distance = a0 / sqrt(2)   (12 neighbors)
  2nd NN distance = a0             (6 neighbors)
  Correct CNA cutoff = (1st_NN + 2nd_NN) / 2 = (a0/sqrt(2) + a0) / 2
                      ≈ 0.854 * a0

Per-element values:
  Cu (a0 = 3.615 Å): cutoff = 3.086 Å
  Al (a0 = 4.050 Å): cutoff = 3.457 Å
  Au (a0 = 4.078 Å): cutoff = 3.481 Å
  Ni (a0 = 3.524 Å): cutoff = 3.008 Å
  Ag (a0 = 4.085 Å): cutoff = 3.487 Å

COMMON MISTAKE — DO NOT USE:
  1.207 * a0 → This is TOO LARGE (e.g., 4.365 Å for Cu).
  It includes both 1st and 2nd NN shells (18 neighbors instead of 12),
  causing CNA to classify ALL atoms as "Unknown/Other" (type 5).

LAMMPS command:
  compute cna all cna/atom 3.086    # For Cu with a0 = 3.615 Å
"""

    # OVITO correct import reference
    OVITO_IMPORT_REFERENCE = """
OVITO PYTHON API — CORRECT IMPORTS (use exactly these, no others):

from ovito.modifiers import (
    PythonModifier,                    # For custom per-frame functions
    CommonNeighborAnalysisModifier,    # CNA structural analysis
    ExpressionSelectionModifier,       # Select atoms by expression
    DeleteSelectedModifier,            # Delete selected atoms
    ColorCodingModifier,               # Color by property value
    CentroSymmetryModifier,            # Centro-symmetry parameter
    CoordinationAnalysisModifier,      # Coordination numbers
)
from ovito.pipeline import Pipeline    # Pipeline class (NOT PythonModifier)
from ovito.io import import_file       # File loading
from ovito.vis import Viewport, TachyonRenderer  # Rendering
import numpy as np                     # Numeric arrays

IMPORTANT:
- PythonModifier is in ovito.modifiers, NOT ovito.pipeline
- There is NO ColorMap class in ovito.vis
- PythonModifier function signature: def my_func(frame, data):
- Append via: pipeline.modifiers.append(PythonModifier(function=my_func))
- Do NOT use PythonScriptModifier (deprecated)
- Do NOT import from ovito.plugins (internal)

OVITO SCRIPT RULES:
- The script runs inside an executor where 'pipeline' already exists — do NOT create viewports, renderers, or call render_image() — the rendering system handles that
- Only append modifiers to 'pipeline'
- Do NOT call pipeline.compute() for side effects — use PythonModifier instead
"""

    # OVITO per-frame property guards pattern
    OVITO_PROPERTY_GUARDS = """
PER-FRAME PROPERTY GUARDS (CRITICAL):

OVITO recomputes the entire modifier stack for every requested animation frame. Any property you create just once (e.g., by calling pipeline.compute() outside the modifier chain) will not persist across frames. To avoid "property missing" fallbacks, wrap every derived-property requirement in a modifier that executes per frame:

1. Use PythonModifier (docs/ovito_docs/user_modifier.md) when you simply need to copy, rename, or aggregate existing per-atom data (e.g., accessing 'Structure Type' — OVITO's auto-mapped name for the LAMMPS c_cna column — computing a selection flag, calculating statistics, snapshot logging).
2. Use built-in analysis modifiers (CommonNeighborAnalysisModifier, CentroSymmetryModifier, DislocationAnalysisModifier, etc.) inside that PythonModifier via `data.apply(...)` (in-place) or by appending them directly to the pipeline so they execute each frame.
3. Never rely on a one-off `pipeline.compute()` side-effect to add properties, because the executor will call compute() again for each frame and those properties will vanish.
4. When multiple downstream modifiers depend on the same derived property, install exactly one guard modifier at the *start* of the pipeline to prepare that property, and let later steps assume it exists.

This pattern applies broadly: if you need per-frame selections, classifications, statistics, or color channels, ensure a modifier establishes the required data during every pipeline evaluation.
"""

    # Griffith crack seeding technique
    GRIFFITH_CRACK_TECHNIQUE = """
GRIFFITH CRACK SEEDING TECHNIQUE:
This technique seeds a crack by severing atomic bonds across a crack plane using
neigh_modify exclude. No atoms are deleted — the crack is created by preventing
interactions between atoms on opposite sides of the crack face.

Steps (must come AFTER create_atoms and force field setup):
1. Define geometry variables for crack position and extent:
   variable xmid equal lx/2
   variable xbound_low equal lx/2-2*${a0}
   variable xbound_high equal lx/2+2*${a0}
   variable ybound_low equal ly/2-0.15*ly
   variable ybound_high equal ly/2+0.15*ly

2. Create two regions for the upper and lower crack faces:
   region crack_lower block ${xbound_low} ${xmid} ${ybound_low} ${ybound_high} INF INF units box
   region crack_upper block ${xmid} ${xbound_high} ${ybound_low} ${ybound_high} INF INF units box

3. Create groups from the regions:
   group crack_lower region crack_lower
   group crack_upper region crack_upper

4. Sever bonds across the crack plane:
   neigh_modify exclude group crack_upper crack_lower

CRITICAL CONSTRAINTS:
- Regions/groups MUST be defined AFTER create_atoms (atoms must exist first)
- The crack width (xbound_high - xbound_low) should be ~4 lattice parameters
- The crack length (ybound_high - ybound_low) should be ~30% of the box (0.15 * ly on each side)
- Use 'units box' for region commands when working with box dimensions (lx, ly)
- The INF INF for z means the crack extends through the full thickness
"""

    # Cyclic loading technique for fatigue simulations
    CYCLIC_LOADING_TECHNIQUE = """
CYCLIC LOADING TECHNIQUE (FATIGUE):
Use fix deform with the wiggle option to apply oscillating strain for fatigue simulations.

Command: fix <ID> all deform 1 <dir> wiggle <amplitude> <period>

Key rules:
- MUST use NVT ensemble (NOT NPT) — NPT fights the imposed deformation
- Amplitude is in length units: amplitude = strain_fraction * box_length
  Example: 0.25% strain on a 100 Angstrom box → amplitude = 0.0025 * 100 = 0.25
- Period is in time units (ps for metal units)
- The wiggle applies a sinusoidal displacement: A * sin(2*pi*t/period)

Typical parameters:
- Strain amplitude: 0.1% to 1.0% (0.001 to 0.01 fraction)
- Period: 50 ps to 200 ps
- Total runtime: multiple periods (e.g., 10-100 cycles)

Example for x-direction cycling:
  variable ampL equal 0.0025*lx    # 0.25% strain
  fix 1 all nvt temp ${temp} ${temp} 0.1
  fix 2 all deform 1 x wiggle ${ampL} 100.0   # 100 ps period

CRITICAL: Always define amplitude as a variable using current box length,
not a hardcoded value. The box length may change during pre-stress equilibration.
"""

    # Output requirements for crack analysis
    CRACK_OUTPUT_REQUIREMENTS = """
CRACK SIMULATION OUTPUT REQUIREMENTS:
For proper crack analysis, simulations MUST include these computes and dump fields:

Required computes:
  compute peratom all pe/atom
  compute stress_atom all stress/atom NULL
  compute cna all cna/atom <CUTOFF>

CNA CUTOFF — use correct formula (0.854 * a0):
  Cu (a0=3.615): compute cna all cna/atom 3.086
  Al (a0=4.050): compute cna all cna/atom 3.457
  DO NOT use 1.207*a0 — that is too large and misclassifies all atoms.

Required dump fields (include in ALL dump commands):
  dump <ID> all custom <freq> <file> id type x y z c_peratom c_stress_atom[1] c_stress_atom[2] c_stress_atom[3] c_cna

Where stress components are:
  c_stress_atom[1] = xx (tensile stress along x)
  c_stress_atom[2] = yy (tensile stress along y)
  c_stress_atom[3] = zz (tensile stress along z)

Thermo output should include full stress tensor:
  thermo_style custom step time temp pe press pxx pyy pzz vol lx ly lz

For multi-phase simulations, use separate dump files per phase BUT ensure
all dump filenames share a UNIFIED glob pattern so the monitoring system
can find them all. Use a common prefix and extension:
  dump prestressDump all custom 10000 dump.*.prestress id type x y z c_peratom c_stress_atom[1] c_stress_atom[2] c_stress_atom[3] c_cna
  ...
  undump prestressDump
  dump crackDump all custom 10000 dump.*.cracked id type x y z c_peratom c_stress_atom[1] c_stress_atom[2] c_stress_atom[3] c_cna

For dump_filename_pattern, report a unified pattern that matches ALL phase
dump files. Example: if using dump.*.prestress and dump.*.cracked, report
dump_filename_pattern as 'dump.*.*' so the monitor finds files from all phases.
"""


# =============================================================================
# Orchestrator Agent Prompt
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are a request analyzer for LAMMPS molecular dynamics simulations. You operate in multiple phases:

PHASE 1 - ANALYSIS AND PROPOSAL:
When analyzing an initial user request, your role is to:
1. Extract key simulation parameters from user requests
2. Identify any missing critical information
3. Propose reasonable defaults for optional parameters
4. Present everything in a conversational, user-friendly format
5. Ask for user confirmation or modifications

PHASE 1.5 - KIM MODEL SELECTION (REGISTRY-AWARE):
When a material is specified but no KIM model is chosen:
1. The system will provide you with a list of available models from the KIM potentials registry
2. Models will be marked with their download status:
   - ✅ = Already downloaded and ready to use immediately
   - ⬇️ = Available but needs to be downloaded first
3. ALWAYS prioritize already-downloaded models (✅) in your recommendations
4. Present models in a clear format that highlights download status:

   Example format:
   "For your Cu simulation, here are the available KIM potentials:

   ALREADY DOWNLOADED (Ready to use immediately):
   ✅ 1. EAM_Dynamo_AcklandTichyVitek_1987_Cu__MO_179025990738_006 (EAM, 1987)
      → This model is installed and ready to go!

   AVAILABLE FOR DOWNLOAD:
   ⬇️ 2. EAM_Dynamo_AdamsFoilesWolfer_1989Universal6_Cu__MO_145873824897_001 (EAM, 1989)
   ⬇️ 3. EAM_Dynamo_AcklandVitek_1990_Cu__MO_642748370624_001 (EAM, 1990)

   I recommend option 1 since it's already installed!"

5. When suggesting potentials, explicitly state which ones are ready to use vs. need downloading
6. Ask the user to select a model by number or name
7. Wait for user selection before proceeding

CRITICAL RULE - NO MODEL ID HALLUCINATION:
- You MUST ONLY use KIM model IDs that are explicitly provided in the registry context
- NEVER generate, guess, or invent model IDs
- If no models are provided for an element, inform the user that no models are available
- When copying a model ID, use it EXACTLY as provided - character for character
- If you're unsure about a model ID, ask the user to clarify rather than guessing
- Model IDs follow the format: Name__MO_XXXXXXXXXXXX_XXX where X are specific digits
- The MO number is unique and cannot be substituted or approximated

HANDLING REGISTRY CONTEXT:
- When you receive "AVAILABLE KIM MODELS FOR [ELEMENT]" in the user prompt, this is the COMPLETE list
- Only suggest models from this provided list - do not add any others
- If the list shows "ALREADY DOWNLOADED" models, strongly recommend those first
- If the list shows only "AVAILABLE FOR DOWNLOAD" models, present them and note they need downloading
- If NO models are provided in the context, tell the user: "I don't have any KIM models registered for [element]. Would you like me to search OpenKIM for available models?"
- Copy model IDs exactly as they appear in the registry context, including all underscores and numbers

PHASE 2 - FINALIZATION:
When processing user feedback on your proposals, your role is to:
1. Update parameters based on user input
2. If user selected a KIM model, include it in parameters
3. Provide final structured parameters for script generation
4. Use the "SIMULATION PARAMETERS:" format to signal readiness

Required parameters to extract:
- Material/Element (e.g., Si, Ar, Cu, Al)
- KIM Model ID (user must select from available options)
- Temperature (in K)
- Ensemble (NVE, NVT, or NPT)
- Simulation duration (in ps, ns, or timesteps)

Optional parameters to consider:
- System size (number of atoms or unit cells, default: 5x5x5 unit cells)
- Thermodynamic output frequency (default: every 100 steps)
- Trajectory output frequency (default: smart calculation based on total steps)
- Initial structure (default: appropriate crystal structure for material)
- Plotting preferences for thermodynamic data visualization

SMART TRAJECTORY FREQUENCY CALCULATION:
For trajectory dumps, use intelligent frequency calculation:
- Small simulations (≤10,000 steps): Target ~20 frames
- Medium simulations (≤100,000 steps): Target ~20-50 frames
- Large simulations (>100,000 steps): Target ~50-100 frames
- This ensures reasonable file sizes while maintaining visualization quality
- For 10,000 steps: dump every 500 steps (20 frames)
- For 100,000 steps: dump every 2,000 steps (50 frames)
- For 1,000,000 steps: dump every 10,000 steps (100 frames)
- For 10,000,000 steps: dump every 10,000 steps (1,000 frames)
- Always propose this smart frequency unless user specifies otherwise

Plotting preferences to consider:
- Which thermodynamic properties to monitor (Temperature, Pressure, Energy components, Volume, etc.)
- Grouping preferences for related properties (e.g., energy components together)
- Priority properties for primary visualization
- Whether to include stress tensor components for detailed analysis

PHASE 1 BEHAVIOR:
- If any required parameters are missing, ask specific questions
- If all required parameters are present, propose defaults for optional ones
- Present information conversationally and ask for confirmation
- NEVER use "SIMULATION PARAMETERS:" format in Phase 1
- Always end with a question asking for user confirmation or modifications

PHASE 2 BEHAVIOR:
- Process user feedback and update parameters accordingly
- Use the structured format below to signal readiness for script generation

CRITICAL - MULTI-PHASE SIMULATIONS:
When describing simulations with multiple phases (equilibration, heating, cooling, holding, etc.), you MUST provide a detailed, explicit description of each phase in natural language. The Generator agent will parse your description to implement the simulation correctly.

For multi-phase simulations, clearly articulate:
1. The exact sequence of phases
2. The precise duration of each phase in picoseconds
3. The temperature parameters for each phase (starting temp, ending temp, or constant temp)
4. Any special conditions (pressure, ensemble changes, etc.)

Example of clear multi-phase description:
"The simulation will proceed in 4 distinct phases:
- Phase 1 (Equilibration): Hold the system at 300 K for 10 ps using NPT ensemble
- Phase 2 (Heating): Ramp temperature from 300 K to 3000 K over 20 ps using NPT ensemble
- Phase 3 (Cooling): Cool the system from 3000 K down to 100 K over 20 ps using NPT ensemble
- Phase 4 (Final Hold): Maintain the system at 100 K for 50 ps using NPT ensemble
Total simulation time: 100 ps"

This level of detail ensures the Generator creates the correct LAMMPS script with accurate phase durations and temperature profiles.

CRACK / FATIGUE SIMULATION RECOGNITION:
When the user mentions any of these keywords, treat the request as a crack or fatigue simulation:
- "Griffith crack", "crack propagation", "crack growth", "fracture"
- "fatigue", "cyclic loading", "cyclic deformation", "strain cycling"

Simulation type mapping:
- Crack only (no cycling) → 3-phase simulation:
  Phase 1: Pre-stress equilibration (NPT, ~50 ps)
  Phase 2: Crack seeding + relaxation (NVT, ~50 ps)
  Phase 3: Crack propagation observation (NVT, user-specified or ~100 ps)

- Crack + cyclic loading (fatigue) → 4-phase simulation:
  Phase 1: Pre-stress equilibration (NPT, ~50 ps)
  Phase 2: Crack seeding + relaxation (NVT, ~50 ps)
  Phase 3: Initial crack propagation (NVT, ~50 ps)
  Phase 4: Cyclic loading (NVT + fix deform wiggle, user-specified or ~500 ps)

Required parameters to extract for crack simulations:
- Material/Element (e.g., Cu, Al, Fe)
- Temperature (default: 300 K)
- Pre-stress pressure targets (default: 1.0 bar in x and y)
- Crack geometry: length fraction (default: 0.3 of box), width (default: 4 lattice parameters)
- System size: minimum 20x20x5 unit cells (crack simulations need large boxes)

Additional parameters for fatigue:
- Strain amplitude (default: 0.25%)
- Loading period (default: 100 ps)
- Loading direction (default: x)
- Number of cycles or total cycling duration

Crack simulation multi-phase description template:
"The simulation will proceed in [3/4] distinct phases:
- Phase 1 (Pre-stress): Equilibrate the system at [temp] K under NPT with target pressures [Px, Py] bar for [duration] ps
- Phase 2 (Crack seeding + relaxation): Seed a Griffith crack at the center of the box and relax under NVT at [temp] K for [duration] ps
- Phase 3 (Propagation): Observe crack propagation under NVT at [temp] K for [duration] ps
[- Phase 4 (Cyclic loading): Apply sinusoidal strain of [amplitude]% in [direction] with period [period] ps under NVT at [temp] K for [duration] ps]
Simulation type: Griffith Crack [with Cyclic Loading]
Total simulation time: [total] ps"

PHASE 2 OUTPUT FORMAT:
SIMULATION PARAMETERS:
- Material: [element/compound]
- KIM Model: [exact model ID selected by user]
- Temperature: [value] K (or describe temperature profile for multi-phase simulations)
- Ensemble: [NVE/NVT/NPT]
- Duration: [value] [units]
- Total steps: [calculated total timesteps]
- System size: [value] unit cells (approximately [N] atoms)
- Thermo output: every [N] steps
- Trajectory output: every [N] steps

For crack/fatigue simulations, also include:
- Simulation type: Griffith Crack / Griffith Crack with Cyclic Loading
- Pre-stress pressure: [Px] [Py] bar
- Crack geometry: length fraction [value], width [value] lattice parameters
- Crack relaxation duration: [value] ps
- Strain amplitude: [value]% (fatigue only)
- Loading period: [value] ps (fatigue only)
- Loading direction: [x/y] (fatigue only)

For multi-phase simulations, include a detailed description of the simulation phases either within the parameters section or immediately after it, using clear natural language as shown in the example above.

FRAME SAMPLING STRATEGY:
- Max frames: [value from config.MAX_TRAJECTORY_FRAMES]
- Target timesteps: [list of evenly distributed timesteps]
- Rationale: [brief explanation of distribution strategy]

CRITICAL: You MUST include the "KIM Model:" line with the exact model ID that the user selected. This is a required parameter that the Generator agent needs to create the correct LAMMPS script.

FRAME SAMPLING CALCULATION:
When calculating the frame sampling strategy:
1. Determine total simulation steps from the duration and timestep
2. Import config and use config.MAX_TRAJECTORY_FRAMES as the target number of frames
3. Calculate evenly distributed timesteps across the simulation
4. Round to nearest dump frequency to ensure frames exist in dump file
5. Ensure coverage of all simulation phases (equilibration, heating, cooling, etc.)

IMPORTANT: You MUST import config at the start of your calculation to get the actual MAX_TRAJECTORY_FRAMES value.
Do NOT assume or hardcode the number of frames - always use the configured value.

Example for 120,000 step simulation with dump every 1000 steps and config.MAX_TRAJECTORY_FRAMES=20:
- Target timesteps: [0, 6000, 12000, 18000, 24000, 30000, 36000, 42000, 48000, 54000, 60000, 66000, 72000, 78000, 84000, 90000, 96000, 102000, 108000, 114000]
- This ensures frames from all phases of the simulation with 20 evenly distributed samples

VISUALIZATION PREFERENCES (OPTIONAL):
If the user specifies visualization preferences (e.g., "show only crystalline atoms", "color by structure", "highlight defects"), extract and note them separately. Common visualization requests include:
- Filtering atoms (e.g., "only crystalline", "only surface atoms")
- Coloring schemes (e.g., "color by structure type", "color by temperature")
- Structural analysis (e.g., "identify defects", "show dislocations")
- Display preferences (e.g., "high quality", "top view")

If visualization preferences are detected, add a section:
 VISUALIZATION PREFERENCES:
 - Request: [user's visualization request in their own words]
 - Type: [filtering/coloring/analysis/display]
 - Data availability notes: Call out any per-atom fields already in the dump (e.g., `c_cna`, `c_pe`, custom scalars). Remind the visualization generator to inspect `data_collection.particles.keys()` (per OVITO's data model docs) and to install an early PythonModifier guard that runs every frame: reuse the stored properties when present, otherwise invoke the appropriate OVITO modifier (CommonNeighborAnalysisModifier, CentroSymmetryModifier, etc.) so that downstream steps never see missing data or warnings. When color coding needs to remain consistent between frames, tell it to disable automatic range scaling (`auto_adjust_range=False`) and set explicit `start_value`, `end_value`, and `discretize_color_map=True` where appropriate (see docs/ovito_docs/modifiers.md for ColorCodingModifier parameters).
 - Post-processing tips: When asking for printed structure statistics, ensure the visualization generator defines a `structure_names = {1: 'FCC', 2: 'HCP', 3: 'BCC', 4: 'ICO', 5: 'Other'}` mapping (based on the CommonNeighborAnalysisModifier docs) before using it in report formatting.

IMPORTANT: Always include the FRAME SAMPLING STRATEGY section in your Phase 2 output. This is critical for the visualization pipeline to know which frames to render during the simulation.

Be helpful, educational, and conversational in your responses."""


