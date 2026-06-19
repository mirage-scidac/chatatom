# Fresh Plan: Closed-Loop Benchmark Campaign for Griffith Crack

## Context

**Problem**: ChatMD's agent is currently human-in-the-loop — user types a request, chat agent proposes a script, user approves, script runs. For benchmarking the agent's autonomous decision-making in a Griffith crack propagation search space, we need to:

1. Let the user kick off a campaign by typing naturally in the chat
2. **Chat agent interactively asks** for campaign parameters (iterations count, knob ranges, target) — same conversational style it uses for fresh simulations
3. After the user confirms, run **20 autonomous iterations** with no further human approval
4. Each iteration, agent **forms a hypothesis** and proposes 4 knobs (strain_rate, temperature, crack_length, crack_width)
5. Auto-execute each LAMMPS script and **measure Δa** directly from dump files
6. Feed Δa results back into the next iteration's prompt as a synthesized "user message"
7. Persist trajectory in a manifest for later analysis

**Key Decisions** (locked from clarifications):
- **Trigger**: Single new intent `campaign_start` (no separate `campaign` intent)
- **Setup conversation**: Existing `chat_node` handles all interactive setup naturally (asks for iterations, ranges, target, then confirms)
- **Campaign addendum**: Injected into `chat_node` system prompt whenever:
  - The user message mentions "campaign", "benchmark", or similar keywords, OR
  - `session_id.startswith("benchmark_")` (during background iterations)
- **Iteration 1 prompt**: User's original message forwarded verbatim
- **Iteration 2+ prompt**: Synthesized "Past attempts: ... → Δa=...  Propose next config" message
- **Hypothesis required**: Each iteration's orchestrator response must include explicit reasoning ("Hypothesis: lowering strain rate reduces ...")
- **Execution model**: `campaign_executor_node` spawns `CampaignManager.run()` in a background thread and returns immediately
- **Δa measurement**: Direct from LAMMPS dump files via OVITO Python API (no re-render)
- **Failure handling**: Failed reviews → recorded as failure, campaign continues
- **Resumability**: Manifest persisted after every iteration (crash-safe)
- **Graph reuse**: Existing LangGraph used unchanged; campaign is a wrapper + 1 new node + prompt addendum

---

## Flow Diagrams

### Flow 1: Interactive Setup (chat in browser)

```
USER: "I want to run a campaign on Griffith crack Al, minimize Δa"
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│ intent_node → intent="new_simulation" (or "campaign" if      │
│               keyword detected)                              │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ chat_node (with campaign addendum injected)                  │
│                                                              │
│ PHASE 1 RESPONSE — presents fixed template + tunable params: │
│                                                              │
│   "I'll set up a Griffith crack benchmarking campaign for Al.│
│                                                              │
│    Here's what's FIXED:                                      │
│      • Material: Al (60×30×3 unit cells)                     │
│      • Template: 3-phase NPT 5ps → crack-seed NVT 5ps →      │
│        tensile 15ps                                          │
│      • Target: minimize Δa                                   │
│                                                              │
│    Here are the TUNING PARAMETERS and proposed ranges:       │
│      1. Number of iterations: 20                             │
│      2. strain_rate: [1e-3, 2e-2] /ps                        │
│      3. temperature: [100, 600] K                            │
│      4. crack_length_frac: [0.1, 0.6] (10–60% of L_x)        │
│      5. crack_width_a0: [2, 8] × a₀                          │
│                                                              │
│    Would you like to adjust any of these ranges, or shall    │
│    I proceed with the defaults? Reply 'yes' to confirm       │
│    and start."                                               │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
                              END (await user's response)

USER: "Use defaults for everything" (or modifies specific ranges)
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│ intent_node → intent="modify" / "question" / etc.            │
│                                                              │
│ NOTE: intent_node does NOT need to know "this is campaign    │
│ setup". Any intent that's not analyze/hpc/execute/           │
│ campaign_start falls through to chat_agent (default route).  │
│ So during setup conversation, intent can be anything —       │
│ chat_agent will keep the conversation going.                 │
│                                                              │
│ The ONLY intent that matters here is "campaign_start",       │
│ which is ONLY triggered when:                                │
│   (a) the assistant's last message contained a               │
│       "CAMPAIGN PARAMETERS:" block, AND                      │
│   (b) the user replied "yes" / "go ahead" / "confirm"        │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ chat_node (addendum still injected; reads conversation       │
│            history to understand campaign is being set up)   │
│                                                              │
│ PHASE 2 RESPONSE — structured confirmation block:            │
│                                                              │
│   "Great! Here's the final plan:                             │
│                                                              │
│    CAMPAIGN PARAMETERS:                                      │
│    n_iterations: 20                                          │
│    strain_rate_range: [1e-3, 2e-2]                           │
│    temperature_range: [100, 600]                             │
│    crack_length_frac_range: [0.1, 0.6]                       │
│    crack_width_a0_range: [2, 8]                              │
│    target: minimize delta_a                                  │
│                                                              │
│    Reply 'yes' to start the campaign."                       │
│                                                              │
│ chat_node post-processing parses CAMPAIGN PARAMETERS block   │
│ and persists to state (via SqliteSaver checkpoint):          │
│   campaign_n_iterations = 20                                 │
│   campaign_bounds = {strain_rate: [1e-3, 2e-2], ...}         │
│   campaign_target = "minimize delta_a"                       │
│   campaign_original_message = "I want to run a campaign..."  │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
                              END (await user's "yes")

USER: "yes"
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│ intent_node → intent="campaign_start"                        │
│   (classifier sees "yes" + conversation history shows        │
│    campaign was just confirmed)                              │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ campaign_executor_node (NEW)                                 │
│                                                              │
│ 1. Read from checkpoint state:                               │
│    - n_iterations, target, original_message                  │
│ 2. Generate campaign_id = "benchmark_<ts>"                   │
│ 3. Register SESSION_QUEUES[campaign_id]                      │
│ 4. Spawn CampaignManager.run() in daemon thread              │
│ 5. Emit campaign_started SSE event                           │
│ 6. Return state (graph invoke returns immediately)           │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
                              END
```

### Flow 2: Background Iteration Loop

```
┌──────────────────────────────────────────────────────────────┐
│ CampaignManager.run()  [background thread, 20 iterations]    │
│                                                              │
│ for i in 1..20:                                              │
│                                                              │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ STEP 1: Build synthesized "user message"               │ │
│   │                                                        │ │
│   │ • i==1:                                                │ │
│   │     msg = original_user_message                        │ │
│   │     (= "I want to run a campaign on Griffith crack Al")│ │
│   │                                                        │ │
│   │ • i>=2:                                                │ │
│   │     msg = "Past attempts:                              │ │
│   │       1) strain=7e-3, T=300, crack_len=30%,            │ │
│   │           crack_w=4a₀ → Δa=+12.4 Å                     │ │
│   │       2) strain=4e-3, T=250, crack_len=20%,            │ │
│   │           crack_w=4a₀ → Δa=+8.1 Å                      │ │
│   │      Goal: minimize Δa (target Δa ≤ 0).                │ │
│   │      Propose iteration N. State your hypothesis and    │ │
│   │      provide all 4 parameter values."                  │ │
│   └────────────────────────────┬───────────────────────────┘ │
│                                ▼                             │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ STEP 2: Invoke simulation_graph                        │ │
│   │                                                        │ │
│   │   simulation_graph.invoke({                            │ │
│   │     "user_message": msg,                               │ │
│   │     "session_id": campaign_id,                         │ │
│   │     "workflow_stage": "initial",                       │ │
│   │     "execution_target": "local"                        │ │
│   │   }, config={"configurable":                           │ │
│   │              {"thread_id": campaign_id}})              │ │
│   │                                                        │ │
│   │ Graph path:                                            │ │
│   │   intent_node → chat_node (with addendum, forms        │ │
│   │     hypothesis, proposes 4 knobs)                      │ │
│   │   → extract_params (parses knobs into state)           │ │
│   │   → generate_lammps (script + reviewer)                │ │
│   │   → detect_viz                                         │ │
│   │   → generate_ovito                                     │ │
│   │   → END                                                │ │
│   │                                                        │ │
│   │ Returns final_state with lammps_script + params        │ │
│   └────────────────────────────┬───────────────────────────┘ │
│                                ▼                             │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ STEP 3: Check review_passed                            │ │
│   │   - False → record failure, continue to next i         │ │
│   │   - True  → proceed to step 4                          │ │
│   └────────────────────────────┬───────────────────────────┘ │
│                                ▼                             │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ STEP 4: Run LAMMPS (auto, no human approval)           │ │
│   │   run_lammps_simulation(                               │ │
│   │     final_state["lammps_script"],                      │ │
│   │     data_callback=sse_q.put                            │ │
│   │   )                                                    │ │
│   │ Trust review — no extra bounds validation              │ │
│   └────────────────────────────┬───────────────────────────┘ │
│                                ▼                             │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ STEP 5: Compute Δa from dump files                     │ │
│   │   compute_delta_a(run_folder) → {                      │ │
│   │     "delta_a": float,                                  │ │
│   │     "a_initial": float, "a_final": float,              │ │
│   │     "n_defect_initial": int, "n_defect_final": int     │ │
│   │   }                                                    │ │
│   └────────────────────────────┬───────────────────────────┘ │
│                                ▼                             │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ STEP 6: Append iteration record + save manifest        │ │
│   │   self.past_runs.append({...})                         │ │
│   │   self._save_manifest()  ← every iteration             │ │
│   │   self._emit("campaign_iteration_complete", {...})     │ │
│   └────────────────────────────────────────────────────────┘ │
│                                                              │
│ After loop: emit campaign_complete                           │
└──────────────────────────────────────────────────────────────┘
```

---

## Architecture & Components

### 1. **src/benchmark/ Package** (new)

#### `src/benchmark/__init__.py`
Empty init file.

#### `src/benchmark/campaign_prompts.py`

- **`get_campaign_system_prompt_addendum() -> str`**
  - Multi-paragraph string to append to `_build_orchestrator_prompt()` output
  - Mirrors the existing Phase 1 → Phase 2 confirmation pattern that chat_node already uses for fresh simulations (see [src/prompts.py:733-747](../src/prompts.py#L733-L747))
  - Contents:
    - **Mode awareness**: "You are in CAMPAIGN MODE. Help the user set up an autonomous benchmarking campaign."

    - **PHASE 1 BEHAVIOR (Setup conversation, mirrors fresh-simulation pattern):**
      - When the user requests a campaign, present the tuning parameters and their default ranges conversationally — same way you currently propose simulation parameters for a fresh run.
      - Required parameters to elicit:
        | Parameter | Description | Default Range |
        |---|---|---|
        | `n_iterations` | How many simulations to run | 20 |
        | `strain_rate` | Tensile loading rate | [1e-3, 2e-2] /ps |
        | `temperature` | NVT thermostat temperature | [100, 600] K |
        | `crack_length_frac` | Crack length as fraction of L_x | [0.1, 0.6] |
        | `crack_width_a0` | Crack width in units of a₀ | [2, 8] × a₀ |
      - **Locked template** (NOT searchable, just announce it): Griffith Al (60×30×3 unit cells), 3-phase NPT 5ps → crack-seed NVT 5ps → tensile 15ps
      - **Locked outputs**: CNA + pe/atom + stress/atom; OVITO colors FCC=green, HCP=yellow, Other=red, BCC=blue; camera +z
      - **Locked target**: minimize Δa (post-loading crack length − initial seeded length)
      - Example Phase 1 response:
        ```
        I'll set up a Griffith crack benchmarking campaign for Al.

        Here's what's fixed:
          • Material: Al (60×30×3 unit cells)
          • Template: 3-phase (NPT 5ps → crack-seed NVT 5ps → tensile 15ps)
          • Target: minimize Δa

        Here are the tuning parameters and proposed ranges:
          1. Number of iterations: 20
          2. strain_rate: [1e-3, 2e-2] /ps
          3. temperature: [100, 600] K
          4. crack_length_frac: [0.1, 0.6] (10–60% of L_x)
          5. crack_width_a0: [2, 8] × a₀

        Would you like to adjust any of these ranges, or shall I proceed with the defaults? Reply 'yes' to confirm and start.
        ```
      - If user requests modifications, update the proposal and ask for confirmation again ("Does this updated plan look right?")
      - **NEVER auto-start the campaign** without explicit user confirmation

    - **PHASE 2 BEHAVIOR (User has confirmed with "yes", "go ahead", etc.):**
      - Emit a structured confirmation block so `chat_node`'s post-processing can extract the campaign params:
        ```
        CAMPAIGN PARAMETERS:
        n_iterations: 20
        strain_rate_range: [1e-3, 2e-2]
        temperature_range: [100, 600]
        crack_length_frac_range: [0.1, 0.6]
        crack_width_a0_range: [2, 8]
        target: minimize delta_a
        ```
      - This block signals to `chat_node` that intent should be classified as `campaign_start` on next turn

    - **DURING ITERATIONS (session_id starts with "benchmark_"):**
      - **STRICT response format required** — LLM MUST output a fixed template with the 4 knob values filled in. This ensures each iteration's `extract_params` + `generate_lammps` see CONSISTENT input → produces UNIFORM LAMMPS scripts (only knob values differ):
        ```
        Hypothesis: <1-3 sentences explaining the REASONING for this config>
                    <Should reference past iteration results when available>
                    <Should explain expected effect on Δa>

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
            crack_length: <crack_length_frac> × L_x
            crack_width: <crack_width_a0> × a₀
            crack method: neigh_modify exclude (NOT delete_atoms)

        - Phase 3 (Monotonic tensile loading):
            ensemble: NVT, temperature: <T> K, duration: 15 ps
            strain_rate: <strain_rate> /ps along y-axis
            method: fix deform y erate <strain_rate> units box remap v
            NO wiggle, monotonic loading only

        - Outputs per phase:
            dump every 500 steps
            properties: c_cna, c_pe, c_stress[1], c_stress[2], c_stress[3]
            cna cutoff: 0.854 × 4.05 = 3.457 Å (FCC Al midpoint)
            dump filenames: dump.lammps_npt_*, dump.lammps_seed_*, dump.lammps_pull_*

        - Visualization (OVITO):
            CNA coloring: FCC=green, HCP=yellow, Other=red, BCC=blue
            camera: down +z axis
        ```
      - **Hypothesis examples:**
        - Iter 1 (no past data): "Hypothesis: Starting near mid-range to establish baseline behavior. Strain rate 7e-3/ps and T=300K are typical Griffith crack conditions."
        - Iter 3: "Hypothesis: Iter 1→2 showed lower strain (7e-3→4e-3) reduced Δa from 12.4 to 8.1 Å. Continuing this trend with strain=2e-3 should further reduce dislocation nucleation."
      - **Why this strict template?**
        - **Uniformity**: Every iteration's `simulation_details` has identical structure — only the 4 knob values differ. Generated LAMMPS scripts are nearly identical across iterations (apples-to-apples comparison).
        - **No drift**: KIM model, system size, dump frequency, OVITO settings are locked — LLM cannot accidentally change them.
        - **Pipeline reuse**: extract_params and generate_lammps receive the same kind of structured natural-language description they always have — no new parsing logic needed.
        - **Reviewer happy**: The DSPy reviewer already knows the Griffith template via `KnowledgeBase.GRIFFITH_CRACK_TECHNIQUE`; a consistent input means consistent reviews.
      - **Why mandatory hypothesis?**
        - Forces LLM to reason explicitly (better proposals)
        - Captured in campaign_log.md for post-campaign analysis
        - LLM's future iterations see past hypotheses via conversation_history → builds on its own reasoning
      - The chat_node response then flows naturally through extract_params → generate_lammps → ... as today, but with GUARANTEED consistent input
  - Cites `KnowledgeBase.GRIFFITH_CRACK_TECHNIQUE` and `CRACK_OUTPUT_REQUIREMENTS`

- **`format_history_message(past_runs: list[dict], iteration: int, target: str) -> str`**
  - Used for iteration ≥ 2 in the background loop
  - Example output:
    ```
    Past attempts:
      1) strain=7e-3/ps, T=300K, crack_len=30%, crack_w=4a0 → Δa = +12.4 Å
      2) strain=4e-3/ps, T=250K, crack_len=20%, crack_w=4a0 → Δa = +8.1 Å

    Goal: minimize Δa (target Δa ≤ 0).
    Propose iteration 3. State your hypothesis and provide all 4 parameter values.
    ```

- **`is_campaign_context(session_id: str, user_message: str = "", conversation_history: list = None) -> bool`**
  - Returns True if ANY of:
    - **(a)** `session_id.startswith("benchmark_")` (we're inside the background iteration loop), OR
    - **(b)** User message contains a campaign-trigger PHRASE (strict matching to avoid fresh-sim false positives):
      - "run a campaign"
      - "run a benchmark campaign"
      - "closed-loop campaign"
      - "benchmark campaign"
      - "start a campaign"
      - "autonomous campaign"
      - Single keywords like "benchmark" or "campaign" alone do NOT trigger — must be one of the multi-word phrases above
    - **(c)** Recent conversation_history (last 6 messages) contains a prior assistant `CAMPAIGN PARAMETERS:` block OR a prior user message matching rule (b) — this keeps the addendum injected throughout the multi-turn setup conversation

#### `src/benchmark/crack_metrics.py`

- **`compute_delta_a(run_folder: str, dump_pattern: str = "dump.lammps*", a0: float = 4.05, ly_slab_fraction: float = 0.1) -> dict`**
  - Uses OVITO Python API directly (NOT sandboxed `ovito_executor`)
  - Steps:
    1. Glob dump files in `run_folder`, sort by timestep
    2. Import via `ovito.io.import_file(dump_files[0])` (or whichever pattern matches multi-frame)
    3. Apply `CommonNeighborAnalysisModifier(cutoff=0.854 * a0)` (formula from `KnowledgeBase.CNA_CUTOFF_FORMULA`)
    4. For frame 0 and frame -1:
       - Filter `non_fcc_mask = (Structure_Type != 1)`
       - Restrict to Y-slab around L_y/2 with thickness `ly_slab_fraction * L_y`
       - Compute X-extent: `max(positions[mask, 0]) - min(positions[mask, 0])`
  - Returns:
    ```python
    {
      "a_initial": float,       # x-extent @ frame 0
      "a_final": float,         # x-extent @ frame -1
      "delta_a": float,         # a_final - a_initial
      "n_defect_initial": int,
      "n_defect_final": int,
    }
    ```
  - Edge cases:
    - Zero atoms in slab → `delta_a=None, n_defect_*=0`
    - Missing dump files → `raise FileNotFoundError`
    - OVITO errors → `{"delta_a": None, "error": str(e)}`

#### `src/benchmark/campaign.py`

- **Class: `CampaignManager`**
  ```python
  class CampaignManager:
      def __init__(
          self,
          campaign_id: str,
          n_iterations: int,
          original_user_message: str,
          sse_q: queue.Queue,
          num_cores: int = 1,
          target: str = "minimize delta_a",
      ):
          self.campaign_id = campaign_id
          self.n_iterations = n_iterations
          self.original_user_message = original_user_message
          self.sse_q = sse_q
          self.num_cores = num_cores
          self.target = target
          self.past_runs = []
          self.best_delta_a = float('inf')
          self.output_folder = f"runs/benchmark_{campaign_id}/"
          os.makedirs(self.output_folder, exist_ok=True)

      def run(self) -> dict:
          # Sync method called in a worker thread
          # Returns manifest dict
          ...

      def _save_manifest(self):
          # Writes manifest.json after every iteration (crash-safe)
          ...

      def _emit(self, event_type: str, data: dict):
          # Helper: self.sse_q.put({"type": event_type, **data})
          ...
  ```

  - **`run()` method** (simpler storage: per-iteration .md files + minimal summary.json):
    ```python
    for i in range(1, self.n_iterations + 1):
        self._emit("campaign_iteration_start", {"iteration": i})

        # Step 1: synthesize message
        if i == 1:
            msg = self.original_user_message
        else:
            msg = format_history_message(self.past_runs, i, self.target)

        # Step 2: invoke graph (SqliteSaver loads past conversation_history
        # from prior iterations via thread_id, so LLM sees all past hypotheses)
        state = {
            "user_message": msg,
            "session_id": self.campaign_id,
            "workflow_stage": "initial",
            "execution_target": "local",
        }
        final_state = simulation_graph.invoke(
            state,
            config={"configurable": {"thread_id": self.campaign_id}}
        )

        chat_response = final_state.get("analysis", "")

        # Step 3: check review
        if not final_state.get("review_passed"):
            self._write_iteration_md(
                iteration=i,
                chat_response=chat_response,
                lammps_script=None,
                run_folder=None,
                delta_a=None,
                review_failed=True,
                review_issues=final_state.get("review_issues"),
            )
            self._update_summary(i, delta_a=None, review_passed=False)
            self._emit("campaign_iteration_failed", {"iteration": i})
            continue

        # Step 4: run LAMMPS
        start = time.time()
        output_files = run_lammps_simulation(
            script=final_state["lammps_script"],
            data_callback=lambda evt: self.sse_q.put(evt),
            num_cores=self.num_cores,
            ...
        )
        wall_clock = time.time() - start

        # Step 5: measure Δa
        run_folder = os.path.dirname(output_files["log"])
        delta_a_dict = compute_delta_a(run_folder)
        delta_a = delta_a_dict.get("delta_a")

        # Step 6: APPEND this iteration to the single campaign_log.md
        # (one shared file accumulates all 20 iterations)
        self._append_to_campaign_log(
            iteration=i,
            chat_response=chat_response,         # ← full LLM text with hypothesis + 4 params
            lammps_script=final_state["lammps_script"],
            run_folder=run_folder,
            delta_a=delta_a,
            wall_clock_s=wall_clock,
            delta_a_details=delta_a_dict,
        )

        # Step 7: update minimal summary.json (just for plotting)
        if delta_a is not None:
            self.best_delta_a = min(self.best_delta_a, delta_a)
        self._update_summary(
            iteration=i,
            delta_a=delta_a,
            best_so_far=self.best_delta_a,
            run_folder=run_folder,
        )

        # Step 8: append to in-memory past_runs (used for format_history_message
        # in the next iteration). Use a regex to pull 4 knobs ONLY for the
        # synthesized message text — not stored anywhere structured.
        self.past_runs.append({
            "iteration": i,
            "params_text": _extract_params_text(chat_response),  # raw "strain=X, T=Y..."
            "delta_a": delta_a,
        })

        self._emit("campaign_iteration_complete", {
            "iteration": i,
            "delta_a": delta_a,
            "best_so_far": self.best_delta_a,
        })

    self._emit("campaign_complete", {"summary_path": ...})
    ```

  - **Helper: `_append_to_campaign_log(...)`**
    - Appends to ONE shared file: `{output_folder}/campaign_log.md`
    - On iteration 1: writes file header + iteration 1 block
    - On iterations 2..N: appends an iteration block (no rewriting of past entries)
    - All 20 iterations end up in this single file
    - File template:
      ```markdown
      # Campaign Log: benchmark_2026_05_27_120000

      **Target**: minimize Δa
      **Template**: Griffith Al (60×30×3), 3-phase NPT 5ps → crack-seed NVT 5ps → tensile 15ps
      **Tuning knobs**: strain_rate, temperature, crack_length_frac, crack_width_a0

      ---

      ## Iteration 1
      ### Hypothesis & Proposed Parameters
      (raw chat_response text — contains "Hypothesis:" and 4 knobs)
      ### Generated LAMMPS Script
      ```lammps
      (full script)
      ```
      ### Run Folder
      `runs/run_2026_05_27_120042/`
      ### Result
      - Δa = +12.4 Å
      - a_initial = 8.5 Å, a_final = 20.9 Å
      - wall_clock = 87.3 s

      ---

      ## Iteration 2
      ### Hypothesis & Proposed Parameters
      (next iteration's chat_response)
      ...

      ---

      ## Iteration 3
      ...

      (continues for all 20 iterations)
      ```
    - Crash-safe: file is appended to, not rewritten, so partial campaigns retain all completed iterations

  - **Helper: `_update_summary(...)`**
    - Updates `{output_folder}/summary.json` (rewritten each iteration):
      ```json
      {
        "campaign_id": "benchmark_2026_05_27_120000",
        "n_iterations": 20,
        "target": "minimize delta_a",
        "log_file": "campaign_log.md",
        "iterations": [
          {"iteration": 1, "delta_a": 12.4, "best_so_far": 12.4,
           "run_folder": "runs/run_..."},
          {"iteration": 2, "delta_a": 8.1, "best_so_far": 8.1,
           "run_folder": "runs/run_..."}
        ],
        "best_so_far": 8.1
      }
      ```
    - Written after every iteration (crash-safe)
    - Used by frontend for plotting Δa-per-iteration chart

  - **Helper: `_extract_params_text(chat_response: str) -> str`**
    - Pulls the "Proposed configuration" block from the LLM response (simple regex)
    - Returns ONLY the human-readable params text (e.g., "strain=7e-3, T=300, crack_len=30%, crack_w=4a₀")
    - Used to build the synthesized message for next iteration's `format_history_message()`
    - This is NOT structured parsing — just substring extraction for the next prompt

  - **Conversation history policy for campaign sessions**:
    - Override `MAX_CONVERSATION_TURNS` for `benchmark_*` sessions — keep FULL conversation history
    - Rationale: LLM benefits from seeing all past hypotheses to avoid repeating failed strategies
    - Risk mitigation: monitor token usage; if approaching limit (~iter 25+), fall back to existing trim
    - Implementation: in `chat_node`, before trimming, check `is_campaign_session(session_id)` → if True, skip trim

---

### 2. **Graph Changes** (minimal, isolated)

#### `src/graph/state.py` — Add fields
- Add to `SimulationState` TypedDict:
  - `campaign_n_iterations: Optional[int]`
  - `campaign_target: Optional[str]`
  - `campaign_original_message: Optional[str]`
  - `campaign_bounds: Optional[Dict]` — user-customized bounds (optional)

#### `src/agents.py` — Extend `ClassifyUserIntent` DSPy signature
- Add ONE new intent type: `"campaign_start"` (signals confirmation, ready to launch loop)
- Update intent docstring with explicit trigger rule:
  - `campaign_start`: **ALL THREE conditions must be true:**
    1. The MOST RECENT assistant message contained a `CAMPAIGN PARAMETERS:` block (use exact string match on "CAMPAIGN PARAMETERS:" prefix)
    2. The current user message is a clear confirmation: "yes", "go ahead", "start", "confirm", "run it", "proceed", "looks good"
    3. The user message does NOT request any modification (e.g., "yes but change temperature to 400")
  - If user says "yes" but ALSO requests changes, classify as `modify` (chat_node updates the proposal and re-confirms)
  - If conditions are not all met → fall through to existing intents (chat_agent will continue the setup conversation)
- This mirrors the existing rule for `execute` intent (see [src/prompts.py:740-746](../src/prompts.py#L740-L746)) where Phase 2 only fires on explicit confirmation with no concurrent modifications

#### `src/graph/nodes.py` — ONE new node + minor `chat_node` tweak

**`campaign_executor_node(state) -> state`** (only new node)
```python
def campaign_executor_node(state):
    # Read from checkpoint state (set during prior conversation turns)
    n_iter = state.get("campaign_n_iterations", 20)
    target = state.get("campaign_target", "minimize delta_a")
    original_msg = state.get("campaign_original_message")

    # Fallback: walk conversation_history to find the first user msg
    # that triggered the campaign discussion
    if not original_msg:
        for msg in state.get("conversation_history", []):
            if msg["role"] == "user" and any(kw in msg["content"].lower()
                                              for kw in ["campaign", "benchmark"]):
                original_msg = msg["content"]
                break

    campaign_id = f"benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    SESSION_QUEUES[campaign_id] = queue.Queue()

    manager = CampaignManager(
        campaign_id=campaign_id,
        n_iterations=n_iter,
        original_user_message=original_msg,
        sse_q=SESSION_QUEUES[campaign_id],
        num_cores=config.LAMMPS_CORES,
        target=target,
    )
    threading.Thread(target=manager.run, daemon=True).start()

    _q(state, {"type": "campaign_started",
                "campaign_id": campaign_id,
                "n_iterations": n_iter})
    return state
```

**`chat_node` enhancement** (minor)
- When campaign addendum is active, after streaming the orchestrator's response, parse it (regex or DSPy mini-extraction) to detect campaign params:
  - Look for "iterations: N" or "Confirming: N iterations"
  - Set `state["campaign_n_iterations"]`, `state["campaign_target"]`, `state["campaign_original_message"]` if found
- These persist via SqliteSaver, so `campaign_executor_node` can read them next turn

#### `src/graph/routing.py` — Update `route_after_intent`
- Add ONE branch:
  - `intent == "campaign_start"` → `campaign_executor`
- Everything else routes as today (no change to existing branches)

#### `src/graph/graph.py` — Register one new node
- Add `campaign_executor` node, edge to END

#### `src/prompts.py` — Inject campaign addendum
- Modify `_build_orchestrator_prompt(session_id: Optional[str] = None, user_message: str = "")`:
  - If `is_campaign_context(session_id, user_message)`: append `get_campaign_system_prompt_addendum()`
- Update call site in `chat_node` to pass both `session_id` and `user_message`

---

### 3. **`src/app.py`** — No Changes
- Campaign uses existing `POST /api/llm-stream-live` endpoint
- SESSION_QUEUES system handles both setup chat AND background iteration events
- During background loop, events are emitted on `SESSION_QUEUES[campaign_id]`
- Frontend opens a fresh SSE connection for `campaign_id` after receiving `campaign_started` event (OR reuses chat session_id — see open question below)

#### Open Detail: How does frontend receive campaign SSE events?
- **Option A** (simpler): The background thread emits to `SESSION_QUEUES[campaign_id]`. Frontend opens a new SSE connection to `/api/llm-stream-live?session_id=<campaign_id>&listen_only=true` to drain those events.
- **Option B**: The background thread also emits to the original chat session's SSE queue (dual-emit). Frontend doesn't need a new connection.
- **Recommended**: Option B (dual-emit) — simpler for frontend, no new API surface.

---

### 4. **`index.html`** — New SSE event handlers (no new tab)

#### Reuse existing UI panels
- **Chat panel**: shows the orchestrator's reasoning/hypothesis per iteration (via existing `agent_content` events)
- **Viz panel**: shows current iteration's trajectory (via existing `trajectory_frame` events)
- **Thermo plot panel**: shows current iteration's live thermo data (via existing `thermo_data_point` events)

#### UX rules (locked from clarifications)
- **Show only current iteration in viz panel**: clear trajectory frames at the start of each `campaign_iteration_start` event
- **Show live thermo plots per iteration**: clear thermo plot at the start of each `campaign_iteration_start` event; reuse existing handlers
- Each iteration looks like a normal simulation: viz + thermo stream live, then clear for the next iteration

#### New SSE event handlers
- `campaign_started`: Show banner "Campaign running (0/N iterations complete)"
- `campaign_iteration_start`:
  - Update banner counter ("Iteration N/20 running")
  - **CLEAR** `trajectoryFrames` array and viz panel display
  - **CLEAR** thermo plot data
  - Scroll chat to new section
- `campaign_iteration_complete`:
  - Update banner with Δa for this iteration
  - Append row to campaign progress table (iteration, params, Δa, best-so-far)
  - Update mini Δa-per-iteration chart with best-so-far line
- `campaign_iteration_failed`: Show "❌ Iteration N failed review", append to progress table, continue
- `campaign_complete`: Show "✅ Campaign complete. Manifest: runs/benchmark_<id>/manifest.json"

#### Optional UI elements
- Collapsible campaign progress table (sticky at top of chat panel)
- Mini chart: bars = per-iteration Δa, line overlay = best-so-far (monotonic non-increase)

---

## Files to Create

1. `src/benchmark/__init__.py`
2. `src/benchmark/campaign_prompts.py`
3. `src/benchmark/crack_metrics.py`
4. `src/benchmark/campaign.py`
5. `tests/test_crack_metrics.py`

## Files to Edit

1. `src/graph/state.py` — 4 new optional fields
2. `src/agents.py` — extend `ClassifyUserIntent` with `campaign_start` intent
3. `src/graph/nodes.py` — ONE new node (`campaign_executor`); minor `chat_node` tweak to persist campaign setup params
4. `src/graph/routing.py` — one new intent branch
5. `src/graph/graph.py` — register `campaign_executor` node
6. `src/prompts.py` — inject campaign addendum when in campaign context
7. `src/run_manager.py` — extend `create_run_folder()` to accept optional `parent_dir` param (defaults to `RUNS_BASE_DIR` for backward compatibility). Iterations in a campaign pass `parent_dir=runs/benchmark_<id>/` so all 20 iteration run folders nest inside the campaign folder
8. `src/tools.py` — extend `run_lammps_simulation()` to accept optional `parent_dir`, threaded down to `create_run_folder()`
9. `index.html` — new SSE event handlers (no new tab)

## Critical Reuse (no reinvention)

- `simulation_graph` from [src/graph/graph.py](../src/graph/graph.py) — invoked unchanged per iteration
- `SESSION_QUEUES` from [src/graph/nodes.py:16](../src/graph/nodes.py#L16) — reused for SSE plumbing
- `chat_node` from [src/graph/nodes.py:82](../src/graph/nodes.py#L82) — handles all interactive setup conversation
- `run_lammps_simulation()` from [src/tools.py:630](../src/tools.py#L630) — auto-runs after each iteration's graph invoke
- `_build_orchestrator_prompt()` from [src/prompts.py](../src/prompts.py) — appended with campaign addendum in campaign context
- `KnowledgeBase` constants (`GRIFFITH_CRACK_TECHNIQUE`, `CNA_CUTOFF_FORMULA`, etc.)
- Existing DSPy agents & review pipeline — fully reused

---

## Folder Structure (Nested — Option B)

All output for a single campaign lives inside `runs/benchmark_<id>/`. The 20 iteration run folders are nested inside, NOT scattered at the top level. Fresh simulations (non-campaign) continue to live at `runs/run_<timestamp>/` as today — unchanged.

```
runs/
  │
  ├── benchmark_2026_05_27_120000/          ← Single campaign, self-contained
  │     ├── campaign_log.md                  ← All 20 iterations appended
  │     ├── summary.json                     ← Δa per iteration (for plotting)
  │     ├── run_2026_05_27_120042/           ← Iteration 1 LAMMPS output
  │     │     ├── input.lammps
  │     │     ├── log.lammps
  │     │     ├── dump.lammps_npt_*
  │     │     ├── dump.lammps_seed_*
  │     │     ├── dump.lammps_pull_*
  │     │     └── simulation_info.json
  │     ├── run_2026_05_27_120347/           ← Iteration 2
  │     ├── run_2026_05_27_120602/           ← Iteration 3
  │     ├── ...
  │     └── run_2026_05_27_154200/           ← Iteration 20
  │
  ├── run_2026_05_27_180012/                 ← Fresh simulation (separate, normal)
  └── run_2026_05_27_181500/                 ← Another fresh simulation
```

### Benefits

- **Self-contained**: one folder = one complete campaign. Easy to share/archive (`tar -czf campaign.tar.gz runs/benchmark_xyz/`) or delete (`rm -rf runs/benchmark_xyz/`)
- **No clutter**: fresh simulations and campaign iterations don't mix at the top level
- **Iteration paths are predictable**: campaign_log.md references `runs/benchmark_<id>/run_<ts>/` paths
- **Existing fresh-sim flow unchanged**: `create_run_folder()` still defaults to `RUNS_BASE_DIR`

### Code change required

`src/run_manager.py:14`:
```python
def create_run_folder(parent_dir: Optional[str] = None) -> str:
    base = parent_dir or RUNS_BASE_DIR
    os.makedirs(base, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    run_folder_path = os.path.join(base, f"run_{timestamp}")
    os.makedirs(run_folder_path, exist_ok=True)
    return os.path.abspath(run_folder_path)
```

`src/tools.py:630` (`run_lammps_simulation`):
- Accept optional `parent_dir` kwarg, thread it down to `create_run_folder(parent_dir=parent_dir)`

CampaignManager:
```python
output_files = run_lammps_simulation(
    script=final_state["lammps_script"],
    parent_dir=self.output_folder,  # runs/benchmark_<id>/
    data_callback=...,
    num_cores=self.num_cores
)
```

Fresh simulations call `run_lammps_simulation(...)` without `parent_dir` → folder created at `runs/run_<ts>/` as before.

---

## Verification Plan

### End-to-End Manual Test

1. **Start app**: `pixi run run-app`
2. **Open chat** at `http://localhost:2002`
3. **Type**: "I want to run a benchmark campaign on Griffith crack Al, minimize Δa" (smoke test)
4. **Observe**:
   - Chat agent responds interactively asking for iteration count + knob ranges
5. **Reply**: "2 iterations, defaults"
6. **Observe**:
   - Chat agent confirms: "2 iterations, defaults, target minimize Δa. Reply 'yes' to start."
7. **Reply**: "yes"
8. **Observe SSE stream**:
   - `intent: campaign_start`
   - `campaign_started: {campaign_id, n_iterations: 2}`
   - `campaign_iteration_start: {iteration: 1}`
   - `agent_start` → `agent_content` (orchestrator forms hypothesis + proposes config) → `agent_complete`
   - `status: generating_lammps` → `script_generated` → `review_complete: passed`
   - `start_streaming_simulation` (WebSocket triggered)
   - `trajectory_frame` (live frames) → `trajectory_complete`
   - `campaign_iteration_complete: {iteration: 1, params, delta_a, best_so_far}`
   - `campaign_iteration_start: {iteration: 2}`
   - (repeat with synthesized "Past attempts: ... Propose next" message)
   - `campaign_complete: {manifest_path}`
9. **Verify manifest** at `runs/benchmark_<ts>/manifest.json`:
   - Both iterations recorded with `params`, `delta_a`, `best_so_far`, `run_folder`
   - Manifest existed after iteration 1 (timestamp check) — proves resumability
10. **Full campaign** with N=20:
    - Confirm best_so_far monotonically non-increases
    - All 20 iterations have valid params within bounds

### Unit Test: `tests/test_crack_metrics.py`

- Create fixture: two tiny LAMMPS dump files (~5–10 atoms each) at `tests/test_assets/crack_pair/`
  - Frame 0: atoms at X ∈ [10, 15] Å (a_initial ≈ 5)
  - Frame -1: atoms at X ∈ [10, 18] Å (a_final ≈ 8)
  - Expected Δa ≈ +3 Å
- Test `compute_delta_a()` returns correct Δa, defect counts
- Test edge cases: no atoms in slab → `delta_a is None`; missing files → `FileNotFoundError`

### Sanity Checks

- **Force a failure**: Edit campaign addendum to suggest impossible param bounds; verify iteration records `review_passed=False` and campaign continues
- **Kill server mid-campaign**: Verify summary.json + campaign_log.md are up-to-date through last completed iteration
- **Hypothesis presence**: Verify each iteration's chat output starts with "Hypothesis:" or includes reasoning before the simulation specification
- **History continuity**: Verify orchestrator references prior iterations' Δa in iteration 3+'s proposal
- **Template uniformity (CRITICAL for benchmark validity)**:
  - Run a 3-iteration smoke test
  - Diff the generated LAMMPS scripts pairwise (`diff iter1.in iter2.in`)
  - Expected: ONLY differences are the 4 knob values (strain rate value, temperature value, crack region coordinates, crack width coordinates)
  - All other lines (KIM init, mass commands, dump definitions, fix names, etc.) should be IDENTICAL
  - If scripts diverge in structure: the addendum template is too loose → tighten it

---

## Implementation Order

1. **Phase 1**: Create `src/benchmark/` modules (prompts, metrics, campaign manager)
2. **Phase 2**: Extend `ClassifyUserIntent` with `campaign_start` intent
3. **Phase 3**: Add `campaign_executor_node` + minor `chat_node` tweak to persist campaign setup params
4. **Phase 4**: Update `_build_orchestrator_prompt()` to inject campaign addendum in campaign context
5. **Phase 5**: Add SSE event handlers in `index.html`
6. **Phase 6**: Write unit tests for `compute_delta_a`
7. **Phase 7**: End-to-end manual testing

---

## Known Behaviors to Observe During Testing

- **Previous LAMMPS script injection in chat_node** ([src/graph/nodes.py:101-112](../src/graph/nodes.py#L101-L112)):
  - chat_node currently appends `[PREVIOUS SESSION CONTEXT: ... LAMMPS script used: ... ]` to the user message whenever `prev_material` and `prev_script` are in the checkpoint state
  - For campaigns, this means the LLM sees the previous iteration's full LAMMPS script in every prompt
  - **Potential risks**: (a) LLM anchors on previous script instead of strict template adherence, (b) prompt grows large by iter 20, (c) script structure may drift across iterations
  - **Decision**: Leave as-is initially. Observe iter 1→2→3 scripts. If we see drift toward copy-and-modify behavior or template violations, add `if not is_campaign_session(session_id):` guard around the injection block

---

## Out of Scope (Deferred)

- Baseline algorithms (random, Latin hypercube, Bayesian optimization)
- Multi-seed replicas or statistical Δa estimation
- HPC execution (local LAMMPS only)
- Campaign comparison UI (single campaign only)
- Campaign cancellation mid-run (kill switch can be added later)
