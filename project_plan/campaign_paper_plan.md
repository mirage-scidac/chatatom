# Campaign Mode — Paper-Ready Refactor Plan

> **Goal:** make the campaign mode in ChatMD strong enough to support a paper case study with a credible "agent finds robust safe regions" claim.
>
> **Two phases:**
> - **Phase 2e** — Foundation: hypothesis memory + locked LAMMPS script template. Must land first.
> - **Phase 3** — Paper-prep: multi-seed runs, P(Δa ≤ 0) objective, statistical reporting, random baseline.

---

## 1. Context

Today's campaign mode has three real problems that block a paper claim:

1. **Hypothesis amnesia.** Each iteration sees only past knob values + Δa numbers, NOT the LLM's prior reasoning. So iteration N writes "without explicit values from iteration 1, I'll assume defaults" — it didn't even remember what it just tried.

2. **LAMMPS script drift.** Every iteration regenerates a fresh LAMMPS script from prose. Non-knob fields (dump frequency rounding, variable naming, units handling) can vary across iterations. Δa differences are no longer cleanly attributable to knob changes.

3. **Greedy single-point objective.** The skill says "minimize Δa." Agent chases the single most negative number it can find. MD has noise, so it overfits to lucky runs. No notion of robustness, no statistical thinking.

The new architecture (orchestrator + tools) is in place but these campaign-mode bugs predate the refactor. Fixing them is what makes the system paper-ready.

---

## 2. Phase 2e — Foundation

### 2e.1 Hypothesis memory

**What changes**
- `CampaignManager` already captures `chat_response` per iteration (the LLM's reply containing its hypothesis prose). It's stored in `past_runs[i]` but never read back.
- Enrich `format_history_message` in [src/benchmark/campaign_prompts.py](src/benchmark/campaign_prompts.py) so each "Past attempt:" entry includes the full hypothesis text, not just knobs → Δa.

**Before (today)**
```
Past attempts:
  1) strain=7e-3, T=300, crack_len=30%, crack_w=4a₀ → Δa = +12.4 Å
  2) strain=4e-3, T=250, crack_len=20%, crack_w=4a₀ → Δa = +8.1 Å
Goal: minimize Δa. Propose iteration 3. ...
```

**After**
```
Past attempts:

  Iter 1:
    Hypothesis: Starting near mid-range to establish baseline behavior
    at typical Griffith crack conditions.
    Knobs: strain_rate=7e-3, T=300, crack_length_frac=0.30, crack_width_a0=4
    Result: Δa = +12.4 Å

  Iter 2:
    Hypothesis: Iter 1 showed significant crack growth. Reducing strain
    rate to test whether slower loading suppresses dislocation nucleation
    at the crack tip.
    Knobs: strain_rate=4e-3, T=250, crack_length_frac=0.20, crack_width_a0=4
    Result: Δa = +8.1 Å (-4.3 Å improvement; hypothesis supported)

Propose iter 3. Reason about whether to continue the trend, pivot,
or test a different mechanism. State your hypothesis and explain why
the proposed knob values follow from prior results.
```

**Files touched**
- [src/benchmark/campaign_prompts.py](src/benchmark/campaign_prompts.py): rewrite `format_history_message`
- [src/benchmark/campaign.py](src/benchmark/campaign.py): make sure `chat_response` is captured + stored in `past_runs[i]["hypothesis"]`

**Risk**
- `format_history_message` could blow up the prompt size after 10+ iterations. Cap to last N=5 iterations to keep prompt bounded.

---

### 2e.2 Locked LAMMPS script template

**What changes**

Today: every iteration calls the `simulate` tool → simulate_subgraph regenerates a fresh LAMMPS script from prose. Wasteful and inconsistent.

New design:
1. **One-time template generation** at campaign start (before iteration 1).
2. Save the template with knob placeholders as `self.script_template`.
3. **Per-iteration: substitute** the knob values into the template via Python string.replace. No LLM call for script generation.

**Where the template comes from**

Option A: ask LLM to produce a parameterized script with `${T}`, `${strain_rate}`, etc. as LAMMPS variables.

Option B: run the existing `simulate` tool once with mid-range placeholder knob values; then mark up the result post-hoc to make it parameterized.

**Recommended: Option A** — instruct the LLM during template generation:

```
You are generating a TEMPLATE LAMMPS script for a multi-iteration
campaign. The 4 knob values strain_rate, temperature, crack_length_frac,
crack_width_a0 MUST appear as LAMMPS variables that I can substitute
per iteration:

  variable strain_rate equal {{STRAIN_RATE}}
  variable T equal {{TEMPERATURE}}
  variable crack_length_frac equal {{CRACK_LENGTH_FRAC}}
  variable crack_width_a0 equal {{CRACK_WIDTH_A0}}

Use these variables throughout the script. Do NOT bake in concrete
numbers for these 4. All other parameters (system size, KIM model,
phase durations, dump frequency) should be locked at the user-confirmed
values.
```

Then `CampaignManager` does:
```python
def _build_iteration_script(self, knobs: dict) -> str:
    s = self.script_template
    s = s.replace("{{STRAIN_RATE}}", str(knobs["strain_rate"]))
    s = s.replace("{{TEMPERATURE}}", str(knobs["temperature"]))
    s = s.replace("{{CRACK_LENGTH_FRAC}}", str(knobs["crack_length_frac"]))
    s = s.replace("{{CRACK_WIDTH_A0}}", str(knobs["crack_width_a0"]))
    return s
```

**Per-iteration LLM call simplifies dramatically**

Today: full orchestrator → load_skill → simulate tool → subgraph → script gen → review → return.
After: orchestrator gets a simple "propose 4 knob values + hypothesis" prompt → returns typed args → CampaignManager substitutes → runs LAMMPS.

The LLM is no longer doing template work. Saves ~10× tokens per iteration.

**Files touched**
- [src/benchmark/campaign.py](src/benchmark/campaign.py): add `_prepare_script_template()`, called once in `run()` before the loop; add `_build_iteration_script(knobs)`; rewrite `_run_one_iteration` to use it
- [src/orchestrator/skills/campaign.md](src/orchestrator/skills/campaign.md): simplify iteration-mode section (no more "write full simulation specification" — just "propose knobs + hypothesis")
- Possibly: new lightweight tool `propose_iteration_knobs` if we want to bypass the simulate tool entirely for iterations (deferred — Option A is enough for now)

**Risk**
- The LLM-generated template might not actually use the variables consistently. We'll need to validate the template once at campaign start (check all 4 placeholders appear at least once).
- If template generation fails or doesn't use variables properly, fall back to today's behavior (per-iteration regeneration).

---

### 2e.3 Per-iteration final snapshot + report

**What changes**

After each iteration completes (LAMMPS run + Δa measured), render the LAST dump frame to a PNG. Save alongside the run folder.

At campaign end, build a `FINAL_REPORT.md` that interleaves snapshots + hypotheses + Δa per iteration — the "before / mid-campaign / final" visual story the user wants.

**Files touched**
- [src/benchmark/campaign.py](src/benchmark/campaign.py): new `_render_final_snapshot(run_folder, ovito_script)` helper called per iteration; new `_build_final_report()` called once at campaign end
- [src/benchmark/campaign.py](src/benchmark/campaign.py): emit `campaign_final_report` SSE so frontend renders it inline at end
- May need to thread `ovito_script` through CampaignManager (the locked template's OVITO script — generated once with the LAMMPS template in 2e.2)

**Output structure per campaign**
```
runs/benchmark_<id>/
  ├── FINAL_REPORT.md           ← new
  ├── campaign_log.md           ← existing
  ├── summary.json              ← existing (now includes hypothesis + snapshot path per iter)
  └── run_<ts>/                 ← per-iteration LAMMPS output
        ├── log.lammps
        ├── dump.*.load
        └── final_snapshot.png  ← new
```

**Risks**
- OVITO rendering can be slow (~2-5 s per snapshot) — acceptable for end-of-campaign
- OVITO module sometimes hangs on bad GL/headless setup. Need graceful fallback (skip snapshot, record placeholder, continue campaign)

### 2e.4 Verification

**Smoke test sequence:**
1. Run a 3-iteration campaign through the new path.
2. Verify `runs/benchmark_<id>/run_<ts>/input.lammps` files across iterations differ ONLY in the 4 knob values (everything else identical).
3. Verify iteration 3's "Past attempts:" message includes iteration 2's hypothesis text.
4. Manually read the LLM's iter 3 hypothesis to confirm it references prior iterations' reasoning.
5. Verify each `run_<ts>/final_snapshot.png` exists and renders the final frame.
6. Verify `FINAL_REPORT.md` exists at campaign end with all 3 snapshots + hypotheses + Δa.

**Acceptance:**
- Campaign runs end-to-end without crash
- Scripts byte-identical except knob lines
- Iteration N's hypothesis references iter N-1's hypothesis (qualitatively)
- Final report renders all iterations with snapshots + reasoning

---

## 3. Phase 3 — Paper-prep (medium claim)

**Target paper claim:** "Our agent finds robust safe parameter regions for crack suppression more sample-efficiently than random search."

**Headline figure:** P(Δa ≤ 0) of best-region-so-far vs iteration count, for agent and random baseline.

### 3a. Probabilistic objective P(Δa ≤ 0)

**What changes**

Skill objective changes from "minimize Δa" to "maximize P(Δa ≤ 0)".

Implications:
- Per-iteration result is no longer a single Δa number; it's a distribution (3-5 seeds).
- Best-so-far is no longer "lowest Δa observed"; it's the parameter region with highest P(safe).

**Skill prompt changes** ([src/orchestrator/skills/campaign.md](src/orchestrator/skills/campaign.md)):

```
Goal: maximize P(Δa ≤ 0), the probability that crack growth is
suppressed at the proposed parameters. Past attempts now report:

  - mean Δa across multiple seeds
  - stddev (run-to-run variance)
  - P(Δa ≤ 0) directly (fraction of seeds that gave safe outcomes)

Reason about ROBUSTNESS, not just point-best:
  - A region with Δa = 0.0 ± 0.2 Å (low variance, near boundary) is
    PREFERABLE to a single lucky run of Δa = -5.0 with σ unknown
  - Explore broader parameter neighborhoods, don't lock onto a single
    lucky point
  - When choosing knobs, balance exploitation of known-safe regions
    against exploration of under-sampled regions
```

**Files touched**
- [src/orchestrator/skills/campaign.md](src/orchestrator/skills/campaign.md): rewrite objective + iteration-mode reasoning instructions
- [src/benchmark/campaign_prompts.py](src/benchmark/campaign_prompts.py): `format_history_message` reports mean/σ/P(safe)

---

### 3b. Multi-seed per iteration

**What changes**

`CampaignManager._run_one_iteration` runs N seeds (default 3) per iteration:

```python
def _run_one_iteration(self, i, knobs):
    deltas = []
    for seed in self._next_seeds(n=3):
        script = self._build_iteration_script(knobs, seed=seed)
        run_folder = run_lammps_simulation(script)
        delta_a = compute_delta_a(run_folder)
        deltas.append(delta_a)
    return {
        "knobs": knobs,
        "delta_a_mean": statistics.mean(deltas),
        "delta_a_std":  statistics.stdev(deltas) if len(deltas) > 1 else 0,
        "p_safe":       sum(1 for d in deltas if d <= 0) / len(deltas),
        "deltas":       deltas,
        "n_seeds":      len(deltas),
    }
```

Seed is injected into the LAMMPS script as another variable:
```
variable seed equal {{SEED}}
velocity all create ${T} ${seed} dist gaussian mom yes rot yes
```

So the template now has 5 placeholders (4 knobs + seed).

**Wall-clock impact**
- 20 iters × 3 seeds = 60 LAMMPS runs per campaign
- For Cu crack at small system size (~90s per run): ~90 minutes per campaign

**Files touched**
- [src/benchmark/campaign.py](src/benchmark/campaign.py): wrap LAMMPS-run loop with seed iteration
- [src/orchestrator/skills/campaign.md](src/orchestrator/skills/campaign.md) template-generation instructions: include `${seed}` variable
- New config: `CAMPAIGN_N_SEEDS` (default 3)

---

### 3c. Statistical reporting in format_history_message

**Before (Phase 2e)**
```
Iter 1:
  Hypothesis: ...
  Knobs: strain=7e-3, T=300, ...
  Result: Δa = +12.4 Å
```

**After (Phase 3)**
```
Iter 1:
  Hypothesis: ...
  Knobs: strain=7e-3, T=300, ...
  Result: n=3 seeds → Δa = +12.1 ± 2.3 Å, P(Δa≤0) = 0%

Iter 2:
  Hypothesis: ...
  Knobs: strain=2e-3, T=200, ...
  Result: n=3 seeds → Δa = -0.3 ± 0.8 Å, P(Δa≤0) = 67% (2/3 safe)
```

The LLM now reasons about both central tendency AND variance.

---

### 3d. Random baseline mode

**What changes**

A `CampaignManager` variant that ignores the LLM and picks knob values uniformly at random within bounds:

```python
class RandomBaselineCampaign(CampaignManager):
    def _propose_knobs(self, i):
        return {
            knob: random.uniform(*self.bounds[knob])
            for knob in self.bounds
        }
```

User triggers it via either a new tool (`start_random_baseline_campaign(...)`) OR a flag on `start_campaign(..., strategy="random")`.

**Files touched**
- [src/benchmark/campaign.py](src/benchmark/campaign.py): refactor `_run_one_iteration` to call a `_propose_knobs(i)` method that's overridable
- [src/orchestrator/tools.py](src/orchestrator/tools.py): add `strategy` arg to `start_campaign` (default `"agent"`, alternative `"random"`)

**Recommended:** flag on `start_campaign`, not a new tool. The LLM doesn't need to know it can spawn a random baseline — that's a researcher's tool, invoked from a script or admin endpoint.

---

### 3e. Paper figure generation

A script (NOT in the live app) that reads campaign artifacts from disk and produces figures:

```python
# scripts/plot_campaign_results.py
- Reads runs/benchmark_<id>/summary.json from N campaigns
- Computes best-P(safe)-so-far per iteration
- Plots agent runs (with stddev band) vs random runs (with stddev band)
- Saves figure to docs/figures/
```

**Files added**
- `scripts/plot_campaign_results.py` (matplotlib)
- `scripts/run_paper_experiments.py` (orchestrates: 3 agent campaigns + 3 random campaigns)
- `docs/paper_experiments.md` (how to reproduce)

---

## 4. Out of scope (for now)

- **Grid search baseline** — defer to "Full" claim later if needed
- **Bayesian optimization layer** — defer; LLM-only suffices for medium claim
- **Multi-objective optimization** (e.g. maximize P(safe) AND minimize variance) — defer
- **Real Δa noise model** — we measure noise empirically, not parametrically
- **Continuing a campaign across server restarts** — campaigns are still one-shot

---

## 5. Order of operations

### Week 1
- Day 1: Phase 2e.1 (hypothesis memory) + smoke test
- Day 2-3: Phase 2e.2 (locked script template) + smoke test
- Day 4: Phase 2e verification — run a 3-iteration campaign, confirm reasoning is coherent
- Day 5: write up Phase 2e "before/after" comparison (one paragraph for the paper)

### Week 2
- Day 1: Phase 3a (skill prompt rewrite for P(Δa≤0))
- Day 2: Phase 3b (multi-seed runs)
- Day 3: Phase 3c (statistical history)
- Day 4: Phase 3d (random baseline)
- Day 5: Phase 3e (figure script + first real experiment)

### Week 3
- Run actual paper experiments (3 agent + 3 random campaigns)
- Aggregate, plot, iterate
- Decide if you want to add Phase 3-Full (grid search) based on what reviewers might ask

---

## 6. Cost estimate

| Phase | Engineering time | Compute |
|---|---|---|
| 2e | 2-3 days | minimal (3-iter smoke tests) |
| 3a-d | 4-5 days | ~90 min per agent campaign |
| 3e (experiments) | 1-2 days code + 1-2 days running | 6 campaigns × 90 min = ~9 hours of LAMMPS |
| **Total to paper-ready** | **~2 weeks** | **~10-15 hours of compute** |

---

## 7. Acceptance criteria for "paper-ready"

A reviewer reading the paper sees:

✓ Headline figure: P(Δa ≤ 0) vs iteration, agent line above random with non-overlapping confidence bands by iteration ~10
✓ Per-iteration hypothesis table (Table 1) showing the agent's reasoning trajectory — readable as a scientific narrative
✓ Parameter-space trajectory plot showing agent zeroing in on a region vs random scattering
✓ Methods section: clear description of objective, multi-seed evaluation, baseline
✓ Reproducible: scripts + config in repo so reviewers could re-run
✓ Honest about variance: error bars on every line, multiple trials per condition

If any of those are missing, it's not ready.

---

## 8. Open questions for later

- How many seeds per iteration is "enough"? (3 is minimal; 5 is comfortable; 10 is rigorous)
- Should the agent see the seed-level deltas, or only summary statistics? (Probably summary — exposes less noise to the LLM)
- Should we add a "convergence detector" that ends the campaign early if P(safe) plateaus for K iterations? (Nice-to-have, not required)
- Do we want to publish the SKILL.md as supplementary material? (Yes, probably — shows transparency)
