# Campaign Vision Cleanup — plan (scoped to campaign vision only)

## Context (why this change)

While reviewing the campaign visualization tool (`campaign_visualize_all`), three things
surfaced:

1. **A real bug** — campaign's color-legend has only the `if ovito_script:` branch and no
   `else`, so a campaign run with **no custom OVITO script** sends the vision model no
   warning and it can hallucinate FCC/HCP structure meaning from plain particle-type colors.
   (Standalone `visualize_run` has both branches and is correct.)

2. **Campaign vision isn't actually "separate" yet** — it currently imports standalone's
   `_VIZ_PROMPT`. We want it separate on purpose, because campaign vision will later write its
   summary into `trials.jsonl` to feed **hypothesis generation**, so its prompt needs to
   become crack-specific/structured independently of standalone. Sharing the prompt today
   would force a teardown later.

3. **Prompt-location inconsistency** — every LLM prompt in the project lives in `prompts.py`
   (`ANALYST_SYSTEM_PROMPT`, `HPC_EXECUTOR_PROMPT`, orchestrator/agent prompts) **except** the
   vision prompt, which sits inline in `orchestrator/tools.py`. It's the one stray exception,
   not a convention.

This pass fixes only these. It does **not** unify standalone and campaign vision (kept
separate by design), and does **not** build the trials.jsonl → hypothesis-gen loop (future
feature). Other cleanup (dead code, viz_run.py move, AGENTS map) is deferred to a later pass.

## Decisions locked in
- Standalone and campaign vision stay **separate** (different futures).
- Both vision prompts move to **`prompts.py`** (fixes the stray-exception inconsistency;
  a prompt can still diverge later while living in prompts.py — divergence = own constant,
  not own file).
- Campaign gets its **own** prompt constant now (identical text for now, diverges later).

## The work (3 edits + prompt moves)

### 1. Move the standalone vision prompt into `prompts.py`
- Cut `_VIZ_PROMPT` from [tools.py:725–734](../src/orchestrator/tools.py#L725) → define it in
  `prompts.py` (near `ANALYST_SYSTEM_PROMPT`). Keep the name `_VIZ_PROMPT` (or rename to
  `VISION_ANALYSIS_PROMPT` — cosmetic; keeping the name minimizes churn).
- In `tools.py`, add `from prompts import _VIZ_PROMPT` (tools.py currently imports nothing
  from prompts.py). Usage at [tools.py:681](../src/orchestrator/tools.py#L681) unchanged.

### 2. Add `CAMPAIGN_VIZ_PROMPT` in `prompts.py` and point campaign at it
- Add `CAMPAIGN_VIZ_PROMPT` in `prompts.py` — **identical text to `_VIZ_PROMPT` for now**,
  its own constant so it can become crack-specific/structured later without touching
  standalone.
- In `campaign_tools.py`:
  - change the import at [line 809](../src/orchestrator/campaign_tools.py#L809)
    `from orchestrator.tools import _pick_viz_frames, _VIZ_PROMPT`
    → `from orchestrator.tools import _pick_viz_frames` **plus**
    `from prompts import CAMPAIGN_VIZ_PROMPT`.
  - change the usage at [line 901](../src/orchestrator/campaign_tools.py#L901)
    `_VIZ_PROMPT` → `CAMPAIGN_VIZ_PROMPT`.
- Result: campaign no longer depends on standalone's prompt; it's genuinely separate.

### 3. Fix the campaign legend bug (add the missing `else`)
At [campaign_tools.py:894–901](../src/orchestrator/campaign_tools.py#L894), add the else
branch so it matches standalone's safety warning:

```python
if ovito_script:
    content.append({... "Interpret colors per its color_map ..."})
else:                                          # ← ADD
    content.append({
        "type": "text",
        "text": ("No custom OVITO script was used. Atoms are colored by LAMMPS "
                 "particle type (NOT crystal structure). Do not interpret colors "
                 "as FCC/HCP/BCC. Focus on gross spatial features."),
    })
content.append({"type": "text", "text": CAMPAIGN_VIZ_PROMPT})
```

## Files touched
- `src/prompts.py` — add `_VIZ_PROMPT` (moved) + `CAMPAIGN_VIZ_PROMPT` (new).
- `src/orchestrator/tools.py` — remove local `_VIZ_PROMPT`, import it from prompts.py.
- `src/orchestrator/campaign_tools.py` — import `CAMPAIGN_VIZ_PROMPT`, use it, add legend else.

## Explicitly NOT in this pass
- NOT unifying standalone + campaign vision (separate by design).
- NOT building the trials.jsonl / hypothesis-gen wiring (future feature).
- NOT touching dead code, viz_run.py, or the AGENTS map (later pass).
- NOT changing the vision LLM call, frame picking, or report writers.

## Verification
1. **Imports resolve, no stray refs:**
   `cd src && pixi run python -c "from prompts import _VIZ_PROMPT, CAMPAIGN_VIZ_PROMPT; from orchestrator.tools import visualize_run; from orchestrator.campaign_tools import campaign_visualize_all; print('ok')"`
2. `git grep -n "_VIZ_PROMPT" src/` → appears only in prompts.py (def), tools.py (import+use);
   `git grep -n "CAMPAIGN_VIZ_PROMPT" src/` → prompts.py (def) + campaign_tools.py (import+use).
3. `pixi run run-app` → run a 1-iteration campaign with **NO** OVITO script → open
   `runs/<campaign_id>/viz_report.md` → confirm the summary reflects the "colored by particle
   type, not structure" caveat (proves the legend bug fix). A single-run
   `visualize_run` still streams correctly (proves the standalone prompt move works).

## Risk & rollback
- Small, mechanical edits: one prompt moved, one added, one import repointed, one else added.
- Single commit → `git revert`-able. No runtime data touched; launch command unchanged.
