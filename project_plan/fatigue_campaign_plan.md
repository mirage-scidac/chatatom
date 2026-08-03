# Fatigue Campaign — plan (enable cyclic-loading campaigns end to end)

## Context (why this change)

Fatigue (cyclic-loading) campaigns are *almost* fully supported. What the campaign skill
(`src/orchestrator/skills/campaign.md`) already says about fatigue:

- **Template inference** (lines 51-52): `"Run a campaign on fatigue in Cu"` → 4-phase
  template (adds a cyclic-loading phase).
- **Knobs** (lines 76-77): for cyclic campaigns, replace `strain_rate` with
  `strain_amplitude` (0.001–0.01 fraction of L_y).
- **Fatigue phase spec** (lines 340-352): Phase 3 = passive relaxation (NVT, 10 ps, no
  `fix deform`); Phase 4 = cyclic loading, `fix cyclic all deform 1 y wiggle amplitude
  10.0 units box`, 5 cycles over 50 ps.
- **Dump names** (line 363): fatigue writes `dump.*.equil / .relax / .observe / .load`,
  with the cyclic loading in the `dump.*.load` (Phase 4) frames.

So the skill fully describes **how to build and run** a fatigue simulation. The gap is the
**objective metric**:

- AVAILABLE METRICS (lines 26-34) lists **only** `delta_cavity_surface_area` — the
  *monotonic* 2-frame endpoint delta (last relax frame → last load frame). It
  under-reports for cyclic loading, because the last frame is near-zero strain and the
  crack has closed back.
- The correct fatigue extractor — `compute_peak()` in
  `src/benchmark/metrics/surface_void.py:297` — **is written and correct**. It scans every
  `dump.*.load` frame and returns `peak_delta_cavity_surface_area` = worst-case crack
  opening over the cycle, baseline-subtracted so the pre-existing notch does not inflate
  the score.
- But `compute_peak()` is **not in `METRIC_REGISTRY`**
  (`src/benchmark/metrics/__init__.py:21`). Since `campaign_start` validates
  `plan.objective.metric` against `list_metrics()` (`src/orchestrator/campaign_tools.py:151`),
  there is currently **no metric name a fatigue campaign can legally select**.

The rest of the pipeline is metric-agnostic — the objective value is read generically as
`trial.metrics[metric_name]` (`campaign_tools.py:178-188`), and the metric subprocess runs
`compute_metric(name, folder)` through the registry (`campaign_tools.py:457-461`). So
**registering the metric is the single change that unblocks fatigue campaigns end to end.**

## Decisions (recommended defaults)

1. **Metric name** — register under the existing key `peak_delta_cavity_surface_area`
   (zero code churn; the registry name must equal the dict key `compute_peak()` returns,
   because `_metric_value` looks the objective up by that key). *Alternative:* rename to
   `fatigue_crack_area`, which would also require changing `compute_peak()`'s return key.
   → **Recommended: keep the existing key.**
2. **Selection** — the campaign skill auto-picks the fatigue metric for fatigue/cyclic
   requests, the same way it already infers the 3- vs 4-phase template
   (`campaign.md:49-52`). *Alternative:* list both and let the LLM/user choose per campaign.
   → **Recommended: auto-select by campaign type.**

## Changes

### 1. Register the metric — `src/benchmark/metrics/__init__.py`

Add a second entry to `METRIC_REGISTRY`:

```python
"peak_delta_cavity_surface_area": {
    "compute":      _surface_void.compute_peak,
    "direction":    "minimize",
    "units":        "Å²",
    "display_name": "Peak Δ crack surface area (fatigue)",
},
```

This is the only functional line. `list_metrics()`, `compute_metric()`, and the metric
subprocess already handle any registered name generically. `compute_peak` is already
imported via `from . import surface_void as _surface_void`.

### 2. Advertise it in the skill — `src/orchestrator/skills/campaign.md`

- **AVAILABLE METRICS** (lines 26-34): add a bullet for `peak_delta_cavity_surface_area`,
  labeling `delta_cavity_surface_area` as the **monotonic** objective and
  `peak_delta_cavity_surface_area` as the **fatigue / cyclic** objective (peak crack
  opening over the cycle, baseline-subtracted).
- **Selection rule** (near the fatigue section, lines 340-352): add one line — *fatigue /
  cyclic campaigns use `peak_delta_cavity_surface_area`; monotonic campaigns use
  `delta_cavity_surface_area`.*
- **Objective example** (lines 143-151): keep the monotonic block as the default; add a
  short "for fatigue, use:" variant with `metric: "peak_delta_cavity_surface_area"`,
  `direction: "minimize"`, `units: "Å²"`, `display_name: "Peak Δ crack surface area"`.

### 3. Tidy one stale docstring — `src/orchestrator/campaign_tools.py:507`

`campaign_measure_metrics`'s docstring says *"Currently registered:
'delta_cavity_surface_area'."* Update to list both. Cosmetic; the tool dispatches by name
regardless.

## Out of scope (not touching unless asked)

- `compute_peak()` logic itself — already written and correct.
- The `_render_surface_png` / visualization path — unaffected.
- `reporting.py` — metric-agnostic (reads from the plan); no change. Its hardcoded fallback
  default stays `delta_cavity_surface_area` (only used if a plan has no objective, which
  never happens for a real campaign).

## Verification

1. **Registry unit check** (fast, no simulation):
   ```
   python -c "from src.benchmark.metrics import list_metrics; print(list_metrics())"
   ```
   Expect both `delta_cavity_surface_area` and `peak_delta_cavity_surface_area`.
2. **Extractor on a real fatigue run** (if one exists under `runs/` with `dump.*.load`):
   ```
   python -c "from src.benchmark.metrics import compute_metric; import json; \
     print(json.dumps(compute_metric('peak_delta_cavity_surface_area', '<run_folder>'), indent=2))"
   ```
   Expect a dict with `peak_delta_cavity_surface_area`, `peak_timestep`, `n_load_frames`,
   `closure_area`.
3. **End-to-end**: start a fatigue campaign ("run a fatigue campaign on Cu"). Confirm the
   skill proposes `peak_delta_cavity_surface_area` as the objective, `campaign_start`
   accepts it (no "metric not registered" error), and the per-iteration terminal line +
   final report show the peak value.
