#!/usr/bin/env python3
# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE
"""Render publication-quality stills from a run, colored the way that run was.

The live trajectory viewer colors each run with the `ovito_script.py` the
Visualization agent generated for it, so two runs can legitimately use two
different palettes. A still that hardcodes one palette therefore disagrees with
whatever the user watched. This script reads the run's own script and applies it
through the same executor the live render server uses, so the figure and the
viewer agree by construction.

By default nothing is recalculated. Structure types come from the `c_cna`
column LAMMPS already wrote, and coloring resolves in two steps:

  1. the run's ovito_script.py (or --ovito-script), via DynamicOVITOExecutor;
  2. the stored `c_cna` column plus the project palette, if that script is
     missing or fails.

Step 2 exists because generated scripts are LLM-written and a few are broken —
one variant in runs/ raises KeyError on `type_by_id` at compute time, not at
exec time, so a script is only known to be bad once a frame is evaluated.

Recomputing CNA here requires asking for it by name with --cna adaptive. It is
never a silent fallback: adaptive and stored disagree by enough to matter (7403
vs 8560 "Other" on frame 120000 of Cu_final), so a dump without a `c_cna`
column raises rather than quietly substituting a different classifier.

Three of the 18 generated scripts in runs/ call CommonNeighborAnalysisModifier
unconditionally and so overwrite the stored column even on the default path.
That cannot be prevented from here, but it is detected and warned about by
comparing the classification before and after the script runs; --cna stored
bypasses the script entirely when the dump's own values are what you want.

Usage:
    python render_still.py runs/<run>                          # last frame
    python render_still.py runs/<run> --timesteps 20000 76000 120000
    python render_still.py runs/<run> --preset publication -o fig/crack
    python render_still.py runs/<run> --cna adaptive --colors ovito
    python render_still.py runs/<run>/dump.20000.loading       # one dump directly
"""
from __future__ import annotations

import argparse
import glob
import os
import sys


class RenderError(Exception):
    """Raised for user-fixable problems (no dumps, bad timestep, no OVITO).

    Deliberately not SystemExit: this module is imported by the Analyst agent's
    render_still tool, where a raised SystemExit would abort the whole agent
    turn instead of coming back as a message the model can act on.
    """


# Structure-type palettes. Keys are OVITO's encoding, which is what remains
# after the LAMMPS 5 -> 0 remap in _color_by_stored_cna.
COLOR_MAPS = {
    # The palette the generated ovito_script.py files use. Fully saturated, so
    # it matches the live viewer exactly — but a shaded sphere of (0,1,0) has
    # nothing in R or B to fall back on and darkens toward black, which reads as
    # speckle across the bulk. --renderer opengl or --colors ovito avoids that.
    "project": {
        0: (1.0, 0.0, 0.0),   # Other → red
        1: (0.0, 1.0, 0.0),   # FCC   → green
        2: (1.0, 1.0, 0.0),   # HCP   → yellow
        3: (0.0, 0.0, 1.0),   # BCC   → blue
        4: (0.5, 0.5, 0.5),   # ICO   → gray
    },
    # OVITO's own defaults. Every channel sits off zero, so shaded spheres stay
    # light instead of going black. NOTE red means HCP here and Other in
    # "project" — do not mix the two inside one figure.
    "ovito": {
        0: (0.95, 0.95, 0.95),  # Other → near-white
        1: (0.40, 1.00, 0.40),  # FCC   → light green
        2: (1.00, 0.40, 0.40),  # HCP   → salmon
        3: (0.40, 0.40, 1.00),  # BCC   → light blue
        4: (0.95, 0.80, 0.20),  # ICO   → gold
    },
}

STRUCTURE_NAMES = {0: "Other", 1: "FCC", 2: "HCP", 3: "BCC", 4: "ICO"}

BACKGROUNDS = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "gray":  (0.9, 0.9, 0.9),
}

# Quality presets. Pixels per atom dominates perceived quality: a 1000px image
# of a 100k-atom slab gives each atom ~3px and looks soft no matter how the
# sampling is set. Tachyon at these sizes costs ~1-2 s with flat shading, so
# "normal" is not a meaningful saving over "publication" except at 3200px.
PRESETS = {
    "draft":       dict(size=(1000, 1000), aa=6,  ao=6,  renderer="opengl"),
    "normal":      dict(size=(1400, 1400), aa=12, ao=12, renderer="tachyon"),
    "publication": dict(size=(3200, 3200), aa=32, ao=24, renderer="tachyon"),
}

# Module-level so the OVITO PythonModifier closure below can see the palette
# chosen by the caller without threading it through OVITO's callback signature.
_ACTIVE_COLORS = COLOR_MAPS["project"]


# ─── Dump discovery ──────────────────────────────────────────────────────────

def timestep_of(dump_path: str) -> int:
    """Extract the timestep from a dump filename, 0 if absent.

    Matches viz_run._timestep_of so frame ordering is identical to the live
    viewer's. Handles both conventions in runs/:
        dump.<timestep>.<phase>      e.g. dump.20000.loading
        dump.<name>.<timestep>.dat   e.g. dump.Cu_crack.44800.dat
    """
    for token in os.path.basename(dump_path).split("."):
        if token.isdigit():
            return int(token)
    return 0


def find_dumps(run_folder: str, timesteps=None) -> list:
    """Return dump paths in timestep order, optionally filtered to `timesteps`.

    With timesteps=None only the last frame is returned, which is the common
    case for "show me the crack" — rendering all 30 frames of a run by accident
    is slow enough to look like a hang.
    """
    candidates = [p for p in glob.glob(os.path.join(run_folder, "dump.*"))
                  if not p.endswith((".png", ".jpg", ".svg"))]
    if not candidates:
        raise RenderError(f"No dump files found in {run_folder}")

    candidates.sort(key=timestep_of)

    if timesteps is None:
        return [candidates[-1]]

    by_step = {}
    for p in candidates:
        by_step.setdefault(timestep_of(p), p)

    picked, missing = [], []
    for want in timesteps:
        if want in by_step:
            picked.append(by_step[want])
        else:
            missing.append(want)
    if missing:
        available = ", ".join(str(s) for s in sorted(by_step))
        raise RenderError(
            f"No dump at timestep(s) {missing}. Available: {available}")
    return picked


def load_ovito_script(run_folder: str) -> str | None:
    """Read the run's generated ovito_script.py, or None if it has none."""
    path = os.path.join(run_folder, "ovito_script.py")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return fh.read()


# ─── Coloring ────────────────────────────────────────────────────────────────

def _color_by_stored_cna(frame, data):
    """Color atoms from the `c_cna` column already in the dump.

    OVITO auto-renames the LAMMPS `c_cna` column to 'Structure Type' on import
    but leaves the values LAMMPS-encoded (1=FCC .. 5=Other); remap 5 -> 0 to
    OVITO's encoding first. Both writes go through create_property — direct
    [...] assignment fails on auto-imported properties.
    """
    import numpy as np

    st = np.asarray(data.particles["Structure Type"])
    st = np.where(st >= 5, 0, st).astype(np.int32)
    data.particles_.create_property("Structure Type", data=st)

    colors = np.zeros((len(st), 3), dtype=np.float64)
    for type_id, rgb in _ACTIVE_COLORS.items():
        colors[st == type_id] = rgb
    data.particles_.create_property("Color", data=colors)


def _color_by_adaptive_cna(frame, data):
    """Recompute CNA with an adaptive cutoff, then color by the result.

    Applied in-place via data.apply() rather than as a separate pipeline
    modifier so the coloring stays one step and the fallback chain in
    apply_coloring can swap implementations without reordering the pipeline.
    """
    import numpy as np
    from ovito.modifiers import CommonNeighborAnalysisModifier

    data.apply(CommonNeighborAnalysisModifier(
        mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff))

    st = np.asarray(data.particles["Structure Type"]).astype(np.int32)
    colors = np.zeros((len(st), 3), dtype=np.float64)
    for type_id, rgb in _ACTIVE_COLORS.items():
        colors[st == type_id] = rgb
    data.particles_.create_property("Color", data=colors)


def _has_stored_cna(pipeline) -> bool:
    """True if the dump carried a `c_cna` column OVITO mapped to Structure Type."""
    try:
        return "Structure Type" in pipeline.compute(0).particles.keys()
    except Exception:
        return False


def _stored_structure_types(pipeline):
    """The dump's own c_cna values, remapped 5 -> 0, or None if absent.

    Read before any coloring runs so _warn_if_recomputed can tell afterwards
    whether a generated script honoured the stored column or replaced it.
    """
    import numpy as np

    try:
        data = pipeline.compute(0)
    except Exception:                    # noqa: BLE001 — absence is the answer
        return None
    if "Structure Type" not in data.particles.keys():
        return None
    st = np.asarray(data.particles["Structure Type"])
    return np.where(st >= 5, 0, st).astype(np.int32)


def _warn_if_recomputed(pipeline, before):
    """Return a warning if the script's classification differs from the dump's.

    Most generated scripts guard their CNA call behind a 'Structure Type' in
    data.particles check and so reuse c_cna, but 3 of the 18 in runs/ call
    CommonNeighborAnalysisModifier unconditionally and overwrite it. Detecting
    that by grepping the source is unreliable — the guard is written as both
    `if ... in ...:` / `else:` and `if ... not in ...:` — so compare the values
    instead. The comparison is exact and costs nothing beyond a compute already
    being done for validation.
    """
    if before is None:
        return None

    import numpy as np

    try:
        after = pipeline.compute(0)
        if "Structure Type" not in after.particles.keys():
            return None
        n_diff = int((before != np.asarray(after.particles["Structure Type"])).sum())
    except Exception:                    # noqa: BLE001 — never fail on a check
        return None

    if not n_diff:
        return None
    return (f"this run's ovito_script.py recomputed CNA itself rather than using "
            f"the stored c_cna column — {n_diff} atoms ({100.0 * n_diff / len(before):.1f}%) "
            f"are classified differently. Pass --cna stored to force the dump's "
            f"own values.")


def apply_coloring(pipeline, ovito_script=None, cna="auto"):
    """Attach a coloring step to `pipeline`; return (pipeline, source, warnings).

    `source` is a short human-readable label naming which route won, so callers
    can tell the user which palette and which classifier produced the picture
    rather than leaving them to guess.

    A generated script is validated by evaluating frame 0 before it is accepted:
    several failure modes (a bad type_by_id lookup, a property that does not
    exist) only raise at compute time, well after exec() returned cleanly.
    """
    from ovito.modifiers import PythonModifier

    warnings = []

    if ovito_script and cna == "auto":
        repo_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
        if repo_src not in sys.path:
            sys.path.append(repo_src)
        try:
            from ovito_executor import DynamicOVITOExecutor
            before = _stored_structure_types(pipeline)
            data = pipeline.compute(0)
            pipeline = DynamicOVITOExecutor().execute_script(
                ovito_script, pipeline, data)
            pipeline.compute(0)          # trial evaluation; see docstring
            recomputed = _warn_if_recomputed(pipeline, before)
            if recomputed:
                warnings.append(recomputed)
            return pipeline, "run's ovito_script.py", warnings
        except Exception as exc:         # noqa: BLE001 — any failure falls back
            warnings.append(
                f"the run's ovito_script.py failed ({type(exc).__name__}: {exc}); "
                "used the built-in palette instead")
            pipeline.modifiers.clear()

    # Adaptive CNA runs only when named explicitly. It reclassifies every atom
    # and disagrees with the stored column by enough to matter (7403 vs 8560
    # "Other" on frame 120000 of Cu_final), so falling back to it silently would
    # change the reported numbers without the caller ever seeing why. A dump
    # with no c_cna is therefore an error, not an invitation to recompute.
    if cna == "adaptive":
        pipeline.modifiers.append(PythonModifier(function=_color_by_adaptive_cna))
        return pipeline, "built-in palette, recomputed adaptive CNA", warnings

    if not _has_stored_cna(pipeline):
        raise RenderError(
            "This dump has no c_cna column, so there is no stored CNA to color "
            "by. Add 'compute cna/atom' to the LAMMPS dump fields for future "
            "runs, or pass --cna adaptive to recompute it here (the numbers "
            "will not match a stored-CNA figure).")

    pipeline.modifiers.append(PythonModifier(function=_color_by_stored_cna))
    return pipeline, "built-in palette, stored CNA (from the dump)", warnings


# ─── Rendering ───────────────────────────────────────────────────────────────

def _build_renderer(renderer, aa, ao, shading):
    """Construct the OVITO renderer for the requested quality settings.

    'flat' shading reproduces the GUI viewport: no ambient occlusion, no
    shadows. AO darkens the gaps between spheres, and looking down through a
    multi-layer slab it shades deeper layers unevenly, which reads as blotchy
    mottling rather than depth. 'realistic' turns both back on for single-layer
    or perspective views where they do help.
    """
    from ovito.vis import TachyonRenderer, OpenGLRenderer

    if renderer != "tachyon":
        return OpenGLRenderer()

    realistic = shading == "realistic"
    return TachyonRenderer(
        antialiasing=True, antialiasing_samples=aa,
        ambient_occlusion=realistic, ambient_occlusion_samples=ao,
        shadows=realistic, direct_light=True,
    )


def _structure_counts(data) -> tuple:
    """Count atoms per structure type; ({}, 0) if the pipeline produced none.

    A custom ovito_script.py is free to color by stress or displacement and
    never create a Structure Type property, so this must not assume one.
    """
    if "Structure Type" not in data.particles.keys():
        return {}, 0

    import numpy as np
    st = np.asarray(data.particles["Structure Type"])
    counts = {name: int((st == tid).sum()) for tid, name in STRUCTURE_NAMES.items()}
    return counts, max(sum(counts.values()), 1)


def _setup_scene(data, radius, shape, show_cell):
    """Apply the visual settings shared by the camera pass and the render pass."""
    data.particles.vis.radius = radius
    if shape == "circle":
        from ovito.vis import ParticlesVis
        data.particles.vis.shape = ParticlesVis.Shape.Circle
    if data.cell is not None:
        data.cell.vis.enabled = show_cell


def make_camera(dump_path, *, view, size, radius, shape, show_cell,
                ovito_script=None, cna="auto"):
    """Zoom to fit one frame and return (pos, dir, fov) for reuse on the others.

    Call this on the frame with the largest box. Reusing the result is what
    keeps a propagation sequence to scale: zoom_all() per frame rescales each
    panel to its own box, which hides the cell elongation under load — the very
    thing a multi-panel crack figure is meant to show.
    """
    from ovito.io import import_file
    from ovito.vis import Viewport

    pipeline = import_file(dump_path)
    pipeline, _, _ = apply_coloring(pipeline, ovito_script, cna)
    pipeline.add_to_scene()
    try:
        _setup_scene(pipeline.compute(0), radius, shape, show_cell)
        vp = Viewport(type=getattr(Viewport.Type, view))
        vp.zoom_all(size=tuple(size))
        return vp.camera_pos, vp.camera_dir, vp.fov
    finally:
        pipeline.remove_from_scene()
        del pipeline


def render_one(dump_path, out_png, *, frame=-1, size=(1400, 1400), radius=1.2,
               view="Top", background="black", show_cell=False,
               renderer="tachyon", aa=12, ao=24, crop=True, shading="flat",
               downscale_to=None, shape="sphere", camera=None,
               ovito_script=None, cna="auto"):
    """Render one frame of `dump_path` to `out_png`.

    Returns (counts, total, coloring_source, warnings). `camera` is an optional
    (pos, dir, fov) triple from make_camera(); when given the viewport uses it
    verbatim instead of zooming to this frame.
    """
    from ovito.io import import_file
    from ovito.vis import Viewport

    pipeline = import_file(dump_path)

    n_frames = pipeline.source.num_frames
    if frame < 0:
        frame = n_frames + frame
    if not 0 <= frame < n_frames:
        raise RenderError(f"frame {frame} out of range ({dump_path} has {n_frames})")

    pipeline, source, warnings = apply_coloring(pipeline, ovito_script, cna)
    pipeline.add_to_scene()
    try:
        data = pipeline.compute(frame)
        _setup_scene(data, radius, shape, show_cell)
        counts, total = _structure_counts(data)

        vp = Viewport(type=getattr(Viewport.Type, view))
        if camera is None:
            vp.zoom_all()
        else:
            vp.camera_pos, vp.camera_dir, vp.fov = camera

        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        vp.render_image(
            filename=out_png,
            size=tuple(size),
            background=BACKGROUNDS.get(background, BACKGROUNDS["black"]),
            renderer=_build_renderer(renderer, aa, ao, shading),
            crop=crop,
        )

        # Render big, then resample down: supersampling keeps the high-sample
        # antialiasing while returning each atom to the few pixels it occupies
        # in the GUI, where per-sphere shading averages out. Rendering directly
        # at the small size would alias the edges instead.
        if downscale_to:
            warnings.extend(_downscale(out_png, downscale_to))

        return counts, total, source, warnings
    finally:
        pipeline.remove_from_scene()
        del pipeline


def _downscale(png_path, width) -> list:
    """Resample `png_path` to `width` in place; return warnings, never raise.

    Pillow is not a hard dependency of this repo and is absent from at least one
    working env, so a missing PIL must degrade to a full-size image with a note
    rather than throwing away a render that already succeeded.
    """
    try:
        from PIL import Image
    except ImportError:
        return ["Pillow is not installed, so --downscale-to was skipped "
                "(the image is full size). Install it with: pip install pillow"]

    with Image.open(png_path) as im:
        height = round(im.height * width / im.width)
        im.resize((width, height), Image.LANCZOS).save(png_path)
    return []


def render_stills(run_or_dump, *, timesteps=None, out=None, preset="normal",
                  colors="project", cna="auto", ovito_script=None,
                  view="Top", background="black", show_cell=False,
                  radius=1.2, shape="sphere", shading="flat",
                  size=None, aa=None, ao=None, renderer=None,
                  downscale_to=None, crop=True, shared_camera=True) -> dict:
    """Render one or more stills from a run folder or a single dump file.

    This is the entry point the Analyst agent's tool calls; the CLI below is a
    thin wrapper over it. Returns a dict rather than printing so the caller
    decides how to surface the result:

        {"run": str, "coloring": str, "warnings": [str],
         "images": [{"timestep": int, "path": str, "dump": str,
                     "counts": {...}, "total": int}]}
    """
    try:
        import ovito  # noqa: F401
    except ImportError as exc:
        raise RenderError(
            "OVITO is not available in this environment, so no image can be "
            f"rendered ({exc}).") from exc

    global _ACTIVE_COLORS
    if colors not in COLOR_MAPS:
        raise RenderError(f"Unknown palette '{colors}'. Choose from: "
                          f"{', '.join(sorted(COLOR_MAPS))}")
    _ACTIVE_COLORS = COLOR_MAPS[colors]

    if preset not in PRESETS:
        raise RenderError(f"Unknown preset '{preset}'. Choose from: "
                          f"{', '.join(sorted(PRESETS))}")
    p = PRESETS[preset]
    size = tuple(size or p["size"])
    aa = p["aa"] if aa is None else aa
    ao = p["ao"] if ao is None else ao
    renderer = renderer or p["renderer"]

    # A single dump path and a run folder are both accepted: chat users name a
    # run, while someone iterating on a figure names the exact file.
    if os.path.isdir(run_or_dump):
        run_folder = os.path.abspath(run_or_dump)
        dumps = find_dumps(run_folder, timesteps)
    elif os.path.isfile(run_or_dump):
        run_folder = os.path.dirname(os.path.abspath(run_or_dump))
        dumps = [run_or_dump]
    else:
        raise RenderError(f"Not a run folder or a dump file: {run_or_dump}")

    if ovito_script is None:
        ovito_script = load_ovito_script(run_folder)

    multi = len(dumps) > 1
    # Cropping trims to content, which differs per frame and would misalign the
    # panels of a sequence. Silently disabling beats emitting unusable figures.
    crop = crop and not multi

    camera = None
    if shared_camera and multi:
        widest = max(dumps, key=lambda d: _box_extent(d))
        camera = make_camera(widest, view=view, size=size, radius=radius,
                             shape=shape, show_cell=show_cell,
                             ovito_script=ovito_script, cna=cna)

    images, warnings, source = [], [], "unknown"
    for dump in dumps:
        step = timestep_of(dump)
        if out and not multi:
            png = out if out.endswith(".png") else out + ".png"
        else:
            prefix = out or os.path.join(run_folder, "figures", "still")
            png = f"{prefix}_{step}.png"

        counts, total, source, warn = render_one(
            dump, png, size=size, radius=radius, view=view,
            background=background, show_cell=show_cell, renderer=renderer,
            aa=aa, ao=ao, crop=crop, shading=shading,
            downscale_to=downscale_to, shape=shape, camera=camera,
            ovito_script=ovito_script, cna=cna,
        )
        for w in warn:
            if w not in warnings:      # the same script fails identically per frame
                warnings.append(w)
        images.append({"timestep": step, "path": png, "dump": dump,
                       "counts": counts, "total": total})

    return {"run": os.path.basename(os.path.normpath(run_folder)),
            "coloring": source, "warnings": warnings, "images": images}


def _box_extent(dump_path: str) -> float:
    """Largest box edge from a dump header, read without starting OVITO.

    Only the first frame's header is parsed: one file per timestep is the normal
    case here, and for a concatenated file the camera is sized from the union
    anyway. Returns 0.0 on a malformed header so camera selection degrades to
    "pick any frame" instead of failing the whole render.
    """
    bounds, mode = [], None
    try:
        with open(dump_path, errors="replace") as fh:
            for line in fh:
                if line.startswith("ITEM: BOX BOUNDS"):
                    mode = "box"
                    continue
                if line.startswith("ITEM:"):
                    if len(bounds) == 3:
                        break
                    mode = None
                    continue
                if mode == "box":
                    lo, hi = (float(x) for x in line.split()[:2])
                    bounds.append(hi - lo)
                    if len(bounds) == 3:
                        break
    except (OSError, ValueError):
        return 0.0
    return max(bounds) if bounds else 0.0


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="run folder, or a single dump file")
    ap.add_argument("--timesteps", type=int, nargs="+", default=None,
                    help="timesteps to render (default: last frame only). "
                         "Several render as a sequence sharing one camera.")
    ap.add_argument("-o", "--out", default=None,
                    help="output PNG for a single frame, or a filename prefix "
                         "for several (default: <run>/figures/still_<step>.png)")
    ap.add_argument("--preset", default="normal", choices=sorted(PRESETS),
                    help="quality preset; sets size, sampling and renderer "
                         "(publication = 3200px, 32x AA, Tachyon)")
    ap.add_argument("--cna", default="auto", choices=["auto", "stored", "adaptive"],
                    help="where structure types come from. auto (default) uses "
                         "the run's ovito_script.py, falling back to the stored "
                         "c_cna column; stored skips the script and always uses "
                         "the dump's own values; adaptive recomputes CNA here, "
                         "which reclassifies every atom and will not match a "
                         "stored-CNA figure. Nothing is recomputed unless you "
                         "pass adaptive.")
    ap.add_argument("--colors", default="project", choices=sorted(COLOR_MAPS),
                    help="palette for the built-in fallbacks (ignored when the "
                         "run's own script supplies colors). project = Other "
                         "red, HCP yellow; ovito = OVITO's lighter defaults, "
                         "where red means HCP instead.")
    ap.add_argument("--ovito-script", default=None, metavar="PATH",
                    help="use this script instead of the run's own one")
    ap.add_argument("--size", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="override the preset's image size")
    ap.add_argument("--aa", type=int, default=None, help="antialiasing samples")
    ap.add_argument("--ao", type=int, default=None, help="ambient-occlusion samples")
    ap.add_argument("--renderer", default=None, choices=["tachyon", "opengl"],
                    help="override the preset's renderer. tachyon ray-traces "
                         "(smooth, shaded); opengl is flatter and avoids the "
                         "dark speckle that saturated colors pick up.")
    ap.add_argument("--view", default="Top",
                    choices=["Top", "Bottom", "Front", "Back", "Left", "Right",
                             "Ortho", "Perspective"],
                    help="camera direction (default: Top)")
    ap.add_argument("--background", default="black", choices=sorted(BACKGROUNDS))
    ap.add_argument("--cell", dest="show_cell", action="store_true",
                    help="draw the simulation cell wireframe")
    ap.add_argument("--radius", type=float, default=1.2,
                    help="atom radius in Angstrom. 1.2 is OVITO's default: for "
                         "Cu (nearest neighbour 2.556 A) spheres nearly touch, "
                         "so the front layer occludes the ones behind and the "
                         "bulk reads as solid. Smaller values leave gaps that "
                         "show deeper layers through them, which looks speckled.")
    ap.add_argument("--shape", default="sphere", choices=["sphere", "circle"],
                    help="sphere (default) draws shaded 3D balls; circle draws "
                         "flat unshaded discs, keeping the look of a small GUI "
                         "export at print size.")
    ap.add_argument("--shading", default="flat", choices=["flat", "realistic"],
                    help="flat (default) matches the GUI: no AO or shadows. "
                         "realistic adds both, which mottles a thick slab.")
    ap.add_argument("--downscale-to", type=int, default=None, metavar="WIDTH",
                    help="resample to this width after rendering (supersampling)")
    ap.add_argument("--no-crop", dest="crop", action="store_false",
                    help="keep the full canvas instead of cropping to content")
    ap.add_argument("--per-frame-zoom", dest="shared_camera",
                    action="store_false",
                    help="zoom each frame to its own box instead of sharing one "
                         "camera. Panels will not be to scale.")
    args = ap.parse_args(argv)

    script = None
    if args.ovito_script:
        with open(args.ovito_script) as fh:
            script = fh.read()

    try:
        result = render_stills(
            args.target, timesteps=args.timesteps, out=args.out,
            preset=args.preset, colors=args.colors, cna=args.cna,
            ovito_script=script, view=args.view, background=args.background,
            show_cell=args.show_cell, radius=args.radius, shape=args.shape,
            shading=args.shading, size=args.size, aa=args.aa, ao=args.ao,
            renderer=args.renderer, downscale_to=args.downscale_to,
            crop=args.crop, shared_camera=args.shared_camera,
        )
    except RenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"run:      {result['run']}")
    print(f"coloring: {result['coloring']}")
    for w in result["warnings"]:
        print(f"warning:  {w}")
    for img in result["images"]:
        print(f"\nstep {img['timestep']}  ->  {img['path']}")
        total = img["total"]
        for name, n in img["counts"].items():
            if n:
                print(f"  {name:5} {n:8d}  {100.0 * n / total:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
