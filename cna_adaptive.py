#!/usr/bin/env python3
"""Adaptive Common Neighbor Analysis (CNA) for a LAMMPS run folder.

Reads EVERY dump in a run, recomputes the crystal-structure classification with
OVITO's *adaptive-cutoff* CNA (strain-robust, unlike the fixed-cutoff `c_cna`
baked into the dumps), and writes, into a `cna_analysis/` subfolder:

  * cna_fractions.csv  — per-frame fraction of FCC/HCP/BCC/ICO/Other atoms,
                          from BOTH adaptive CNA and the stored fixed c_cna
  * cna_fractions.png  — Other-fraction vs timestep, adaptive vs fixed
  * snapshots/         — a few rendered frames (first / peak-Other / last),
                          atoms colored by adaptive structure type (top view)

Why adaptive: a single fixed cutoff (`compute cna/atom 3.086`) is tuned to the
undeformed lattice. Under tensile load / near a crack tip the spacing changes,
so fixed CNA over-reports "Other". Adaptive CNA derives a per-atom cutoff from
the local spacing, so only genuinely disordered atoms are flagged.

Frame ordering is taken from `ITEM: TIMESTEP` INSIDE each dump — NOT from the
filename — so it is independent of the dump naming scheme (works for both
`dump.<step>.<phase>` and `dump.<phase>.<step>`).

Usage:
    python cna_adaptive.py runs/run_2026_08_12_102150_Cu_final
    python cna_adaptive.py <run_dir> --out-subdir cna_analysis --no-snapshots
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys


# OVITO 'Structure Type' encoding produced by CommonNeighborAnalysisModifier.
OVITO_STRUCT = {0: "Other", 1: "FCC", 2: "HCP", 3: "BCC", 4: "ICO"}
# The stored LAMMPS c_cna encoding (Other is 5, not 0).
LAMMPS_STRUCT = {1: "FCC", 2: "HCP", 3: "BCC", 4: "ICO", 5: "Other"}
STRUCT_ORDER = ["FCC", "HCP", "BCC", "ICO", "Other"]


def read_timestep(path):
    """Return the timestep stored inside a LAMMPS dump (ITEM: TIMESTEP), or None
    if the file is not a trajectory dump. Format-independent frame ordering."""
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if line.startswith("ITEM: TIMESTEP"):
                    return int(next(fh).strip())
    except (StopIteration, ValueError, OSError):
        return None
    return None


def phase_from_name(path):
    """Best-effort phase label parsed from the filename (e.g. 'loading',
    'prestress', 'phase2'); '' if none is present. Purely cosmetic."""
    base = os.path.basename(path)
    parts = base.split(".")[1:]  # drop the leading 'dump'
    for p in parts:
        if not p.lstrip("-").isdigit():  # the non-numeric token is the phase
            return p
    return ""


def collect_frames(run_dir):
    """Return [(timestep, phase, path), ...] for every trajectory dump, sorted
    by the timestep read from inside each file (naming-scheme independent)."""
    frames = []
    for path in glob.glob(os.path.join(run_dir, "dump.*")):
        ts = read_timestep(path)
        if ts is None:
            continue  # skip non-dump files
        frames.append((ts, phase_from_name(path), path))
    frames.sort(key=lambda t: t[0])
    return frames


def adaptive_fractions(path):
    """Run OVITO adaptive CNA on one dump; return {struct_name: fraction}."""
    from ovito.io import import_file
    from ovito.modifiers import CommonNeighborAnalysisModifier

    pipeline = import_file(path)
    pipeline.modifiers.append(
        CommonNeighborAnalysisModifier(
            mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff
        )
    )
    data = pipeline.compute()
    st = data.particles["Structure Type"][...]
    n = len(st)
    frac = {name: 0.0 for name in STRUCT_ORDER}
    if n:
        import numpy as np
        for code, name in OVITO_STRUCT.items():
            frac[name] = float(np.count_nonzero(st == code)) / n
    del pipeline, data
    return frac


def fixed_fractions(path):
    """Fractions from the stored fixed-cutoff c_cna column (LAMMPS encoding).

    OVITO auto-renames 'c_cna' -> 'Structure Type' on import, but the VALUES stay
    LAMMPS-encoded (Other=5). We read them directly without applying any modifier.
    """
    from ovito.io import import_file
    import numpy as np

    pipeline = import_file(path)
    data = pipeline.compute()
    # After import, the fixed c_cna is under 'Structure Type' (LAMMPS-encoded).
    if "Structure Type" not in data.particles:
        del pipeline, data
        return None
    st = data.particles["Structure Type"][...]
    n = len(st)
    frac = {name: 0.0 for name in STRUCT_ORDER}
    if n:
        for code, name in LAMMPS_STRUCT.items():
            frac[name] = float(np.count_nonzero(st == code)) / n
    del pipeline, data
    return frac


def render_snapshot(path, out_png):
    """Render one dump colored by ADAPTIVE structure type, top view, to a PNG."""
    from ovito.io import import_file
    from ovito.modifiers import CommonNeighborAnalysisModifier
    from ovito.vis import Viewport, TachyonRenderer

    pipeline = import_file(path)
    pipeline.modifiers.append(
        CommonNeighborAnalysisModifier(
            mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff
        )
    )
    pipeline.add_to_scene()
    try:
        data = pipeline.compute()
        data.particles.vis.radius = 0.6

        vp = Viewport(type=Viewport.Type.Top)
        vp.zoom_all()
        vp.render_image(
            filename=out_png,
            size=(1600, 1200),
            background=(0.9, 0.9, 0.9),  # light gray so white 'Other' stays visible
            renderer=TachyonRenderer(),
        )
    finally:
        pipeline.remove_from_scene()
        del pipeline


def write_csv(path, rows):
    cols = (["timestep", "phase"]
            + [f"adaptive_{s}" for s in STRUCT_ORDER]
            + [f"fixed_{s}" for s in STRUCT_ORDER])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            adap = r["adaptive"]
            fix = r["fixed"] or {s: "" for s in STRUCT_ORDER}
            w.writerow(
                [r["timestep"], r["phase"]]
                + [f"{adap[s]:.6f}" for s in STRUCT_ORDER]
                + [(f"{fix[s]:.6f}" if fix[s] != "" else "") for s in STRUCT_ORDER]
            )


def plot_fractions(path, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({exc}); skipping PNG.")
        return None

    steps = [r["timestep"] for r in rows]
    adap_other = [r["adaptive"]["Other"] * 100 for r in rows]
    has_fixed = all(r["fixed"] is not None for r in rows)
    fix_other = [r["fixed"]["Other"] * 100 for r in rows] if has_fixed else None

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(steps, adap_other, "-o", ms=3, lw=1.8, color="#1f4e9c",
            label="Adaptive CNA")
    if fix_other is not None:
        ax.plot(steps, fix_other, "--s", ms=3, lw=1.4, color="#c1440e",
                label="Fixed CNA (stored c_cna)")

    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel('"Other" atoms (%)', fontsize=13)
    ax.tick_params(axis="both", labelsize=11, direction="in", top=True, right=True)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run folder containing dump.* files")
    ap.add_argument("--out-subdir", default="cna_analysis",
                    help="output subfolder inside run_dir (default: cna_analysis)")
    ap.add_argument("--no-snapshots", action="store_true",
                    help="skip the rendered atom images (CSV + plot only)")
    ap.add_argument("--no-fixed", action="store_true",
                    help="skip the stored fixed-c_cna comparison")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.run_dir):
        ap.error(f"not a directory: {args.run_dir}")

    frames = collect_frames(args.run_dir)
    if not frames:
        ap.error(f"no LAMMPS dump files found in {args.run_dir}")

    out_dir = os.path.join(args.run_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Found {len(frames)} frames. Writing to {out_dir}")

    rows = []
    for i, (ts, phase, path) in enumerate(frames):
        adap = adaptive_fractions(path)
        fix = None if args.no_fixed else fixed_fractions(path)
        rows.append({"timestep": ts, "phase": phase, "adaptive": adap,
                     "fixed": fix, "path": path})
        print(f"  [{i+1}/{len(frames)}] step {ts:>8} {phase:<10} "
              f"adaptive Other={adap['Other']*100:5.1f}%")

    csv_path = os.path.join(out_dir, "cna_fractions.csv")
    write_csv(csv_path, rows)

    png_path = plot_fractions(os.path.join(out_dir, "cna_fractions.png"), rows)

    snap_paths = []
    if not args.no_snapshots:
        snap_dir = os.path.join(out_dir, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        # Choose 3 informative frames: first, peak-Other, last.
        peak_i = max(range(len(rows)), key=lambda k: rows[k]["adaptive"]["Other"])
        picks = sorted({0, peak_i, len(rows) - 1})
        tags = {0: "first", peak_i: "peakOther", len(rows) - 1: "last"}
        for k in picks:
            ts = rows[k]["timestep"]
            name = f"cna_{tags.get(k, 'frame')}_step{ts}.png"
            out_png = os.path.join(snap_dir, name)
            print(f"  rendering snapshot: {name}")
            render_snapshot(rows[k]["path"], out_png)
            snap_paths.append(out_png)

    print("=" * 60)
    print(f"CSV:   {csv_path}")
    if png_path:
        print(f"Plot:  {png_path}")
    for s in snap_paths:
        print(f"Image: {s}")
    print(f"Final adaptive Other: {rows[-1]['adaptive']['Other']*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
