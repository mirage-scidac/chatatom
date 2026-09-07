#!/usr/bin/env python3
"""Dislocation Extraction Algorithm (DXA) analysis for a LAMMPS run folder.

Runs OVITO's DislocationAnalysisModifier over EVERY dump in a run and writes,
into a `dxa_analysis/` subfolder:

  * dxa_summary.csv   — per-frame dislocation density, total line length, line
                        count, line length broken down by Burgers-vector family,
                        and the DXA structure-type counts (FCC/HCP/BCC/Other)
  * dxa_segments.csv  — one row per individual dislocation line per frame
                        (id, length, true Burgers vector). Only with --segments,
                        because this file gets large.
  * dxa_density.png   — dislocation density vs timestep (top panel) and line
                        length per Burgers family vs timestep (bottom panel)
  * snapshots/        — a few rendered frames (first / peak-density / last) with
                        perfect-lattice atoms deleted so the extracted
                        dislocation lines and stacking faults are visible

Density is reported as

    rho = total_line_length / cell_volume        [Angstrom^-2]
        = that * 1e20                            [m^-2]

which is the standard definition (line length per unit volume).

DXA works directly on atomic POSITIONS — it runs its own structure
identification internally — so it needs nothing beyond `id type x y z` in the
dump. The stored `c_cna` column is ignored here; DXA's own classification is
reported instead (and uses a different encoding, see DXA_STRUCT below).

Frame ordering and phase labels are reused from cna_adaptive.py, which reads the
timestep from INSIDE each dump (`ITEM: TIMESTEP`) rather than from the filename,
so this works for both `dump.<step>.<phase>` and `dump.<phase>.<step>`.

DXA is considerably slower and more memory-hungry than CNA. Use --stride to
subsample frames on long runs.

Usage:
    python dxa_analysis.py runs/run_2026_08_24_110015
    python dxa_analysis.py <run_dir> --stride 5 --no-snapshots
    python dxa_analysis.py <run_dir> --lattice BCC --segments
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

# Frame discovery is identical to the CNA script's — reuse it rather than
# maintaining a second copy of the dump-ordering logic.
from cna_adaptive import collect_frames


# 'Structure Type' encoding produced by DislocationAnalysisModifier. NOTE this
# differs from CommonNeighborAnalysisModifier: there is no ICO, and the two
# diamond phases take slots 4 and 5.
DXA_STRUCT = {
    0: "Other",
    1: "FCC",
    2: "HCP",
    3: "BCC",
    4: "CubicDiamond",
    5: "HexagonalDiamond",
}

# Attribute suffixes OVITO uses for the per-structure atom counts. 'OTHER' is
# upper-case; the rest are not.
COUNT_KEYS = ["FCC", "HCP", "BCC", "CubicDiamond", "HexagonalDiamond", "OTHER"]

# Human-readable names for the standard FCC Burgers-vector families, used in the
# plot legend and the printed summary. Any family OVITO reports that is not
# listed here still gets collected — the CSV columns are discovered from the
# data, not from this table.
BURGERS_LABEL = {
    "1/2<110>": "Perfect 1/2<110>",
    "1/6<112>": "Shockley 1/6<112>",
    "1/6<110>": "Stair-rod 1/6<110>",
    "1/3<100>": "Hirth 1/3<100>",
    "1/3<111>": "Frank 1/3<111>",
    "1/2<111>": "1/2<111> (BCC)",
    "<100>": "<100> (BCC)",
    "<110>": "<110> (BCC)",
    "other": "Other",
}

LENGTH_PREFIX = "DislocationAnalysis.length."
ANG2_TO_M2 = 1.0e20  # 1 Angstrom^-2 = 1e20 m^-2

LATTICES = ["FCC", "HCP", "BCC", "CubicDiamond", "HexagonalDiamond"]


def _lattice_enum(name):
    """Map a --lattice string to the DislocationAnalysisModifier.Lattice enum."""
    from ovito.modifiers import DislocationAnalysisModifier
    return getattr(DislocationAnalysisModifier.Lattice, name)


def _build_modifier(lattice, line_sep, circuit_len, stretchability,
                    only_perfect):
    from ovito.modifiers import DislocationAnalysisModifier

    mod = DislocationAnalysisModifier()
    mod.input_crystal_structure = _lattice_enum(lattice)
    mod.line_point_separation = line_sep
    mod.trial_circuit_length = circuit_len
    mod.circuit_stretchability = stretchability
    mod.only_perfect_dislocations = only_perfect
    return mod


def _lines_of(dislocations):
    """Return the list of dislocation lines, tolerating the OVITO version
    difference between `.lines` and the older `.segments`."""
    for attr in ("lines", "segments"):
        lines = getattr(dislocations, attr, None)
        if lines is not None:
            return lines
    return []


def analyze_frame(path, lattice="FCC", line_sep=2.5, circuit_len=14,
                  stretchability=9, only_perfect=False, want_segments=False):
    """Run DXA on one dump. Returns a dict of per-frame results.

    Keys:
      total_length     total dislocation line length (Angstrom)
      cell_volume      simulation cell volume (Angstrom^3)
      density_A2       total_length / cell_volume (Angstrom^-2)
      density_m2       the same in m^-2
      n_lines          number of extracted dislocation lines
      lengths          {burgers_family: length_Angstrom}
      counts           {structure_name: atom_count}
      segments         [(line_id, length, bx, by, bz), ...] if want_segments
    """
    from ovito.io import import_file

    pipeline = import_file(path)
    pipeline.modifiers.append(
        _build_modifier(lattice, line_sep, circuit_len, stretchability,
                        only_perfect)
    )
    data = pipeline.compute()
    attrs = data.attributes

    total_length = float(attrs.get("DislocationAnalysis.total_line_length", 0.0))
    cell_volume = float(attrs.get("DislocationAnalysis.cell_volume", 0.0))
    density_A2 = (total_length / cell_volume) if cell_volume > 0 else 0.0

    # Discover the Burgers families present in THIS frame rather than assuming a
    # fixed set — the families OVITO emits depend on the input lattice and on
    # what it actually found.
    lengths = {
        key[len(LENGTH_PREFIX):]: float(val)
        for key, val in attrs.items()
        if key.startswith(LENGTH_PREFIX)
    }

    counts = {
        DXA_STRUCT_NAME(k): int(attrs.get(f"DislocationAnalysis.counts.{k}", 0))
        for k in COUNT_KEYS
    }

    dislocations = getattr(data, "dislocations", None)
    lines = _lines_of(dislocations) if dislocations is not None else []
    n_lines = len(lines)

    segments = []
    if want_segments:
        for line in lines:
            b = line.true_burgers_vector
            segments.append((
                int(getattr(line, "id", -1)),
                float(line.length),
                float(b[0]), float(b[1]), float(b[2]),
            ))

    del pipeline, data
    return {
        "total_length": total_length,
        "cell_volume": cell_volume,
        "density_A2": density_A2,
        "density_m2": density_A2 * ANG2_TO_M2,
        "n_lines": n_lines,
        "lengths": lengths,
        "counts": counts,
        "segments": segments,
    }


def DXA_STRUCT_NAME(count_key):
    """'OTHER' -> 'Other'; every other count key is already the display name."""
    return "Other" if count_key == "OTHER" else count_key


def render_snapshot(path, out_png, lattice="FCC", line_sep=2.5, circuit_len=14,
                    stretchability=9, only_perfect=False):
    """Render one frame with perfect-lattice atoms removed.

    Deleting the FCC (or BCC/HCP, per --lattice) bulk leaves the defect atoms —
    stacking faults, surfaces, crack faces — plus the extracted dislocation
    lines, which OVITO draws from the DislocationNetwork independently of the
    particles. That is the standard way to actually see DXA output.
    """
    from ovito.io import import_file
    from ovito.modifiers import (
        ExpressionSelectionModifier,
        DeleteSelectedModifier,
    )
    from ovito.vis import Viewport, TachyonRenderer

    pipeline = import_file(path)
    pipeline.modifiers.append(
        _build_modifier(lattice, line_sep, circuit_len, stretchability,
                        only_perfect)
    )
    # Structure Type 1 is the reference lattice for every DXA input structure.
    pipeline.modifiers.append(
        ExpressionSelectionModifier(expression="StructureType==1")
    )
    pipeline.modifiers.append(DeleteSelectedModifier())

    pipeline.add_to_scene()
    try:
        data = pipeline.compute()
        try:
            data.particles.vis.radius = 0.6
        except Exception:
            pass  # no particles left after the delete — lines still render

        vp = Viewport(type=Viewport.Type.Perspective)
        vp.camera_dir = (-1, -1, -1)
        vp.zoom_all()
        vp.render_image(
            filename=out_png,
            size=(1600, 1200),
            background=(0.9, 0.9, 0.9),
            renderer=TachyonRenderer(),
        )
    finally:
        pipeline.remove_from_scene()
        del pipeline


def family_columns(rows):
    """Union of Burgers families seen across all frames, ordered with the known
    FCC/BCC families first (in BURGERS_LABEL order) and anything unexpected
    appended alphabetically."""
    seen = set()
    for r in rows:
        seen.update(r["dxa"]["lengths"].keys())
    known = [f for f in BURGERS_LABEL if f in seen]
    extra = sorted(seen - set(known))
    return known + extra


def write_summary_csv(path, rows, families):
    cols = (
        ["timestep", "phase", "n_lines", "total_line_length_A",
         "cell_volume_A3", "density_A-2", "density_m-2"]
        + [f"len_{f}_A" for f in families]
        + [f"count_{DXA_STRUCT_NAME(k)}" for k in COUNT_KEYS]
    )
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            d = r["dxa"]
            w.writerow(
                [r["timestep"], r["phase"], d["n_lines"],
                 f"{d['total_length']:.4f}", f"{d['cell_volume']:.4f}",
                 f"{d['density_A2']:.6e}", f"{d['density_m2']:.6e}"]
                + [f"{d['lengths'].get(f, 0.0):.4f}" for f in families]
                + [d["counts"][DXA_STRUCT_NAME(k)] for k in COUNT_KEYS]
            )


def write_segments_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestep", "phase", "line_id", "length_A",
                    "burgers_x", "burgers_y", "burgers_z"])
        for r in rows:
            for (lid, length, bx, by, bz) in r["dxa"]["segments"]:
                w.writerow([r["timestep"], r["phase"], lid,
                            f"{length:.4f}",
                            f"{bx:.6f}", f"{by:.6f}", f"{bz:.6f}"])


def plot_density(path, rows, families):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({exc}); skipping PNG.")
        return None

    steps = [r["timestep"] for r in rows]
    density = [r["dxa"]["density_m2"] for r in rows]

    # Only plot families that are actually non-zero somewhere — otherwise the
    # legend fills up with flat lines at zero.
    active = [f for f in families
              if any(r["dxa"]["lengths"].get(f, 0.0) > 0 for r in rows)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)

    ax1.plot(steps, density, "-o", ms=3, lw=1.8, color="#1f4e9c")
    ax1.set_ylabel(r"Dislocation density (m$^{-2}$)", fontsize=13)
    ax1.tick_params(axis="both", labelsize=11, direction="in",
                    top=True, right=True)
    ax1.grid(True, alpha=0.25, lw=0.6)

    cmap = plt.get_cmap("tab10")
    for i, fam in enumerate(active):
        ax2.plot(steps, [r["dxa"]["lengths"].get(fam, 0.0) for r in rows],
                 "-", lw=1.4, color=cmap(i % 10),
                 label=BURGERS_LABEL.get(fam, fam))
    ax2.set_xlabel("Timestep", fontsize=13)
    ax2.set_ylabel("Line length (Å)", fontsize=13)
    ax2.tick_params(axis="both", labelsize=11, direction="in",
                    top=True, right=True)
    ax2.grid(True, alpha=0.25, lw=0.6)
    if active:
        ax2.legend(fontsize=10, frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run folder containing dump.* files")
    ap.add_argument("--lattice", choices=LATTICES, default="FCC",
                    help="input crystal structure for DXA (default: FCC — "
                         "correct for Cu, Al, Ni, Au, Ag)")
    ap.add_argument("--out-subdir", default="dxa_analysis",
                    help="output subfolder inside run_dir (default: dxa_analysis)")
    ap.add_argument("--stride", type=int, default=1,
                    help="analyze every Nth frame (default: 1 = all). DXA is "
                         "slow; use this on long runs.")
    ap.add_argument("--segments", action="store_true",
                    help="also write dxa_segments.csv, one row per dislocation "
                         "line per frame (can get large)")
    ap.add_argument("--no-snapshots", action="store_true",
                    help="skip the rendered images (CSV + plot only)")
    ap.add_argument("--line-point-separation", type=float, default=2.5,
                    help="DXA line coarsening, in interatomic spacings "
                         "(default: 2.5)")
    ap.add_argument("--trial-circuit-length", type=int, default=14,
                    help="max Burgers circuit length in atom-to-atom steps "
                         "(default: 14)")
    ap.add_argument("--circuit-stretchability", type=int, default=9,
                    help="how far a Burgers circuit may stretch (default: 9; "
                         "raise it with --only-perfect)")
    ap.add_argument("--only-perfect", action="store_true",
                    help="extract only perfect dislocations, no partials")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.run_dir):
        ap.error(f"not a directory: {args.run_dir}")
    if args.stride < 1:
        ap.error("--stride must be >= 1")

    frames = collect_frames(args.run_dir)
    if not frames:
        ap.error(f"no LAMMPS dump files found in {args.run_dir}")
    if args.stride > 1:
        kept = frames[::args.stride]
        print(f"Stride {args.stride}: analyzing {len(kept)} of {len(frames)} frames")
        frames = kept

    out_dir = os.path.join(args.run_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Found {len(frames)} frames. Lattice={args.lattice}. "
          f"Writing to {out_dir}")

    dxa_kwargs = dict(
        lattice=args.lattice,
        line_sep=args.line_point_separation,
        circuit_len=args.trial_circuit_length,
        stretchability=args.circuit_stretchability,
        only_perfect=args.only_perfect,
    )

    rows = []
    for i, (ts, phase, path) in enumerate(frames):
        d = analyze_frame(path, want_segments=args.segments, **dxa_kwargs)
        rows.append({"timestep": ts, "phase": phase, "dxa": d, "path": path})
        print(f"  [{i+1}/{len(frames)}] step {ts:>8} {phase:<10} "
              f"rho={d['density_m2']:.3e} m^-2  "
              f"lines={d['n_lines']:>4}  L={d['total_length']:9.1f} A")

    families = family_columns(rows)

    csv_path = os.path.join(out_dir, "dxa_summary.csv")
    write_summary_csv(csv_path, rows, families)

    seg_path = None
    if args.segments:
        seg_path = os.path.join(out_dir, "dxa_segments.csv")
        write_segments_csv(seg_path, rows)

    png_path = plot_density(os.path.join(out_dir, "dxa_density.png"),
                            rows, families)

    snap_paths = []
    if not args.no_snapshots:
        snap_dir = os.path.join(out_dir, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        peak_i = max(range(len(rows)), key=lambda k: rows[k]["dxa"]["density_m2"])
        picks = sorted({0, peak_i, len(rows) - 1})
        tags = {0: "first", peak_i: "peakDensity", len(rows) - 1: "last"}
        for k in picks:
            ts = rows[k]["timestep"]
            name = f"dxa_{tags.get(k, 'frame')}_step{ts}.png"
            out_png = os.path.join(snap_dir, name)
            print(f"  rendering snapshot: {name}")
            try:
                render_snapshot(rows[k]["path"], out_png, **dxa_kwargs)
                snap_paths.append(out_png)
            except Exception as exc:  # noqa: BLE001
                print(f"  [snapshot] failed for step {ts}: {exc}")

    peak = max(rows, key=lambda r: r["dxa"]["density_m2"])
    final = rows[-1]

    print("=" * 68)
    print(f"CSV:   {csv_path}")
    if seg_path:
        print(f"CSV:   {seg_path}")
    if png_path:
        print(f"Plot:  {png_path}")
    for s in snap_paths:
        print(f"Image: {s}")
    print("-" * 68)
    print(f"Peak density:  {peak['dxa']['density_m2']:.3e} m^-2 "
          f"at step {peak['timestep']} ({peak['dxa']['n_lines']} lines)")
    print(f"Final density: {final['dxa']['density_m2']:.3e} m^-2 "
          f"at step {final['timestep']} ({final['dxa']['n_lines']} lines)")

    final_lengths = final["dxa"]["lengths"]
    nonzero = sorted(((v, k) for k, v in final_lengths.items() if v > 0),
                     reverse=True)
    if nonzero:
        print("Final frame, line length by Burgers family:")
        for v, k in nonzero:
            print(f"    {BURGERS_LABEL.get(k, k):<22} {v:9.1f} A")
    else:
        print("No dislocations found in the final frame.")
    print("=" * 68)


if __name__ == "__main__":
    sys.exit(main())
