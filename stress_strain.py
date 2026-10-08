#!/usr/bin/env python3
"""Standalone stress-vs-strain curve from a LAMMPS run.

Reads a run's ``log.lammps``, extracts the tensile-loading phase, and computes
the engineering stress-strain response for a uniaxial pull:

    strain  e_yy = (Ly - Ly0) / Ly0            (dimensionless)
    stress  s_yy = -Pyy                          (tension positive)

LAMMPS reports pressure, so the tensile stress is the negative of the pressure
component along the loading axis. Pressure in "metal" units is bar; we convert
to GPa (1 GPa = 1e4 bar).

The loading phase is identified as the thermo block whose box length along the
loading axis actually changes (a `fix deform` block). Ly0 is the box length at
the first row of that block, so strain is measured from the start of loading,
not from the start of the whole run.

Outputs:
  * a PNG of the stress-strain curve (if matplotlib is available)
  * a CSV of (strain, stress_GPa, step, time, temp)
  * a printed summary: Young's modulus (low-strain fit), peak stress, and the
    strain at peak stress

Usage:
    python stress_strain.py runs/run_2026_08_05_053006
    python stress_strain.py runs/run_2026_08_05_053006 --axis y
    python stress_strain.py <run_dir> --axis y --fit-max-strain 0.01 --no-plot
"""
from __future__ import annotations

import argparse
import csv
import os
import sys


# Axis -> (pressure column name, box-length column name) in thermo output.
_AXIS_COLUMNS = {
    "x": ("pxx", "lx"),
    "y": ("pyy", "ly"),
    "z": ("pzz", "lz"),
}

BAR_TO_GPA = 1.0e-4  # 1 GPa = 1e4 bar

# Mirrors dxa_analysis.PLOT_STYLE so the two figures sit together in one paper.
# Duplicated rather than imported on purpose: importing dxa_analysis would pull
# in ovito transitively, and this script must stay runnable without it.
#
# Sizes are deliberately absent. They are passed explicitly at each call below,
# where they are set larger than DXA's because this is one 7-inch panel against
# DXA's two ~4.4-inch ones, so the same point size reads smaller once the figure
# is scaled to a column width.
PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 1.1,
    # Embed real glyphs instead of Type 3 outlines, which several publishers
    # reject outright in submitted vector figures. No effect on PNG output.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# 150 is fine on screen but soft in print; most journals want >= 300 for raster
# line art. Matches the dpi dxa_analysis saves at.
PLOT_DPI = 600


def parse_thermo_blocks(log_path):
    """Parse every thermo table in a LAMMPS log into a list of blocks.

    A block is a dict {column_name (lowercased): [values...]}. A new block
    begins at each header line that starts with "Step". Rows are collected
    until a non-numeric line (e.g. "Loop time...") ends the table.

    Uses errors="replace" so a stray non-UTF-8 byte in the log (KIM banner
    glyphs, etc.) degrades to a placeholder instead of crashing the read.
    """
    blocks = []
    current_cols = None
    current = None

    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue

            # Header row of a thermo table.
            if stripped.split()[0] == "Step":
                if current is not None and current[next(iter(current))]:
                    blocks.append(current)
                current_cols = [c.lower() for c in stripped.split()]
                current = {c: [] for c in current_cols}
                continue

            if current is None:
                continue

            parts = stripped.split()
            # A data row is all-numeric and has the right number of columns.
            if len(parts) != len(current_cols):
                # End of this table.
                if current[current_cols[0]]:
                    blocks.append(current)
                current = None
                current_cols = None
                continue
            try:
                values = [float(p) for p in parts]
            except ValueError:
                if current[current_cols[0]]:
                    blocks.append(current)
                current = None
                current_cols = None
                continue
            for col, val in zip(current_cols, values):
                current[col].append(val)

    if current is not None and current[current_cols[0]]:
        blocks.append(current)

    return blocks


def pick_loading_block(blocks, length_col):
    """Return the thermo block in which the box length along the loading axis
    changes the most (fractionally) — that is the `fix deform` phase.

    Raises if no block varies the length (no deformation found).
    """
    best = None
    best_span = 0.0
    for blk in blocks:
        lengths = blk.get(length_col)
        if not lengths or len(lengths) < 2:
            continue
        lo, hi = min(lengths), max(lengths)
        if lo <= 0:
            continue
        span = (hi - lo) / lo
        if span > best_span:
            best_span = span
            best = blk
    if best is None or best_span < 1e-9:
        raise ValueError(
            f"No thermo block shows a changing box length ({length_col}); "
            "cannot identify the loading phase."
        )
    return best


def compute_stress_strain(block, press_col, length_col):
    """Compute engineering strain and tensile stress (GPa) for a block.

    strain = (L - L0) / L0 ; L0 = length at the first row of the block.
    stress = -pressure * BAR_TO_GPA (tension positive).
    """
    lengths = block[length_col]
    press = block[press_col]
    L0 = lengths[0]
    strain = [(L - L0) / L0 for L in lengths]
    stress = [-p * BAR_TO_GPA for p in press]
    return strain, stress


def youngs_modulus(strain, stress, fit_max_strain):
    """Least-squares slope of stress vs strain over the low-strain (elastic)
    region, using only points with strain <= fit_max_strain. Returns GPa.
    """
    xs, ys = [], []
    for e, s in zip(strain, stress):
        if 0.0 <= e <= fit_max_strain:
            xs.append(e)
            ys.append(s)
    if len(xs) < 2:
        return None
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        return None
    return (n * sxy - sx * sy) / denom


def write_csv(path, block, strain, stress, length_col):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strain", "stress_GPa", "step", "time", "temp", length_col])
        steps = block.get("step", [None] * len(strain))
        times = block.get("time", [None] * len(strain))
        temps = block.get("temp", [None] * len(strain))
        lengths = block[length_col]
        for i in range(len(strain)):
            w.writerow([
                f"{strain[i]:.6e}",
                f"{stress[i]:.6e}",
                steps[i], times[i], temps[i], lengths[i],
            ])


def plot(path, strain, stress, modulus, fit_max_strain):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({exc}); skipping PNG.")
        return None

    # rc_context rather than rcParams.update: the Analyst agent imports this
    # module in-process, so a global style change here would follow every other
    # figure drawn afterwards in the same server.
    with plt.rc_context(PLOT_STYLE):
        # Colors match dxa_analysis.plot_density so the two figures sit together
        # in a manuscript without looking like they came from different papers.
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(strain, stress, "-", color="#14305e", lw=2.2,
                label="stress-strain")

        # Dashed overlay of the fitted elastic slope, kept off the figure but
        # not deleted. The slope is still computed and still appears in the
        # text summary; only the line and its "E = ... GPa" label are
        # suppressed, since under laterally constrained loading that slope is
        # not Young's modulus. Uncomment to restore.
        # if modulus is not None:
        #     xs = [e for e in strain if 0.0 <= e <= fit_max_strain]
        #     if xs:
        #         x_end = max(xs)
        #         ax.plot([0, x_end], [0, modulus * x_end], "--",
        #                 color="#8f1d21", lw=1.9,
        #                 label=f"elastic fit E = {modulus:.1f} GPa")

        # Larger than the 13/11 used in dxa_analysis, on purpose: this is a
        # single 7-inch panel where DXA draws two ~4.4-inch ones, so the same
        # point size reads smaller here once scaled to a column width.
        ax.set_xlabel("Strain", fontsize=16)
        ax.set_ylabel("Stress (GPa)", fontsize=16)
        # No title. The PNG goes to a journal as-is, where the caption carries
        # the description and an in-figure heading is redundant at best. Which
        # run the figure came from is identified by the folder it is written
        # into, and by the "Run:" line of the terminal summary.
        ax.legend(fontsize=13, loc="lower right", frameon=False)

        # Same conventions as dxa_analysis.plot_density: black frame on all four
        # sides, ticks turned inward with minor ticks shown. tick_params is
        # called twice because which="both" cannot take separate major and
        # minor lengths. Unlike DXA, this panel also carries a light grid; see
        # below.
        ax.minorticks_on()
        for side in ax.spines.values():
            side.set_linewidth(1.1)
            side.set_color("black")
        ax.tick_params(which="major", direction="in", length=5.5, width=1.1,
                       top=True, right=True, color="black", labelsize=14)
        ax.tick_params(which="minor", direction="in", length=3.0, width=0.8,
                       top=True, right=True, color="black")

        # Light grid, drawn behind the curve so it never breaks the data line.
        # Major gridlines only: at the minor-tick density set above a minor grid
        # reads as hatching rather than as a reference. A solid light gray is
        # used instead of a dashed darker one because dashes at this spacing
        # alias into a dotted texture once the 600-dpi figure is downscaled by
        # the typesetter.
        ax.set_axisbelow(True)
        ax.grid(True, which="major", color="#D9D9D9", lw=0.6)

        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run folder containing log.lammps")
    ap.add_argument("--axis", choices=["x", "y", "z"], default="y",
                    help="loading axis (default: y)")
    ap.add_argument("--log", default=None,
                    help="path to log file (default: <run_dir>/log.lammps)")
    ap.add_argument("--fit-max-strain", type=float, default=0.01,
                    help="max strain for the elastic (Young's modulus) fit "
                         "(default: 0.01)")
    ap.add_argument("--out-prefix", default=None,
                    help="output file prefix (default: <run_dir>/stress_strain)")
    ap.add_argument("--no-plot", action="store_true", help="skip the PNG")
    args = ap.parse_args(argv)

    log_path = args.log or os.path.join(args.run_dir, "log.lammps")
    if not os.path.isfile(log_path):
        ap.error(f"log file not found: {log_path}")

    press_col, length_col = _AXIS_COLUMNS[args.axis]

    blocks = parse_thermo_blocks(log_path)
    if not blocks:
        ap.error(f"no thermo tables found in {log_path}")

    block = pick_loading_block(blocks, length_col)

    for needed in (press_col, length_col):
        if needed not in block:
            ap.error(
                f"thermo output is missing column '{needed}'. "
                f"Add it to thermo_style (available: {', '.join(block)})."
            )

    strain, stress = compute_stress_strain(block, press_col, length_col)
    modulus = youngs_modulus(strain, stress, args.fit_max_strain)

    peak_stress = max(stress)
    peak_idx = stress.index(peak_stress)
    peak_strain = strain[peak_idx]

    prefix = args.out_prefix or os.path.join(args.run_dir, "stress_strain")
    csv_path = prefix + ".csv"
    write_csv(csv_path, block, strain, stress, length_col)

    png_path = None
    if not args.no_plot:
        png_path = plot(prefix + ".png", strain, stress, modulus,
                        args.fit_max_strain)

    print("=" * 60)
    print(f"Run:            {args.run_dir}")
    print(f"Loading axis:   {args.axis}  (stress = -{press_col}, strain from {length_col})")
    print(f"Data points:    {len(strain)}  (loading phase)")
    print(f"L0 ({length_col}):        {block[length_col][0]:.4f}")
    print(f"Max strain:     {max(strain):.4f}")
    if modulus is not None:
        print(f"Young's modulus: {modulus:.1f} GPa  (fit strain <= {args.fit_max_strain})")
    else:
        print("Young's modulus: n/a (not enough low-strain points)")
    print(f"Peak stress:    {peak_stress:.3f} GPa  at strain {peak_strain:.4f}")
    print("-" * 60)
    print(f"CSV:  {csv_path}")
    if png_path:
        print(f"PNG:  {png_path}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
