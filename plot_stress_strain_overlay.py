#!/usr/bin/env python3
"""Overlay two stress-strain curves (Cu and Al) on a single publication figure.

Reads the `stress_strain.csv` produced by stress_strain.py for each run and
plots stress (GPa) vs strain for both materials on one set of axes.

Usage:
    python plot_stress_strain_overlay.py
    python plot_stress_strain_overlay.py --out figs/cu_al_stress_strain
"""
from __future__ import annotations

import argparse
import csv
import os


# (label, csv_path, color) for each curve.
CURVES = [
    ("Cu", "runs/run_2026_08_12_102150_Cu_final/stress_strain.csv", "#c1440e"),  # copper
    ("Al", "runs/run_2026_08_12_114944_Al_final/stress_strain.csv", "#1f4e9c"),  # blue
]


def read_curve(path):
    """Return (strain[], stress_GPa[]) from a stress_strain.csv."""
    strain, stress = [], []
    with open(path, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            strain.append(float(row["strain"]))
            stress.append(float(row["stress_GPa"]))
    return strain, stress


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="stress_strain_overlay",
                    help="output prefix (default: stress_strain_overlay)")
    ap.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    for label, path, color in CURVES:
        if not os.path.isfile(path):
            ap.error(f"CSV not found: {path}")
        strain, stress = read_curve(path)
        peak = max(stress)
        peak_e = strain[stress.index(peak)]
        ax.plot(strain, stress, "-", color=color, lw=1.8,
                label=f"{label}  (peak {peak:.2f} GPa @ {peak_e*100:.1f}%)")

    ax.set_xlabel("Strain", fontsize=13)
    ax.set_ylabel("Stress (GPa)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11, direction="in", top=True, right=True)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.legend(fontsize=11, frameon=False)
    ax.set_xlim(left=0)
    fig.tight_layout()

    png = args.out + ".png"
    pdf = args.out + ".pdf"
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(pdf)   # vector for the manuscript
    plt.close(fig)

    print(f"Saved:\n  {png}  ({args.dpi} dpi)\n  {pdf}  (vector)")


if __name__ == "__main__":
    raise SystemExit(main())
