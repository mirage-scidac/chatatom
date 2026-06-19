"""
Δa (crack extension) measurement from LAMMPS dump files.

Loads the first and last frames of the tensile loading phase, filters
non-FCC atoms in a thin Y-slab centered on the crack plane (y = L_y/2),
and reports the X-extent change.
"""
import glob
import os
from typing import Dict, List, Optional

import numpy as np

# LAMMPS CNA structure type codes
CNA_OTHER = 0   # also commonly 5 in some LAMMPS versions
CNA_FCC = 1
CNA_HCP = 2
CNA_BCC = 3
CNA_ICO = 4


def _parse_dump(path: str) -> Dict:
    """Parse a single LAMMPS dump file into {timestep, box, columns, data}."""
    with open(path) as f:
        lines = f.readlines()

    timestep = int(lines[1].strip())

    # ITEM: BOX BOUNDS ...
    box_lo = []
    box_hi = []
    for line in lines[5:8]:
        lo, hi = line.strip().split()
        box_lo.append(float(lo))
        box_hi.append(float(hi))

    # ITEM: ATOMS id type x y z ...
    header = lines[8].strip().split()[2:]  # drop "ITEM: ATOMS"

    rows = [list(map(float, line.split())) for line in lines[9:]]
    data = np.array(rows)  # shape (N, len(header))

    return {
        "timestep": timestep,
        "box_lo": np.array(box_lo),
        "box_hi": np.array(box_hi),
        "columns": header,
        "data": data,
    }


def _column(frame: Dict, name: str) -> np.ndarray:
    """Get a column from a parsed dump by name (e.g. 'x', 'c_cna')."""
    idx = frame["columns"].index(name)
    return frame["data"][:, idx]


def _x_extent_of_defects(
    frame: Dict, ly_slab_fraction: float, reference_cna: int
) -> Dict:
    """
    Filter non-reference atoms in a Y-slab centered on L_y/2, return X-extent.

    Atoms whose CNA value differs from `reference_cna` (the perfect-crystal
    structure of the material) are flagged as defects — these surround the
    crack tip and faces.

    Returns dict with:
      x_extent: max(x) - min(x) of selected atoms, or 0.0 if none
      n_defect: count of selected atoms
    """
    x = _column(frame, "x")
    y = _column(frame, "y")
    cna = _column(frame, "c_cna").astype(int)

    ly = frame["box_hi"][1] - frame["box_lo"][1]
    y_mid = 0.5 * (frame["box_hi"][1] + frame["box_lo"][1])
    thickness = ly_slab_fraction * ly
    y_lo = y_mid - 0.5 * thickness
    y_hi = y_mid + 0.5 * thickness

    mask = (cna != reference_cna) & (y >= y_lo) & (y <= y_hi)
    n_defect = int(mask.sum())

    if n_defect == 0:
        return {"x_extent": 0.0, "n_defect": 0}

    x_selected = x[mask]
    return {
        "x_extent": float(x_selected.max() - x_selected.min()),
        "n_defect": n_defect,
    }


def compute_delta_a(
    run_folder: str,
    dump_pattern: str = "dump.*.load",
    ly_slab_fraction: float = 0.1,
    reference_cna: int = CNA_FCC,
) -> Dict:
    """
    Compute Δa = a_final − a_initial across the tensile loading phase.

    Args:
        run_folder: LAMMPS run output directory.
        dump_pattern: glob pattern selecting the loading-phase dumps
                      (default matches Griffith template: "dump.*.load").
        ly_slab_fraction: thickness of Y-slab around crack plane,
                          as a fraction of L_y. Default 0.1 = 10%.
        reference_cna: CNA code for the material's perfect-crystal structure.
                       Atoms with a different code are flagged as defects.
                         CNA_FCC=1 (Al, Cu, Ni, Au, Ag, Pt) — DEFAULT
                         CNA_HCP=2 (Ti, Mg, Zn)
                         CNA_BCC=3 (Fe, Cr, W, Mo)
                         CNA_ICO=4

    Returns:
        {
          "a_initial": float,        # x-extent at first load frame
          "a_final": float,          # x-extent at last load frame
          "delta_a": float,          # a_final - a_initial
          "n_defect_initial": int,   # non-reference atom count in slab @ first
          "n_defect_final": int,     # count @ last
          "first_timestep": int,
          "last_timestep": int,
        }

    Raises:
        FileNotFoundError if no dump files match dump_pattern.
    """
    paths = sorted(
        glob.glob(os.path.join(run_folder, dump_pattern)),
        key=lambda p: int(os.path.basename(p).split(".")[1]),
    )
    if not paths:
        raise FileNotFoundError(
            f"No dump files matching {dump_pattern!r} in {run_folder}"
        )

    first = _parse_dump(paths[0])
    last = _parse_dump(paths[-1])

    initial = _x_extent_of_defects(first, ly_slab_fraction, reference_cna)
    final = _x_extent_of_defects(last, ly_slab_fraction, reference_cna)

    return {
        "a_initial": initial["x_extent"],
        "a_final": final["x_extent"],
        "delta_a": final["x_extent"] - initial["x_extent"],
        "n_defect_initial": initial["n_defect"],
        "n_defect_final": final["n_defect"],
        "first_timestep": first["timestep"],
        "last_timestep": last["timestep"],
    }


if __name__ == "__main__":
    import sys

    run = sys.argv[1] if len(sys.argv) > 1 else "runs/run_2026_05_27_190601"
    result = compute_delta_a(run)
    print(f"Run folder: {run}")
    print(f"First frame timestep: {result['first_timestep']}")
    print(f"Last frame timestep:  {result['last_timestep']}")
    print(f"a_initial (Å): {result['a_initial']:.3f}  "
          f"({result['n_defect_initial']} defect atoms)")
    print(f"a_final   (Å): {result['a_final']:.3f}  "
          f"({result['n_defect_final']} defect atoms)")
    print(f"Δa         (Å): {result['delta_a']:+.3f}")
