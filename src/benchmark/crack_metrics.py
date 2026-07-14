# """
# Δa (crack extension) measurement from LAMMPS dump files.

# Loads the first and last frames of the tensile loading phase, filters
# non-FCC atoms in a thin Y-slab centered on the crack plane (y = L_y/2),
# and reports the X-extent change.
# """
# import glob
# import os
# from typing import Dict, List, Optional

# import numpy as np

# # LAMMPS CNA structure type codes
# CNA_OTHER = 0   # also commonly 5 in some LAMMPS versions
# CNA_FCC = 1
# CNA_HCP = 2
# CNA_BCC = 3
# CNA_ICO = 4


# def _parse_dump(path: str) -> Dict:
#     """Parse a single LAMMPS dump file into {timestep, box, columns, data}."""
#     with open(path) as f:
#         lines = f.readlines()

#     timestep = int(lines[1].strip())

#     # ITEM: BOX BOUNDS ...
#     box_lo = []
#     box_hi = []
#     for line in lines[5:8]:
#         lo, hi = line.strip().split()
#         box_lo.append(float(lo))
#         box_hi.append(float(hi))

#     # ITEM: ATOMS id type x y z ...
#     header = lines[8].strip().split()[2:]  # drop "ITEM: ATOMS"

#     rows = [list(map(float, line.split())) for line in lines[9:]]
#     data = np.array(rows)  # shape (N, len(header))

#     return {
#         "timestep": timestep,
#         "box_lo": np.array(box_lo),
#         "box_hi": np.array(box_hi),
#         "columns": header,
#         "data": data,
#     }


# def _column(frame: Dict, name: str) -> np.ndarray:
#     """Get a column from a parsed dump by name (e.g. 'x', 'c_cna')."""
#     idx = frame["columns"].index(name)
#     return frame["data"][:, idx]


# def _x_extent_of_defects(
#     frame: Dict, ly_slab_fraction: float, reference_cna: int
# ) -> Dict:
#     """
#     Filter non-reference atoms in a Y-slab centered on L_y/2, return X-extent.

#     Atoms whose CNA value differs from `reference_cna` (the perfect-crystal
#     structure of the material) are flagged as defects — these surround the
#     crack tip and faces.

#     Returns dict with:
#       x_extent: max(x) - min(x) of selected atoms, or 0.0 if none
#       n_defect: count of selected atoms
#     """
#     x = _column(frame, "x")
#     y = _column(frame, "y")
#     cna = _column(frame, "c_cna").astype(int)

#     ly = frame["box_hi"][1] - frame["box_lo"][1]
#     y_mid = 0.5 * (frame["box_hi"][1] + frame["box_lo"][1])
#     thickness = ly_slab_fraction * ly
#     y_lo = y_mid - 0.5 * thickness
#     y_hi = y_mid + 0.5 * thickness

#     mask = (cna != reference_cna) & (y >= y_lo) & (y <= y_hi)
#     n_defect = int(mask.sum())

#     if n_defect == 0:
#         return {"x_extent": 0.0, "n_defect": 0}

#     x_selected = x[mask]
#     return {
#         "x_extent": float(x_selected.max() - x_selected.min()),
#         "n_defect": n_defect,
#     }


# def compute_delta_a(
#     run_folder: str,
#     dump_pattern: str = "dump.*.load",
#     ly_slab_fraction: float = 0.1,
#     reference_cna: int = CNA_FCC,
# ) -> Dict:
#     """
#     Compute Δa = a_final − a_initial across the tensile loading phase.

#     Args:
#         run_folder: LAMMPS run output directory.
#         dump_pattern: glob pattern selecting the loading-phase dumps
#                       (default matches Griffith template: "dump.*.load").
#         ly_slab_fraction: thickness of Y-slab around crack plane,
#                           as a fraction of L_y. Default 0.1 = 10%.
#         reference_cna: CNA code for the material's perfect-crystal structure.
#                        Atoms with a different code are flagged as defects.
#                          CNA_FCC=1 (Al, Cu, Ni, Au, Ag, Pt) — DEFAULT
#                          CNA_HCP=2 (Ti, Mg, Zn)
#                          CNA_BCC=3 (Fe, Cr, W, Mo)
#                          CNA_ICO=4

#     Returns:
#         {
#           "a_initial": float,        # x-extent at first load frame
#           "a_final": float,          # x-extent at last load frame
#           "delta_a": float,          # a_final - a_initial
#           "n_defect_initial": int,   # non-reference atom count in slab @ first
#           "n_defect_final": int,     # count @ last
#           "first_timestep": int,
#           "last_timestep": int,
#         }

#     Raises:
#         FileNotFoundError if no dump files match dump_pattern.
#     """
#     paths = sorted(
#         glob.glob(os.path.join(run_folder, dump_pattern)),
#         key=lambda p: int(os.path.basename(p).split(".")[1]),
#     )
#     if not paths:
#         raise FileNotFoundError(
#             f"No dump files matching {dump_pattern!r} in {run_folder}"
#         )

#     first = _parse_dump(paths[0])
#     last = _parse_dump(paths[-1])

#     initial = _x_extent_of_defects(first, ly_slab_fraction, reference_cna)
#     final = _x_extent_of_defects(last, ly_slab_fraction, reference_cna)

#     return {
#         "a_initial": initial["x_extent"],
#         "a_final": final["x_extent"],
#         "delta_a": final["x_extent"] - initial["x_extent"],
#         "n_defect_initial": initial["n_defect"],
#         "n_defect_final": final["n_defect"],
#         "first_timestep": first["timestep"],
#         "last_timestep": last["timestep"],
#     }


# if __name__ == "__main__":
#     import sys

#     run = sys.argv[1] if len(sys.argv) > 1 else "runs/run_2026_05_27_190601"
#     result = compute_delta_a(run)
#     print(f"Run folder: {run}")
#     print(f"First frame timestep: {result['first_timestep']}")
#     print(f"Last frame timestep:  {result['last_timestep']}")
#     print(f"a_initial (Å): {result['a_initial']:.3f}  "
#           f"({result['n_defect_initial']} defect atoms)")
#     print(f"a_final   (Å): {result['a_final']:.3f}  "
#           f"({result['n_defect_final']} defect atoms)")
#     print(f"Δa         (Å): {result['delta_a']:+.3f}")


# ============================================================================
# Robust Δa (crack extension) measurement — v2
#
# Same idea as the commented version above (compare defect-cloud X-extent
# between the first and last loading-phase dump), with two robustness fixes:
#
#   1. Percentile span instead of max−min.
#      max(x)−min(x) is corrupted by a single stray defect atom far from the
#      crack (thermal flicker, a lone dislocation elsewhere in the slab).
#      We use percentile(98) − percentile(2) so the extreme ~2% at each end
#      are ignored — one outlier atom can no longer fake a wider crack.
#
#   2. Δa is clamped at 0.
#      A physical crack cannot un-grow. Negative Δa is thermal noise; leaving
#      it negative lets an optimizer chase noise (treating -0.2 as "better"
#      than 0). max(0, ...) makes a fully-arrested crack read as 0 (the best
#      possible), and removes the noise-chasing failure mode.
#
# CNA ENCODING NOTE:
#   Raw LAMMPS `compute cna/atom` dumps use: 1=FCC 2=HCP 3=BCC 4=ICO 5=Other.
#   (OVITO remaps Other→0 on import, but this module reads the RAW dump files,
#   so Other is 5 here.) The defect mask filters by `cna != FCC`, so it is
#   correct regardless of whether Other is 0 or 5 — anything that isn't the
#   perfect-crystal structure counts as a defect.
# ============================================================================
import glob
import os
from typing import Dict

import numpy as np

# LAMMPS compute cna/atom structure codes (raw dump encoding).
CNA_FCC = 1
CNA_HCP = 2
CNA_BCC = 3
CNA_ICO = 4

# Extreme fraction (percent) trimmed from EACH end when measuring the defect
# X-span. 2 → use percentile(2)..percentile(98). Light outlier rejection that
# keeps essentially the whole real crack while dropping lone stray atoms.
DEFAULT_TRIM_PCT = 2.0


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

    rows = [list(map(float, line.split())) for line in lines[9:] if line.strip()]
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
    frame: Dict,
    ly_slab_fraction: float,
    reference_cna: int,
    trim_pct: float = DEFAULT_TRIM_PCT,
) -> Dict:
    """
    Filter non-reference atoms in a Y-slab centered on L_y/2, return a robust
    X-extent (percentile span, not max−min).

    Atoms whose CNA value differs from `reference_cna` (the perfect-crystal
    structure of the material) are flagged as defects — these surround the
    crack tip and faces.

    Returns dict with:
      x_extent: percentile(100-trim) − percentile(trim) of defect X, or 0.0
                if none. Robust to lone stray defect atoms.
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

    # Percentile span: ignore the extreme trim_pct% at each end so a single
    # stray defect atom cannot inflate the measured crack width.
    hi = float(np.percentile(x_selected, 100.0 - trim_pct))
    lo = float(np.percentile(x_selected, trim_pct))

    return {
        "x_extent": hi - lo,
        "n_defect": n_defect,
    }


def compute_delta_a(
    run_folder: str,
    dump_pattern: str = "dump.*.load",
    ly_slab_fraction: float = 0.1,
    reference_cna: int = CNA_FCC,
    trim_pct: float = DEFAULT_TRIM_PCT,
) -> Dict:
    """
    Compute Δa = max(0, a_final − a_initial) across the tensile loading phase.

    a_initial / a_final are the robust (percentile-span) X-extents of the
    defect cloud at the first and last loading-phase dump. Δa is clamped at 0
    because a crack cannot physically un-grow (negative values are noise).

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
        trim_pct: percent trimmed from each end of the defect X-distribution
                  before measuring the span (outlier rejection). Default 2.

    Returns:
        {
          "a_initial": float,        # robust X-extent at first load frame
          "a_final": float,          # robust X-extent at last load frame
          "delta_a": float,          # max(0, a_final - a_initial)
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

    initial = _x_extent_of_defects(first, ly_slab_fraction, reference_cna, trim_pct)
    final = _x_extent_of_defects(last, ly_slab_fraction, reference_cna, trim_pct)

    delta_a = max(0.0, final["x_extent"] - initial["x_extent"])

    return {
        "a_initial": initial["x_extent"],
        "a_final": final["x_extent"],
        "delta_a": delta_a,
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
    print(f"Δa         (Å): {result['delta_a']:.3f}")
