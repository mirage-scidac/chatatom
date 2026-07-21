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







########## V2 ####################



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

# Full-fracture guard: if the loaded frame's crack-plane slab retains fewer than
# this FRACTION of the atoms the pre-loading (relax) slab held, the crack has
# torn the slab apart — atoms have evacuated the centre, so the X-span measure
# collapses to ~0 and reports a spuriously tiny Δa. In that case we treat the
# crack as fully propagated and report a_final = box X-width (the maximum
# possible extent) instead. Expressed as a FRACTION of the relax slab count so
# it auto-scales with system size (no hard-coded atom count). Chosen with margin:
# a normal lens crack keeps ~15% of atoms; full fracture drops to ~0%.
FRACTURE_SLAB_FRACTION = 0.05


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
      n_defect: count of defect (non-reference) atoms in the slab
      n_slab:   TOTAL atoms in the slab (defect + reference). Used by the
                full-fracture guard to detect when the crack has emptied the
                crack-plane slab.
      box_x_width: X-extent of the whole box (max possible crack length),
                used as a_final when the full-fracture guard trips.
    """
    x = _column(frame, "x")
    y = _column(frame, "y")
    cna = _column(frame, "c_cna").astype(int)

    ly = frame["box_hi"][1] - frame["box_lo"][1]
    y_mid = 0.5 * (frame["box_hi"][1] + frame["box_lo"][1])
    thickness = ly_slab_fraction * ly
    y_lo = y_mid - 0.5 * thickness
    y_hi = y_mid + 0.5 * thickness

    in_slab = (y >= y_lo) & (y <= y_hi)
    n_slab = int(in_slab.sum())                     # ALL atoms in the slab
    mask = (cna != reference_cna) & in_slab
    n_defect = int(mask.sum())
    box_x_width = float(frame["box_hi"][0] - frame["box_lo"][0])

    if n_defect == 0:
        return {"x_extent": 0.0, "n_defect": 0,
                "n_slab": n_slab, "box_x_width": box_x_width}

    x_selected = x[mask]

    # Percentile span: ignore the extreme trim_pct% at each end so a single
    # stray defect atom cannot inflate the measured crack width.
    hi = float(np.percentile(x_selected, 100.0 - trim_pct))
    lo = float(np.percentile(x_selected, trim_pct))

    return {
        "x_extent": hi - lo,
        "n_defect": n_defect,
        "n_slab": n_slab,
        "box_x_width": box_x_width,
    }


def compute_delta_a(
    run_folder: str,
    dump_pattern: str = "dump.*.load",
    relax_pattern: str = "dump.*.relax",
    ly_slab_fraction: float = 0.1,
    reference_cna: int = CNA_FCC,
    trim_pct: float = DEFAULT_TRIM_PCT,
) -> Dict:
    """
    Compute Δa = max(0, a_final − a_initial) across the tensile loading phase.

    a_initial is the robust (percentile-span) X-extent of the defect cloud at
    the LAST relaxation-phase frame (dump.*.relax) — the true pre-loading
    baseline. a_final is the same measure at the LAST loading-phase frame
    (dump.*.load). Δa is clamped at 0 because a crack cannot physically un-grow
    (negative values are noise).

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
    def _sorted_by_timestep(pattern: str):
        return sorted(
            glob.glob(os.path.join(run_folder, pattern)),
            key=lambda p: int(os.path.basename(p).split(".")[1]),
        )

    paths = _sorted_by_timestep(dump_pattern)
    if not paths:
        raise FileNotFoundError(
            f"No dump files matching {dump_pattern!r} in {run_folder}"
        )

    # Baseline (a_initial) = the crack state just BEFORE loading begins, i.e.
    # the LAST relaxation-phase frame (dump.*.relax). This is the true
    # pre-loading state. Using the first load frame instead would measure the
    # crack a few dump-intervals INTO loading (the dump clock is global and does
    # not reset when loading starts), by which point the crack may already be
    # opening — biasing Δa low. Fall back to the first load frame if no relax
    # dumps exist (e.g. a non-crack run).
    relax_paths = _sorted_by_timestep(relax_pattern)
    baseline_path = relax_paths[-1] if relax_paths else paths[0]

    first = _parse_dump(baseline_path)   # last relax frame (pre-loading baseline)
    last = _parse_dump(paths[-1])        # last load frame (end of loading)

    initial = _x_extent_of_defects(first, ly_slab_fraction, reference_cna, trim_pct)
    final = _x_extent_of_defects(last, ly_slab_fraction, reference_cna, trim_pct)

    a_final = final["x_extent"]

    # ── Full-fracture guard ──────────────────────────────────────────────
    # If the loaded slab has lost almost all its atoms relative to the relax
    # slab, the crack has torn the crack plane apart: the atoms evacuated the
    # centre, so the defect X-span collapses to ~0 and (wrongly) reports Δa≈0
    # for what is actually the most-propagated case. Detect this by comparing
    # slab atom counts (as a fraction, so it auto-scales with system size) and,
    # if tripped, report a_final = full box X-width (maximum possible crack).
    # NOTE: the return dict below is intentionally kept identical to the original
    # shape (no extra fields) — only the a_final VALUE changes when tripped.
    relax_slab = initial["n_slab"]
    if relax_slab > 0 and final["n_slab"] < FRACTURE_SLAB_FRACTION * relax_slab:
        a_final = final["box_x_width"]

    delta_a = max(0.0, a_final - initial["x_extent"])

    return {
        "a_initial": initial["x_extent"],
        "a_final": a_final,
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










# ################ V3 ##########################################


# # ============================================================================
# # Robust Δa (crack extension) measurement — v3  (cluster-based)
# #
# # WHY v3: v2 measured the X-span of ALL non-FCC atoms in the crack-plane slab
# # (percentile-trimmed). That fails at high temperature: thermal disorder
# # scatters isolated non-FCC atoms across the whole slab, so the span reflects
# # thermal noise, not the crack. On a real 600 K run v2 reported a=140.6 Å for a
# # crack that was actually ~24 Å (seeded length ~21.7 Å) and a spurious Δa=37.7 Å
# # for a crack that did not grow at all.
# #
# # THE FIX — measure the CRACK, not every defect atom:
# #   1. Cluster the slab's non-FCC atoms by physical connectivity (atoms within a
# #      link cutoff are the same cluster). The crack faces+tip form one big
# #      connected cluster; scattered thermal atoms are size-1 singletons.
# #   2. Pick the cluster ANCHORED on the seeded crack origin (box center x by
# #      default) — not merely the largest — so a rogue off-center defect blob
# #      cannot hijack the measurement.
# #   3. a = x-span of that crack cluster. Thermal singletons are excluded.
# #
# # GUARDS (edge cases discussed & retained):
# #   • Full fracture / empty band: if the loaded slab lost almost all its atoms
# #     relative to the relax slab, the crack tore the plane apart → report
# #     a_final = box X-width (max possible), not ~0.  [same idea as v2]
# #   • Crack ran off an X-boundary: if the crack cluster reaches the box X-edge,
# #     it has propagated to/through the boundary → also report a_final = box width.
# #   • No defects / arrested crack: a = 0 (correct — the best possible outcome).
# #
# # VALIDATION STATUS (be honest): proven on the thermal-noise case (140.6→23.7 Å,
# # Δa 37.7→0.0 on run_2026_07_20_093138). The full-fracture and off-edge guards
# # are NOT yet exercised on a real fractured run — validate on one before trusting
# # absolute Δa for publication.
# #
# # DEPENDENCIES: numpy only (grid-based neighbor search, O(N)). No scipy.
# # CNA ENCODING: raw LAMMPS dump — 1=FCC 2=HCP 3=BCC 4=ICO 5=Other. Defect mask
# # is `cna != reference_cna`, correct whether Other is 0 or 5.
# # ============================================================================
# import glob as _glob
# import os as _os
# from typing import Dict as _Dict, Optional as _Optional

# import numpy as _np

# # ── Material (edit these two lines to switch material) ──────────────────────
# # The metric no longer reads input.lammps — the lattice constant and type are
# # fixed here. Current campaigns are Copper (FCC, a0 = 3.615 Å). Change these to
# # run a different material (e.g. Al: A0=4.05; Fe: A0=2.87, LATTICE="bcc").
# A0 = 3.615          # lattice constant (Å) — Copper
# LATTICE = "fcc"     # "fcc" (d_nn=a0/√2), "bcc" (d_nn=a0·√3/2), else d_nn=a0

# # Structure codes (raw LAMMPS dump encoding).
# V3_CNA_FCC = 1
# V3_CNA_HCP = 2
# V3_CNA_BCC = 3
# V3_CNA_ICO = 4

# # Slab thickness around the crack plane (fraction of L_y).
# V3_LY_SLAB_FRACTION = 0.10

# # Connectivity link cutoff as a MULTIPLE of the 1st-nearest-neighbor distance.
# # Two defect atoms closer than link_mult * d_nn are in the same cluster.
# # 1.3 keeps a real crack connected while leaving thermal singletons alone.
# V3_LINK_MULT = 1.3

# # Full-fracture guard: loaded slab retains < this fraction of relax-slab atoms.
# V3_FRACTURE_SLAB_FRACTION = 0.05

# # Off-edge guard: crack cluster within this many Å of a box X-boundary counts
# # as "ran off the edge" → treat as full propagation.
# V3_EDGE_MARGIN_A = 4.0


# def _v3_parse_dump(path: str) -> _Dict:
#     """Parse one LAMMPS dump into {timestep, box_lo, box_hi, columns, data}."""
#     with open(path) as f:
#         lines = f.readlines()
#     timestep = int(lines[1].strip())
#     box_lo, box_hi = [], []
#     for line in lines[5:8]:
#         lo, hi = line.strip().split()[:2]
#         box_lo.append(float(lo))
#         box_hi.append(float(hi))
#     header = lines[8].strip().split()[2:]  # drop "ITEM: ATOMS"
#     rows = [list(map(float, ln.split())) for ln in lines[9:] if ln.strip()]
#     return {
#         "timestep": timestep,
#         "box_lo": _np.array(box_lo),
#         "box_hi": _np.array(box_hi),
#         "columns": header,
#         "data": _np.array(rows),
#     }


# def _v3_col(frame: _Dict, name: str) -> _np.ndarray:
#     return frame["data"][:, frame["columns"].index(name)]


# def _v3_cluster_labels(pts: _np.ndarray, box_len: _np.ndarray,
#                        link: float) -> _np.ndarray:
#     """Union-find clustering with a uniform grid for O(N) neighbor lookup.

#     pts: (N,3) coordinates. box_len: (3,) box lengths (z treated periodic).
#     Two points within `link` distance share a cluster. Returns an (N,) array
#     of integer root labels.
#     """
#     n = len(pts)
#     if n == 0:
#         return _np.array([], dtype=int)
#     if n == 1:
#         return _np.array([0], dtype=int)

#     parent = list(range(n))

#     def find(a):
#         while parent[a] != a:
#             parent[a] = parent[parent[a]]
#             a = parent[a]
#         return a

#     def union(a, b):
#         ra, rb = find(a), find(b)
#         if ra != rb:
#             parent[ra] = rb

#     # Bucket points into a grid of cell size = link. Only same/adjacent cells
#     # can contain neighbors, so each point checks a small local set.
#     cell = _np.floor(pts / link).astype(_np.int64)
#     buckets: dict = {}
#     for i in range(n):
#         buckets.setdefault(tuple(cell[i]), []).append(i)

#     lz = box_len[2]
#     link2 = link * link
#     offsets = [(dx, dy, dz)
#                for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

#     for i in range(n):
#         ci = cell[i]
#         for off in offsets:
#             key = (ci[0] + off[0], ci[1] + off[1], ci[2] + off[2])
#             for j in buckets.get(key, ()):
#                 if j <= i:
#                     continue
#                 d = pts[j] - pts[i]
#                 # z periodic (thin slab wraps); x,y are not
#                 if lz > 0:
#                     d[2] -= lz * round(d[2] / lz)
#                 if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < link2:
#                     union(i, j)

#     return _np.array([find(i) for i in range(n)], dtype=int)


# def _v3_crack_span(frame: _Dict, reference_cna: int, d_nn: float,
#                    crack_origin_x: _Optional[float]) -> _Dict:
#     """Measure the crack cluster's X-span in the crack-plane slab.

#     Returns dict:
#       x_span:       X-extent (Å) of the crack cluster (0.0 if no defects)
#       n_defect:     non-reference atoms in the slab
#       n_slab:       total atoms in the slab (for the full-fracture guard)
#       box_x_width:  box X-width (for guards)
#       touches_edge: True if the crack cluster reaches an X-boundary
#       cluster_size: atom count of the chosen crack cluster
#     """
#     x = _v3_col(frame, "x")
#     y = _v3_col(frame, "y")
#     z = _v3_col(frame, "z")
#     cna = _v3_col(frame, "c_cna").astype(int)

#     ly = frame["box_hi"][1] - frame["box_lo"][1]
#     y_mid = 0.5 * (frame["box_hi"][1] + frame["box_lo"][1])
#     half = 0.5 * V3_LY_SLAB_FRACTION * ly
#     in_slab = (y >= y_mid - half) & (y <= y_mid + half)

#     box_x_lo = float(frame["box_lo"][0])
#     box_x_hi = float(frame["box_hi"][0])
#     box_x_width = box_x_hi - box_x_lo
#     box_len = frame["box_hi"] - frame["box_lo"]

#     n_slab = int(in_slab.sum())
#     defect = (cna != reference_cna) & in_slab
#     n_defect = int(defect.sum())

#     base = {"x_span": 0.0, "n_defect": n_defect, "n_slab": n_slab,
#             "box_x_width": box_x_width, "touches_edge": False,
#             "cluster_size": 0}
#     if n_defect == 0:
#         return base

#     px = x[defect]
#     pts = _np.column_stack([px, y[defect], z[defect]])
#     link = V3_LINK_MULT * d_nn
#     labels = _v3_cluster_labels(pts, box_len, link)

#     # Group atom indices by cluster label.
#     clusters: dict = {}
#     for idx, lb in enumerate(labels):
#         clusters.setdefault(int(lb), []).append(idx)

#     if crack_origin_x is None:
#         crack_origin_x = 0.5 * (box_x_lo + box_x_hi)

#     # Choose the crack cluster: prefer clusters whose X-range CONTAINS the crack
#     # origin; among those, the largest. If none contains it, fall back to the
#     # cluster whose center is nearest the origin, breaking ties by size. This
#     # anchors on the seeded crack and rejects rogue off-center defect blobs.
#     def cluster_xrange(members):
#         cx = px[members]
#         return float(cx.min()), float(cx.max())

#     containing, others = [], []
#     for lb, members in clusters.items():
#         xmin, xmax = cluster_xrange(members)
#         entry = (lb, members, xmin, xmax, len(members))
#         if xmin <= crack_origin_x <= xmax:
#             containing.append(entry)
#         else:
#             others.append(entry)

#     if containing:
#         lb, members, xmin, xmax, size = max(containing, key=lambda e: e[4])
#     elif others:
#         # nearest-center, then size
#         def nearness(e):
#             _, _, xmin, xmax, size = e
#             center = 0.5 * (xmin + xmax)
#             return (abs(center - crack_origin_x), -size)
#         lb, members, xmin, xmax, size = min(others, key=nearness)
#     else:
#         # Defensive: n_defect > 0 should always yield ≥1 cluster, so reaching
#         # here means the label array was malformed (e.g. NaN coords, a parse
#         # quirk, or a length mismatch). Rather than crash the whole metric with
#         # a ValueError from min()/max() on an empty list — which would abort a
#         # campaign iteration — return the no-crack base result. The nonzero
#         # n_defect is preserved in the return so the anomaly is still visible.
#         return base

#     touches_edge = (xmin <= box_x_lo + V3_EDGE_MARGIN_A) or \
#                    (xmax >= box_x_hi - V3_EDGE_MARGIN_A)

#     return {"x_span": xmax - xmin, "n_defect": n_defect, "n_slab": n_slab,
#             "box_x_width": box_x_width, "touches_edge": touches_edge,
#             "cluster_size": size}


# def compute_delta_a(
#     run_folder: str,
#     a0: float = A0,
#     dump_pattern: str = "dump.*.load",
#     relax_pattern: str = "dump.*.relax",
#     reference_cna: int = V3_CNA_FCC,
#     lattice: str = LATTICE,
#     crack_origin_x: _Optional[float] = None,
# ) -> _Dict:
#     """Cluster-based Δa = max(0, a_final − a_initial).

#     Drop-in for the registry: callable as compute_delta_a(run_folder) with no
#     other args — a0 and lattice default to the module constants A0/LATTICE
#     (Copper, FCC). Pass them explicitly to override for another material.

#     Args:
#         run_folder:   LAMMPS output directory.
#         a0:           lattice constant (Å). Defaults to A0 (Cu).
#                       Sets the connectivity cutoff via the 1st-NN distance.
#         dump_pattern: loading-phase dumps (default "dump.*.load").
#         relax_pattern:pre-loading dumps; last one is the a_initial baseline.
#         reference_cna:perfect-crystal structure code (FCC=1 default).
#         lattice:      "fcc"/"bcc"/… Defaults to LATTICE (fcc).
#         crack_origin_x: x of the seeded crack center; default = box center x.

#     Returns a dict with the original keys (a_initial, a_final, delta_a,
#     n_defect_initial, n_defect_final, first_timestep, last_timestep) plus
#     cluster diagnostics (cluster_size_initial/final, guard_tripped).
#     """
#     def _sorted(pattern):
#         return sorted(
#             _glob.glob(_os.path.join(run_folder, pattern)),
#             key=lambda p: int(_os.path.basename(p).split(".")[1]),
#         )

#     load_paths = _sorted(dump_pattern)
#     if not load_paths:
#         raise FileNotFoundError(
#             f"No dump files matching {dump_pattern!r} in {run_folder}"
#         )
#     relax_paths = _sorted(relax_pattern)
#     baseline_path = relax_paths[-1] if relax_paths else load_paths[0]

#     # 1st-nearest-neighbor distance from lattice constant.
#     lat = lattice.lower()
#     if lat == "fcc":
#         d_nn = a0 / _np.sqrt(2.0)
#     elif lat == "bcc":
#         d_nn = a0 * _np.sqrt(3.0) / 2.0
#     else:
#         d_nn = a0

#     first = _v3_parse_dump(baseline_path)
#     last = _v3_parse_dump(load_paths[-1])

#     initial = _v3_crack_span(first, reference_cna, d_nn, crack_origin_x)
#     final = _v3_crack_span(last, reference_cna, d_nn, crack_origin_x)

#     a_initial = initial["x_span"]
#     a_final = final["x_span"]
#     guard = "none"

#     # Guard 1 — full fracture: loaded slab emptied out.
#     if initial["n_slab"] > 0 and \
#             final["n_slab"] < V3_FRACTURE_SLAB_FRACTION * initial["n_slab"]:
#         a_final = final["box_x_width"]
#         guard = "full_fracture"
#     # Guard 2 — crack ran off an X-boundary.
#     elif final["touches_edge"]:
#         a_final = final["box_x_width"]
#         guard = "off_edge"

#     delta_a = max(0.0, a_final - a_initial)

#     return {
#         "a_initial": a_initial,
#         "a_final": a_final,
#         "delta_a": delta_a,
#         "n_defect_initial": initial["n_defect"],
#         "n_defect_final": final["n_defect"],
#         "first_timestep": first["timestep"],
#         "last_timestep": last["timestep"],
#         # ── cluster-metric diagnostics ──
#         "cluster_size_initial": initial["cluster_size"],
#         "cluster_size_final": final["cluster_size"],
#         "guard_tripped": guard,
#     }


# if __name__ == "__main__":
#     import sys

#     run = sys.argv[1] if len(sys.argv) > 1 else "."
#     r = compute_delta_a(run)
#     print(f"Run folder: {run}   (a0={A0} Å, lattice={LATTICE})")
#     print(f"First (baseline) timestep: {r['first_timestep']}")
#     print(f"Last  (load)     timestep: {r['last_timestep']}")
#     print(f"a_initial (Å): {r['a_initial']:.3f}  "
#           f"(crack cluster {r['cluster_size_initial']} atoms, "
#           f"{r['n_defect_initial']} defects in slab)")
#     print(f"a_final   (Å): {r['a_final']:.3f}  "
#           f"(crack cluster {r['cluster_size_final']} atoms, "
#           f"{r['n_defect_final']} defects in slab)")
#     print(f"Δa         (Å): {r['delta_a']:.3f}   guard={r['guard_tripped']}")
