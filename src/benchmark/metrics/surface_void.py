"""Surface-mesh crack metric — cavity area / volume / area-per-volume.

Per-region surface-mesh approach: build a surface mesh with OVITO's
ConstructSurfaceModifier, pick the crack CAVITY region (largest empty region, or
the one nearest a given void center), and read that region's own volume and
surface area — then form the area-per-volume ratio.

Per frame it reports (for the chosen cavity region):
  - cavity_volume        (Å³)
  - cavity_surface_area  (Å²)
  - area_per_volume      = cavity_surface_area / cavity_volume  (1/Å)
  - pca_length + principal extents + aspect ratios (crack shape)

For the run it reports the baseline (last relax frame), final (last load frame),
and deltas. Exports `compute(run_folder) -> dict` (metric-registry contract).

Dependencies: numpy + ovito only. A tortuosity/skeleton analysis is intentionally
omitted to avoid the extra scipy/skimage/trimesh dependencies.
"""
import glob
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

from ovito.io import import_file
from ovito.modifiers import ConstructSurfaceModifier


# ── Parameters ──────────────────────────────────────────────────────────────
SURFACE_RADIUS = 2  # probe-sphere radius (~1 NN spacing for Cu)
SMOOTHING_LEVEL = 2
# Optional seeded-crack center [x,y,z] to disambiguate multiple voids. None ->
# pick the largest empty region (default).
VOID_CENTER: Optional[np.ndarray] = None


# ── Frame selection ─────────────────────────────────────────────────────────
def _sorted_by_timestep(run_folder: str, pattern: str):
    return sorted(
        glob.glob(os.path.join(run_folder, pattern)),
        key=lambda p: int(os.path.basename(p).split(".")[1]),
    )


# ── Region/property helpers (trimmed to confirmed OVITO 3.15 names) ──────────
def _property_names(container) -> list:
    try:
        return list(container.keys())
    except Exception:
        return []


def _get_region_property(surface, names) -> Optional[np.ndarray]:
    have = _property_names(surface.regions)
    for name in names:
        if name in have:
            return np.asarray(surface.regions[name])
    return None


def _get_face_region_ids(surface) -> Optional[np.ndarray]:
    have = _property_names(surface.faces)
    for name in ("Region", "region", "Region ID", "RegionId",
                 "Adjacent Region", "AdjacentRegion"):
        if name in have:
            return np.asarray(surface.faces[name])
    return None


def _get_face_vertex_indices(surface, face_index) -> Optional[np.ndarray]:
    getf = getattr(surface, "get_face_vertices", None)
    if getf is not None:
        try:
            return np.asarray(getf(int(face_index)), dtype=int)
        except Exception:
            pass
    topo = getattr(surface, "topology", None)
    if topo is not None:
        for m in ("get_face_vertices", "get_vertices_of_face", "face_vertices"):
            fn = getattr(topo, m, None)
            if fn is not None:
                try:
                    return np.asarray(fn(int(face_index)), dtype=int)
                except Exception:
                    pass
    return None


def _extract_region_mesh(surface, region_id) -> Optional[np.ndarray]:
    """Return the vertex positions of the faces belonging to `region_id`.

    (Only vertices are needed here, for the PCA shape metrics.)
    """
    all_vertices = np.asarray(surface.vertices["Position"])
    face_region_ids = _get_face_region_ids(surface)
    if face_region_ids is None:
        return None
    face_ids = np.where(face_region_ids == region_id)[0]
    if len(face_ids) == 0:
        return None
    vert_idx = set()
    for fid in face_ids:
        inds = _get_face_vertex_indices(surface, int(fid))
        if inds is None:
            continue
        for vi in inds:
            vert_idx.add(int(vi))
    if not vert_idx:
        return None
    return all_vertices[np.array(sorted(vert_idx))]


# ── PCA crack metrics ────────────────────────────────────────────────────────
def _pca_crack_metrics(vertices: Optional[np.ndarray]) -> Dict:
    nan = float("nan")
    if vertices is None or len(vertices) < 3:
        return {"pca_length": nan, "extent_1": nan, "extent_2": nan,
                "extent_3": nan, "aspect_12": nan, "aspect_13": nan}
    center = vertices.mean(axis=0)
    X = vertices - center
    cov = X.T @ X / len(X)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    projections = X @ eigvecs
    extents = projections.max(axis=0) - projections.min(axis=0)
    e1, e2, e3 = (float(extents[0]), float(extents[1]), float(extents[2]))
    return {
        "pca_length": e1, "extent_1": e1, "extent_2": e2, "extent_3": e3,
        "aspect_12": e1 / e2 if e2 > 0 else nan,
        "aspect_13": e1 / e3 if e3 > 0 else nan,
    }


# ── Cavity-region selection ──────────────────────────────────────────────────
def _choose_cavity_region(surface, void_center=None
                          ) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
    """Return (region_id, volumes, areas) for the crack cavity, or None.

    void_center=None -> largest empty (Filled==0) region by volume.
    Otherwise -> empty region whose vertex centroid is nearest void_center.
    """
    volumes = _get_region_property(surface, ["Volume", "volume"])
    areas = _get_region_property(surface, ["Surface Area", "SurfaceArea", "Area"])
    filled = _get_region_property(surface, ["Filled", "filled"])
    if volumes is None:
        return None

    if filled is not None:
        candidates = np.where(filled == 0)[0]
    else:
        candidates = np.arange(surface.regions.count)
    if len(candidates) == 0:
        return None  # no empty region → no crack

    if void_center is None:
        rid = int(candidates[np.argmax(volumes[candidates])])
        return rid, volumes, areas

    best_rid, best_dist = None, np.inf
    for rid in candidates:
        verts = _extract_region_mesh(surface, int(rid))
        if verts is None:
            continue
        dist = float(np.linalg.norm(verts.mean(axis=0) - void_center))
        if dist < best_dist:
            best_dist, best_rid = dist, int(rid)
    if best_rid is None:
        return None
    return best_rid, volumes, areas


# ── Per-frame analysis (per-region cavity metrics) ───────────────────────────
def _analyze_frame(dump_path: str, radius: float = SURFACE_RADIUS,
                   void_center=None) -> Dict:
    """Cavity metrics for one frame.

    Distinguishes two very different outcomes that must NOT look alike:
      • Mesh built OK but no void region  → intact slab → genuine ZERO (return `zero`).
      • Mesh could NOT be built at all    → we failed to MEASURE → RAISE.

    Returning a fake 0.0 on a mesh-build failure would masquerade as "no crack"
    even when a crack exists (e.g. a bad probe radius on a thin slab), silently
    turning a measurement failure into a false "crack didn't grow" result. So we
    raise instead — compute_metric() catches it and records a FAILED trial, which
    the campaign treats differently from a true zero-crack trial.
    """
    nan = float("nan")
    zero = {
        "cavity_volume": 0.0, "cavity_surface_area": 0.0,
        "area_per_volume": 0.0, "pca_length": nan,
        "extent_1": nan, "extent_2": nan, "extent_3": nan,
        "aspect_12": nan, "aspect_13": nan, "num_voids": 0,
    }
    frame = os.path.basename(dump_path)
    pipe = import_file(dump_path)
    pipe.modifiers.append(ConstructSurfaceModifier(
        radius=radius, smoothing_level=SMOOTHING_LEVEL, identify_regions=True))
    try:
        data = pipe.compute()
    except RuntimeError as e:
        # Delaunay / alpha-shape construction failed (e.g. radius too large for a
        # thin slab, degenerate/empty frame). This is a MEASUREMENT failure, not a
        # zero-size crack — surface it instead of masking it as 0.0.
        raise RuntimeError(
            f"surface mesh construction failed on {frame} (radius={radius}): {e}"
        )
    if "surface" not in data.surfaces:
        raise RuntimeError(
            f"no surface mesh produced on {frame} (radius={radius})"
        )
    surface = data.surfaces["surface"]

    filled = _get_region_property(surface, ["Filled", "filled"])
    n_voids = int(np.sum(filled == 0)) if filled is not None else 0

    chosen = _choose_cavity_region(surface, void_center)
    if chosen is None:
        out = dict(zero)
        out["num_voids"] = n_voids
        return out
    region_id, volumes, areas = chosen

    cavity_volume = float(volumes[region_id]) if volumes is not None else 0.0
    cavity_area = float(areas[region_id]) if areas is not None else 0.0
    verts = _extract_region_mesh(surface, region_id)
    pca = _pca_crack_metrics(verts)

    return {
        "cavity_volume": cavity_volume,
        "cavity_surface_area": cavity_area,
        "area_per_volume": (cavity_area / cavity_volume) if cavity_volume > 0 else 0.0,
        "pca_length": pca["pca_length"],
        "extent_1": pca["extent_1"], "extent_2": pca["extent_2"],
        "extent_3": pca["extent_3"],
        "aspect_12": pca["aspect_12"], "aspect_13": pca["aspect_13"],
        "num_voids": n_voids,
    }


# ── Public entry point (registry contract) ──────────────────────────────────
def compute(run_folder: str) -> Dict:
    """Cavity area/volume/area-per-volume across the loading phase.

    Baseline = last dump.*.relax frame; final = last dump.*.load frame. Reports
    baseline, final, and delta for cavity volume, cavity surface area, and
    area-per-volume, plus the final frame's PCA length / extents / aspect ratios
    and void count.

    Raises FileNotFoundError if no load dumps exist.
    """
    # Debug marker to stderr (stdout is reserved for the JSON result when this
    # module is run as a subprocess by campaign_measure_metrics).
    print(f"[surface_void] compute() RUNNING on {os.path.basename(run_folder)}",
          file=sys.stderr, flush=True)
    load = _sorted_by_timestep(run_folder, "dump.*.load")
    if not load:
        raise FileNotFoundError(f"No 'dump.*.load' files in {run_folder}")
    relax = _sorted_by_timestep(run_folder, "dump.*.relax")
    baseline_path = relax[-1] if relax else load[0]

    ini = _analyze_frame(baseline_path, void_center=VOID_CENTER)
    fin = _analyze_frame(load[-1], void_center=VOID_CENTER)

    def _delta(key):
        return max(0.0, fin[key] - ini[key])

    return {
        # cavity volume
        "cavity_volume_initial": ini["cavity_volume"],
        "cavity_volume_final":   fin["cavity_volume"],
        "delta_cavity_volume":   _delta("cavity_volume"),
        # cavity surface area
        "cavity_surface_area_initial": ini["cavity_surface_area"],
        "cavity_surface_area_final":   fin["cavity_surface_area"],
        "delta_cavity_surface_area":   _delta("cavity_surface_area"),
        # area per volume (ratio of cavity surface area to cavity volume)
        "area_per_volume_initial": ini["area_per_volume"],
        "area_per_volume_final":   fin["area_per_volume"],
        "delta_area_per_volume":   _delta("area_per_volume"),
        # crack shape (final frame)
        "pca_length_final": fin["pca_length"],
        "extent_1_final":   fin["extent_1"],
        "extent_2_final":   fin["extent_2"],
        "extent_3_final":   fin["extent_3"],
        "aspect_12_final":  fin["aspect_12"],
        "aspect_13_final":  fin["aspect_13"],
        # diagnostics
        "num_voids_final": fin["num_voids"],
        "radius": SURFACE_RADIUS,
    }


# ── Fatigue entry point: peak cavity over the whole loading phase ────────────
def compute_peak(run_folder: str) -> Dict:
    """Peak cavity GROWTH across ALL loading frames (fatigue objective).

    Unlike compute() — which is a 2-frame endpoint delta suited to MONOTONIC
    loading — this scans EVERY dump.*.load frame and reports the frame where the
    cavity surface area is largest. In cyclic (fatigue) loading the last frame is
    near-zero strain and the crack has closed back, so an endpoint delta under-
    reports; the max over frames captures the worst-case crack opening wherever
    in the cycle it occurs.

    The objective is baseline-subtracted so a PRE-EXISTING cavity (the built-in
    notch, already present at the last relax frame) does not inflate the score:

        peak_delta_cavity_surface_area = max(0, peak_area - baseline_area)

    where peak_area = max cavity area over all load frames, baseline_area = cavity
    area at the last relax frame. This isolates how much the worst-case opening
    grew BEYOND the starting cavity.

    Raises FileNotFoundError if no load dumps exist. A per-frame mesh-build
    failure propagates (via _analyze_frame) so a bad frame is a FAILED trial, not
    a fake zero — same contract as compute().
    """
    print(f"[surface_void] compute_peak() RUNNING on {os.path.basename(run_folder)}",
          file=sys.stderr, flush=True)
    load = _sorted_by_timestep(run_folder, "dump.*.load")
    if not load:
        raise FileNotFoundError(f"No 'dump.*.load' files in {run_folder}")
    relax = _sorted_by_timestep(run_folder, "dump.*.relax")
    baseline_path = relax[-1] if relax else load[0]

    base = _analyze_frame(baseline_path, void_center=VOID_CENTER)
    baseline_area = base["cavity_surface_area"]

    # Scan every load frame; track the one with the largest ABSOLUTE cavity area.
    peak_idx = 0
    peak_frame = _analyze_frame(load[0], void_center=VOID_CENTER)
    peak_area = peak_frame["cavity_surface_area"]
    for i in range(1, len(load)):
        f = _analyze_frame(load[i], void_center=VOID_CENTER)
        if f["cavity_surface_area"] > peak_area:
            peak_area, peak_frame, peak_idx = f["cavity_surface_area"], f, i

    peak_path = load[peak_idx]
    peak_timestep = int(os.path.basename(peak_path).split(".")[1])
    final_frame = _analyze_frame(load[-1], void_center=VOID_CENTER) \
        if peak_idx != len(load) - 1 else peak_frame

    return {
        # objective (minimize): worst-case crack growth over baseline notch
        "peak_delta_cavity_surface_area": max(0.0, peak_area - baseline_area),
        # the pieces that make it up
        "baseline_cavity_surface_area": baseline_area,
        "peak_cavity_surface_area":     peak_area,
        "final_cavity_surface_area":    final_frame["cavity_surface_area"],
        # WHERE the peak occurred
        "peak_timestep":    peak_timestep,
        "peak_frame_index": peak_idx,
        "peak_frame":       os.path.basename(peak_path),
        "n_load_frames":    len(load),
        # peak-frame shape / volume diagnostics
        "peak_cavity_volume": peak_frame["cavity_volume"],
        "peak_pca_length":    peak_frame["pca_length"],
        "peak_num_voids":     peak_frame["num_voids"],
        # gap between peak and endpoint = how much the crack closed back (fatigue)
        "closure_area": max(0.0, peak_area - final_frame["cavity_surface_area"]),
        "radius": SURFACE_RADIUS,
    }


# ── Standalone surface-mesh PNG (reuses the campaign OVITO render server) ─────
def _render_surface_png(dump_path: str, out_png: str,
                        radius: float = SURFACE_RADIUS,
                        camera_angle: str = "top",
                        with_surface: bool = True) -> Optional[str]:
    """Render one dump frame and save a PNG. Reuses ovito_render_server.py (the
    same subprocess the campaign viz report uses) via tools._send_render_request.

    with_surface=True  → inject the SAME ConstructSurfaceModifier the metric uses,
                         so the image shows the mesh + identified void regions.
    with_surface=False → atoms only, colored by the run's OWN ovito_script.py
                         (CNA structure type: green=FCC bulk, red=Other/defect,
                         etc.), so red atoms mark the crack/defect region and you
                         can judge whether the probe `radius` wraps them correctly.

    Returns the saved path, or None on failure.
    """
    # src/ is two levels up (src/benchmark/metrics/ -> src/); put it on the path
    # so `import tools` (the render-server helpers) resolves.
    src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import base64
    from tools import _start_render_server_process, _send_render_request

    # Injected into the render server.
    #   with_surface=True  → attach the metric's ConstructSurfaceModifier so the
    #                        image shows the mesh + identified void regions.
    #   with_surface=False → atoms only. Reuse the run's own ovito_script.py (the
    #                        CNA remap+color modifier the simulation generated:
    #                        green=FCC, red=Other/defect, …) so the coloring matches
    #                        the rest of the pipeline. Falls back to a no-op script
    #                        (default OVITO color) if the run has no ovito_script.py.
    if with_surface:
        ovito_script = (
            "from ovito.modifiers import ConstructSurfaceModifier\n"
            f"pipeline.modifiers.append(ConstructSurfaceModifier("
            f"radius={radius}, smoothing_level={SMOOTHING_LEVEL}, identify_regions=True))\n"
        )
    else:
        run_script = os.path.join(os.path.dirname(os.path.abspath(dump_path)),
                                  "ovito_script.py")
        if os.path.isfile(run_script):
            with open(run_script) as f:
                ovito_script = f.read()
        else:
            ovito_script = "# atoms only — no ovito_script.py in run; default coloring\npass\n"

    proc = _start_render_server_process()
    try:
        resp = _send_render_request(proc, {
            "dump_file":    os.path.abspath(dump_path),
            "ovito_script": ovito_script,
            "camera_angle": camera_angle,
            "render_size":  [800, 600],
        })
    finally:
        try:
            _send_render_request(proc, {"type": "shutdown"})
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass

    if resp.get("status") != "ok" or "image" not in resp:
        print(f"[surface_void] render failed: {resp.get('error', resp)}", file=sys.stderr)
        return None
    with open(out_png, "wb") as f:
        f.write(base64.b64decode(resp["image"]))
    return out_png


if __name__ == "__main__":
    import json
    run = sys.argv[1] if len(sys.argv) > 1 else "."
    # --peak → fatigue objective (max cavity over ALL load frames).
    # default → monotonic objective (endpoint delta).
    if "--peak" in sys.argv:
        result = compute_peak(run)
    else:
        result = compute(run)
    print(json.dumps(result, indent=2))

    # Drop PNGs of the baseline + final frames into the run folder so the numbers
    # can be eyeballed. For each frame we render TWO images side by side in intent:
    #   *_atoms.png   → plain atoms (no surface)     — the raw structure
    #   *_surface.png → same frame WITH the mesh     — what the metric measured
    # Comparing the two shows whether the chosen probe `radius` wraps the atoms
    # correctly (mesh hugs the atoms, voids appear only where atoms are truly gone).
    # Pass "--no-png" to skip rendering.
    if "--no-png" not in sys.argv:
        load = _sorted_by_timestep(run, "dump.*.load")
        relax = _sorted_by_timestep(run, "dump.*.relax")
        frames = []
        if relax:
            frames.append(("baseline", relax[-1]))
        if load:
            frames.append(("final", load[-1]))
        for label, dump in frames:
            a = _render_surface_png(dump, os.path.join(run, f"surface_void_{label}_atoms.png"),
                                    with_surface=False)
            if a:
                print(f"[surface_void] {label} atoms PNG   -> {a}", file=sys.stderr)
            s = _render_surface_png(dump, os.path.join(run, f"surface_void_{label}_surface.png"),
                                    with_surface=True)
            if s:
                print(f"[surface_void] {label} surface PNG -> {s}", file=sys.stderr)
