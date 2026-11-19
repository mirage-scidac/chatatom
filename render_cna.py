#!/usr/bin/env python3
# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE
"""Render a single LAMMPS dump colored by adaptive-cutoff CNA structure type.

Produces the standard OVITO structure-type picture: FCC green, HCP red, BCC
blue, ICO yellow, Other white — so stacking faults show as red ribbons, the
crack surface and dislocation cores as white, and the intact lattice as green.

Adaptive cutoff (not the fixed `c_cna` stored in the dump) because a single
fixed radius is tuned to the undeformed lattice: under load the spacing shifts
and fixed CNA over-reports "Other". See cna_adaptive.py for the whole-run
version that also writes fraction CSVs and comparison plots.

Usage:
    python render_cna.py runs/<run>/dump.Cu_crack.44800.dat
    python render_cna.py <dump> -o crack.png --size 1400 1400 --background black
    python render_cna.py <dump> --preset publication --background white
    python render_cna.py <dump> --view Front --cell
"""
from __future__ import annotations

import argparse
import os
import sys


# Structure-type colors are deliberately NOT set here. OVITO's own defaults
# already are the standard convention, and overriding them with hand-picked
# approximations is what made earlier renders disagree with the GUI:
#
#   Other (0.95, 0.95, 0.95)   FCC (0.40, 1.00, 0.40)   HCP (1.00, 0.40, 0.40)
#   BCC   (0.40, 0.40, 1.00)   ICO (0.95, 0.80, 0.20)
#
# Read them at runtime rather than trusting this comment:
#   [(s.id, s.name, s.color) for s in CommonNeighborAnalysisModifier().structures]

BACKGROUNDS = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "gray":  (0.9, 0.9, 0.9),
}


# Quality presets. Tachyon's 12/12 sampling defaults are preview-grade: edges
# shimmer and ambient occlusion is blotchy. The dominant factor is pixels per
# atom though — a 1200px image of a 30k-atom slab gives each atom ~8px, which
# is what makes a render look soft no matter how the sampling is set.
PRESETS = {
    "draft":       dict(size=(1000, 1000), aa=6,  ao=6,  renderer="opengl"),
    "normal":      dict(size=(1400, 1400), aa=12, ao=12, renderer="tachyon"),
    "publication": dict(size=(3200, 3200), aa=32, ao=24, renderer="tachyon"),
}


def render(dump_path, out_png, *, frame=-1, size=(1400, 1400), radius=1.2,
           view="Top", background="black", show_cell=False,
           renderer="tachyon", aa=12, ao=24, crop=True, shading="flat",
           downscale_to=None, shape="sphere"):
    """Render one frame of `dump_path`, colored by adaptive CNA, to `out_png`."""
    from ovito.io import import_file
    from ovito.modifiers import CommonNeighborAnalysisModifier
    from ovito.vis import Viewport, TachyonRenderer, OpenGLRenderer

    cna = CommonNeighborAnalysisModifier(
        mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff
    )

    pipeline = import_file(dump_path)
    pipeline.modifiers.append(cna)

    n_frames = pipeline.source.num_frames
    if frame < 0:
        frame = n_frames + frame
    if not 0 <= frame < n_frames:
        raise SystemExit(f"frame {frame} out of range (file has {n_frames})")

    pipeline.add_to_scene()
    try:
        data = pipeline.compute(frame)
        data.particles.vis.radius = radius
        # Circle draws flat, unshaded 2D discs. Sphere (OVITO's default) draws
        # shaded 3D balls whose dark rims become visible once each atom spans
        # more than a few pixels — which is why a high-resolution render looks
        # "rounder" than a small GUI export of the same scene.
        if shape == "circle":
            from ovito.vis import ParticlesVis
            data.particles.vis.shape = ParticlesVis.Shape.Circle
        if data.cell is not None:
            data.cell.vis.enabled = show_cell

        counts = {
            name: data.attributes.get(f"CommonNeighborAnalysis.counts.{key}", 0)
            for name, key in (("FCC", "FCC"), ("HCP", "HCP"), ("BCC", "BCC"),
                              ("ICO", "ICO"), ("Other", "OTHER"))
        }
        total = max(sum(counts.values()), 1)

        vp = Viewport(type=getattr(Viewport.Type, view))
        vp.zoom_all()

        if renderer == "tachyon":
            # 'flat' reproduces the OVITO GUI viewport: no ambient occlusion,
            # no shadows, uniform bright colors. AO darkens the gaps between
            # spheres, and looking down through a multi-layer slab it shades
            # the deeper layers unevenly — which reads as blotchy mottling in
            # the bulk rather than as depth. 'realistic' turns it back on for
            # single-layer or perspective views where it does help.
            realistic = shading == "realistic"
            rnd = TachyonRenderer(
                antialiasing=True, antialiasing_samples=aa,
                ambient_occlusion=realistic, ambient_occlusion_samples=ao,
                shadows=realistic, direct_light=True,
            )
        else:
            rnd = OpenGLRenderer()

        vp.render_image(
            filename=out_png,
            size=tuple(size),
            background=BACKGROUNDS.get(background, BACKGROUNDS["black"]),
            renderer=rnd,
            crop=crop,
        )

        # Render big, then resample down. Supersampling this way keeps the
        # high-sample antialiasing while returning each atom to the few pixels
        # it occupies in the OVITO GUI viewport — where per-sphere shading
        # averages out and the lattice reads as a flat uniform texture.
        # Rendering directly at the small size instead would alias the edges.
        if downscale_to:
            from PIL import Image
            im = Image.open(out_png)
            h = round(im.height * downscale_to / im.width)
            im.resize((downscale_to, h), Image.LANCZOS).save(out_png)

        return counts, total, frame, n_frames
    finally:
        pipeline.remove_from_scene()
        del pipeline


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", help="LAMMPS dump file to render")
    ap.add_argument("-o", "--out", default=None,
                    help="output PNG (default: <dump>.cna.png next to the dump)")
    ap.add_argument("--frame", type=int, default=-1,
                    help="frame index within the file; -1 = last (default)")
    ap.add_argument("--preset", default="normal", choices=sorted(PRESETS),
                    help="quality preset; sets size, sampling and renderer "
                         "(publication = 3200px, 32x AA, 24x AO)")
    ap.add_argument("--size", type=int, nargs=2, default=None,
                    metavar=("W", "H"), help="override the preset's image size")
    ap.add_argument("--aa", type=int, default=None,
                    help="override antialiasing samples")
    ap.add_argument("--ao", type=int, default=None,
                    help="override ambient-occlusion samples")
    ap.add_argument("--no-crop", dest="crop", action="store_false",
                    help="keep the full canvas instead of cropping to content")
    ap.add_argument("--radius", type=float, default=1.2,
                    help="atom radius in Angstrom. 1.2 is OVITO's own default "
                         "and what the GUI shows: for Cu (nearest-neighbour "
                         "2.556 A) spheres nearly touch, so the front layer "
                         "occludes the ones behind and the bulk reads as solid. "
                         "Smaller values leave gaps that show deeper layers "
                         "through them, which looks like speckle.")
    ap.add_argument("--view", default="Top",
                    choices=["Top", "Bottom", "Front", "Back", "Left", "Right",
                             "Ortho", "Perspective"],
                    help="camera direction (default: Top)")
    ap.add_argument("--background", default="black",
                    choices=sorted(BACKGROUNDS), help="background color")
    ap.add_argument("--cell", dest="show_cell", action="store_true",
                    help="draw the simulation cell wireframe")
    ap.add_argument("--renderer", default=None, choices=["tachyon", "opengl"],
                    help="override the preset's renderer")
    ap.add_argument("--downscale-to", type=int, default=None, metavar="WIDTH",
                    help="after rendering, resample the image to this width. "
                         "Supersamples: render at the preset's size for clean "
                         "edges, output at the size you actually want.")
    ap.add_argument("--shape", default="sphere", choices=["sphere", "circle"],
                    help="sphere (default, OVITO's own) draws shaded 3D balls; "
                         "circle draws flat unshaded discs, which keeps the "
                         "look of a low-resolution GUI export at print size.")
    ap.add_argument("--shading", default="flat", choices=["flat", "realistic"],
                    help="flat (default) matches the OVITO GUI: no ambient "
                         "occlusion or shadows. realistic adds both, which "
                         "mottles the bulk of a multi-layer slab.")
    args = ap.parse_args(argv)

    p = PRESETS[args.preset]
    size = args.size or p["size"]
    aa = args.aa if args.aa is not None else p["aa"]
    ao = args.ao if args.ao is not None else p["ao"]
    renderer = args.renderer or p["renderer"]

    out = args.out or os.path.splitext(args.dump)[0] + ".cna.png"
    counts, total, frame, n_frames = render(
        args.dump, out, frame=args.frame, size=size, radius=args.radius,
        view=args.view, background=args.background, show_cell=args.show_cell,
        renderer=renderer, aa=aa, ao=ao, crop=args.crop,
        shading=args.shading, downscale_to=args.downscale_to,
        shape=args.shape,
    )

    try:
        from PIL import Image
        with Image.open(out) as im:
            dims = f"  [{im.width}x{im.height}]"
    except Exception:
        dims = ""
    print(f"frame {frame} of {n_frames}  ->  {out}{dims}")
    for name, n in counts.items():
        print(f"  {name:5} {n:8d}  {100.0 * n / total:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
