#!/usr/bin/env python3
# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE
"""Search OpenKIM for potentials and let the user pick one.

Uses the official kim-query package (openkim/kim-query), which wraps
    https://query.openkim.org/api/get_available_models
Install once:
    pixi run pip install git+https://github.com/openkim/kim-query.git

get_available_models(species, species_logic=["and"], model_interface=["any"],
                     potential_type=["any"], simulator_name=["any"])
-> a flat list of model-ID strings (no other metadata).

Usage:
    python kim_search.py Si C --type meam --interface sm --simulator LAMMPS
    python kim_search.py Al
    python kim_search.py Ni Al --exact
"""
import argparse
import sys

from kim_query import get_available_models


def choose(models):
    """Numbered menu; return the chosen model ID, or None."""
    print(f"\nFound {len(models)} model(s):\n")
    for i, mid in enumerate(models, 1):
        print(f"  [{i:>2}] {mid}")
    while True:
        raw = input(f"\nSelect 1-{len(models)} (q to quit): ").strip()
        if raw.lower() in ("q", "quit", ""):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        print("  invalid choice.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("species", nargs="+", help="chemical symbol(s), e.g. Al  or  Si C")
    ap.add_argument("--type", default="any", help="eam, meam, adp, sw, ... (default: any)")
    ap.add_argument("--interface", default="any", choices=["mo", "sm", "any"],
                    help="mo=Portable, sm=Simulator (default: any)")
    ap.add_argument("--simulator", default="any", help='e.g. "LAMMPS" (default: any)')
    ap.add_argument("--exact", action="store_true",
                    help="only models supporting EXACTLY these species")
    args = ap.parse_args(argv)

    logic = "exact" if args.exact else "and"
    models = get_available_models(
        args.species, [logic], [args.interface], [args.type], [args.simulator],
    )
    if not models:
        print("No models found. Try --interface any, widen --type, or drop --exact.")
        return 1

    picked = choose(models)
    if picked is None:
        print("No selection made.")
        return 1
    print(f"\nSELECTED={picked}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
