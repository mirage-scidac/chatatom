# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE

"""Run the OVITO DXA frame loop in a child process.

OVITO must not run inside the web server process. Two independent reasons, both
observed in this repo:

1. `import_file()` leaks OS semaphores — see the comment above
   `_start_render_server_process` in src/tools.py, which pushes rendering into a
   subprocess for exactly this reason. A DXA run calls it once per frame, 53
   times on a typical run, so it is the worst case for that leak.

2. Calling into OVITO from a thread other than the one that initialized it
   segfaults outright:

       EXC_BAD_ACCESS (SIGSEGV), KERN_INVALID_ADDRESS at 0x60
       Ovito::PythonLongRunningOperation::PythonLongRunningOperation(bool)

   That is what killed the server on 2026-09-25: the Analyst's DXA tool is a
   sync LangChain tool, LangGraph offloads sync work to a thread-pool worker,
   and the frame loop therefore ran on a thread that had not initialized OVITO.
   The event loop itself was healthy (main thread parked in uvloop's
   run_forever) — the process simply died mid-loop, leaving an empty
   dxa_analysis/ directory as the tell.

   Measured: `import ovito` on the main thread plus `analyze_frame()` on a
   worker exits 139; doing both in one single-threaded process is fine. Hence
   this script — one process, one thread, started fresh per call and gone when
   it returns. A crash in here can no longer take the server with it.

Protocol: newline-delimited JSON on stdout, read by `_dxa_run_worker` in
src/graph/nodes.py. OVITO writes chatter to stderr, which the parent merges into
stdout and skips as unparseable, so no event may be written anywhere but stdout.

    {"type": "frames",   "n": 53}
    {"type": "progress", "i": 1, "ts": 1000, "rho": 0.0}
    {"type": "skip",     "file": "dump.2000.loading", "error": "..."}
    {"type": "done",     "rows": 53}
    {"type": "error",    "message": "..."}

Exit codes are advisory only — the parent trusts the CSV's existence, not the
code, because OVITO can abort the process after a successful write.
"""
import argparse
import json
import os
import sys


def emit(obj):
    """Write one protocol event. Flush every time: the parent renders progress
    live, and a block-buffered pipe would deliver the whole run at once."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="DXA frame loop, out of process.")
    ap.add_argument("run_folder")
    ap.add_argument("--lattice", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--png", required=True)
    args = ap.parse_args()

    # Append, never insert — root scripts must not shadow a same-named module
    # under src/. Same rule as _dxa_import in nodes.py.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    import dxa_analysis as dxa

    frames = dxa.collect_frames(args.run_folder)
    emit({"type": "frames", "n": len(frames)})
    if not frames:
        emit({"type": "error", "message": "no-dumps"})
        return 2

    rows = []
    for i, (ts, phase, path) in enumerate(frames):
        try:
            d = dxa.analyze_frame(path, lattice=args.lattice)
        except ImportError as e:
            emit({"type": "error", "message": f"ovito-missing: {e}"})
            return 3
        except Exception as e:  # noqa: BLE001
            # First frame failing means the dumps are unusable; a later frame
            # failing is worth skipping past, but the agent must be told.
            if not rows:
                emit({"type": "error",
                      "message": f"first-frame: {os.path.basename(path)}: {e}"})
                return 4
            emit({"type": "skip", "file": os.path.basename(path), "error": str(e)})
            continue
        rows.append({"timestep": ts, "phase": phase, "dxa": d, "path": path})
        emit({"type": "progress", "i": i + 1, "ts": ts, "rho": d["density_m2"]})

    if not rows:
        emit({"type": "error", "message": "no-usable-frames"})
        return 5

    families = dxa.family_columns(rows)
    try:
        dxa.write_summary_csv(args.csv, rows, families)
    except OSError as e:
        emit({"type": "error", "message": f"csv-write: {e}"})
        return 6

    # Plot here rather than in the parent: this process already holds the rows,
    # and matplotlib is harmless alongside OVITO. The parent still redraws from
    # the CSV if this fails, so a plot failure is not fatal.
    try:
        dxa.plot_density(args.png, rows, families)
    except Exception as e:  # noqa: BLE001
        emit({"type": "skip", "file": os.path.basename(args.png),
              "error": f"plot: {e}"})

    emit({"type": "done", "rows": len(rows)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
