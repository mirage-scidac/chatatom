#!/usr/bin/env python
"""Standalone test: visualize an MD run folder via OVITO + Claude Vision.

Usage:  pixi run python viz_run.py runs/<some_run_folder>/

Steps:
  1. Pick frames from the run folder (pristine + first_load + last_load,
     falls back to pristine + last if no .load dumps exist).
  2. Render each via the existing ovito_render_server.py subprocess
     (uses the run's ovito_script.py if present, else default OVITO).
  3. Send the rendered PNGs to Claude Vision via the OpenAI-compat client
     from config (same LLM the orchestrator uses).
  4. Write viz_report.md at the run folder root: relative-path image
     references + the LLM's markdown observations.

Once this works, the same logic wraps as `visualize_run(run_folder)` tool.
"""
import base64
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config
from openai import OpenAI
from tools import _send_render_request, _start_render_server_process


# ─── Frame selection ─────────────────────────────────────────────────────────

def _timestep_of(dump_path: str) -> int:
    """Extract the numeric timestep from dump filenames. Handles both:
      dump.<timestep>.<suffix>     (campaign convention)
      dump.<name>.<timestep>.dat   (some standalone conventions)
    Returns 0 if no timestep found (keeps sorting stable)."""
    base = os.path.basename(dump_path)
    parts = base.split(".")
    for token in parts:
        if token.isdigit():
            return int(token)
    return 0


def _sort_by_timestep(paths):
    return sorted(paths, key=_timestep_of)


def pick_frames(run_folder: str):
    """Return list of (label, dump_path) tuples.

    Ideal case (dumps with phase suffixes):
        pristine (first .equil, else first overall)
        first_load (first .load)
        last_load (last .load)

    Fallback (no .load suffix at all):
        pristine + last (first + last overall).
    """
    all_dumps = _sort_by_timestep(glob.glob(os.path.join(run_folder, "dump.*")))
    if not all_dumps:
        return []

    picks = []

    equil = _sort_by_timestep(glob.glob(os.path.join(run_folder, "dump.*.equil")))
    picks.append(("pristine", equil[0] if equil else all_dumps[0]))

    load = _sort_by_timestep(glob.glob(os.path.join(run_folder, "dump.*.load")))
    if load:
        picks.append(("first_load", load[0]))
        if load[-1] != load[0]:
            picks.append(("last_load", load[-1]))
    else:
        if all_dumps[-1] != picks[0][1]:
            picks.append(("last", all_dumps[-1]))

    return picks


# ─── Rendering via existing OVITO server ─────────────────────────────────────

def render_frames(picks, run_folder, ovito_script=None):
    """Render each picked frame to run_folder/viz_frames/<label>.png.

    Returns a list of dicts: {label, png_path, b64}.
    b64 is the render server's return image bytes, ready to embed in a
    vision-API request without re-reading from disk.
    """
    frames_dir = os.path.join(run_folder, "viz_frames")
    os.makedirs(frames_dir, exist_ok=True)

    proc = _start_render_server_process()
    results = []
    try:
        for label, dump_path in picks:
            resp = _send_render_request(proc, {
                "dump_file":        os.path.abspath(dump_path),
                "ovito_script":     ovito_script,
                "camera_angle":     "top",
                "render_size":      (800, 600),
                "file_frame_index": 0,
            })
            if resp.get("status") != "ok":
                print(f"[render] {label} failed: {resp.get('error')}")
                if resp.get("tb"):
                    print(resp["tb"])
                continue

            b64 = resp["image"]
            png_path = os.path.join(frames_dir, f"{label}.png")
            with open(png_path, "wb") as f:
                f.write(base64.b64decode(b64))
            results.append({"label": label, "png_path": png_path, "b64": b64})
            print(f"[render] wrote {png_path}")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    return results


# ─── Vision call ─────────────────────────────────────────────────────────────

GENERIC_PROMPT = (
    "You are looking at frames from a molecular dynamics simulation of a "
    "crystalline material, ordered chronologically (pristine → first load → "
    "last load).\n"
    "\n"
    "Write ONE integrated summary (3-6 sentences) describing what happened "
    "across the whole run. Cover:\n"
    "- Crack morphology change (opened / arrested / branched / closed)\n"
    "- Dislocation activity (nucleation, density, slip planes)\n"
    "- Stacking faults or twin nucleation\n"
    "- Phase transitions, amorphization, or notable defects\n"
    "\n"
    "STRICT RULES:\n"
    "- NO per-frame headings. NO bullet lists. NO sections.\n"
    "- Reference frames only inline when strictly needed "
    "(e.g. \"by the last frame ...\").\n"
    "- Do not restate the setup or list the frames.\n"
    "- Write it as flowing prose."
)


def send_to_vision(frames, prompt: str = GENERIC_PROMPT,
                   ovito_script: str = None) -> str:
    """Send images + prompt (+ optional ovito_script for color legend) to the
    configured LLM. Returns markdown text.

    If ovito_script is passed, its source is included so the LLM knows the
    exact color→structure mapping instead of guessing.
    """
    client = OpenAI(api_key=config.API_KEY, base_url=config.API_BASE_URL)

    content = []
    for f in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{f['b64']}"},
        })

    # Color legend first (grounds LLM's interpretation) — then the strict
    # "single-summary" instruction last so it's most recent in the prompt.
    if ovito_script:
        content.append({
            "type": "text",
            "text": (
                "The atom colors in these frames were assigned by the "
                "following OVITO script. Use its color_map to interpret "
                "each color's structural meaning (do NOT guess):\n\n"
                "```python\n" + ovito_script + "\n```"
            ),
        })
    else:
        content.append({
            "type": "text",
            "text": (
                "No custom OVITO script was used. Atoms are colored by "
                "LAMMPS particle type (NOT crystal structure). Do not "
                "interpret colors as FCC/HCP/BCC. Focus on gross spatial "
                "features (crack shape, voids)."
            ),
        })

    content.append({"type": "text", "text": prompt})

    model = getattr(config, "ORCHESTRATOR_MODEL", "") or config.DEFAULT_MODEL
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        **config.temp_kwargs(config.VISION_TEMPERATURE),
    )
    return resp.choices[0].message.content or ""


# ─── Write reports (both .md and .html) ──────────────────────────────────────

_HTML_STYLE = """
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px;
         margin: 20px auto; padding: 0 20px; color: #222; }
  h1 { color: #333; }
  table { border-collapse: collapse; margin: 20px 0; }
  td { padding: 10px; }
  img { border: 1px solid #ccc; border-radius: 4px; }
  em { color: #666; }
  .summary { line-height: 1.6; margin-top: 30px; }
  hr { border: 0; border-top: 1px solid #ddd; margin: 30px 0; }
</style>
"""


def write_report(run_folder: str, frames, observation: str):
    """Write BOTH viz_report.md (for GitHub / editors) and viz_report.html
    (for Chrome, renders natively without any extension).

    Same content in both — one file uses markdown wrapper, the other HTML.
    Returns (md_path, html_path).
    """
    title = f"Visual Analysis — {os.path.basename(run_folder.rstrip('/'))}"

    # Build side-by-side images (works verbatim inside both .md and .html)
    img_cells = []
    caption_cells = []
    for f in frames:
        rel = os.path.relpath(f["png_path"], run_folder)
        img_cells.append(
            f'<td align="center"><img src="{rel}" alt="{f["label"]}" width="320"></td>'
        )
        caption_cells.append(f'<td align="center"><em>{f["label"]}</em></td>')
    table_html = (
        "<table>\n"
        "  <tr>" + "".join(img_cells) + "</tr>\n"
        "  <tr>" + "".join(caption_cells) + "</tr>\n"
        "</table>"
    )

    # ── .md version (for editors, GitHub) ──
    md_lines = [f"# {title}", "", table_html, "", "---", "", observation.strip(), ""]
    md_path = os.path.join(run_folder, "viz_report.md")
    with open(md_path, "w") as fout:
        fout.write("\n".join(md_lines))

    # ── .html version (for Chrome — no extension needed) ──
    # Escape observation minimally so any stray < > & don't break HTML.
    import html as _html
    observation_html = _html.escape(observation.strip()).replace("\n\n", "</p><p>")
    html_body = (
        "<!DOCTYPE html>\n"
        "<html><head>\n"
        f"<meta charset='utf-8'>\n<title>{_html.escape(title)}</title>\n"
        f"{_HTML_STYLE}\n"
        "</head><body>\n"
        f"<h1>{_html.escape(title)}</h1>\n"
        f"{table_html}\n"
        "<hr>\n"
        f"<div class='summary'><p>{observation_html}</p></div>\n"
        "</body></html>\n"
    )
    html_path = os.path.join(run_folder, "viz_report.html")
    with open(html_path, "w") as fout:
        fout.write(html_body)

    return md_path, html_path


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("usage: pixi run python viz_run.py <run_folder>")
        sys.exit(1)

    run_folder = sys.argv[1].rstrip("/")
    if not os.path.isdir(run_folder):
        print(f"not a directory: {run_folder}")
        sys.exit(1)

    picks = pick_frames(run_folder)
    if not picks:
        print(f"no dumps found in {run_folder}")
        sys.exit(1)
    print(f"[viz] picked {len(picks)} frames: {[p[0] for p in picks]}")

    ovito_script = None
    ovito_script_path = os.path.join(run_folder, "ovito_script.py")
    if os.path.exists(ovito_script_path):
        with open(ovito_script_path) as f:
            ovito_script = f.read()
        print(f"[viz] using custom ovito_script.py ({len(ovito_script)} chars)")
    else:
        print("[viz] no ovito_script.py — falling back to default OVITO render")

    frames = render_frames(picks, run_folder, ovito_script)
    if not frames:
        print("no frames rendered — aborting")
        sys.exit(1)

    print(f"[viz] sending {len(frames)} frames to vision LLM "
          f"(model={getattr(config, 'ORCHESTRATOR_MODEL', '') or config.DEFAULT_MODEL})...")
    observation = send_to_vision(frames, ovito_script=ovito_script)

    md_path, html_path = write_report(run_folder, frames, observation)
    print(f"\n✓ wrote {md_path}")
    print(f"✓ wrote {html_path}   ← open this in Chrome directly")


if __name__ == "__main__":
    main()
