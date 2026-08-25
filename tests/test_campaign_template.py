#!/usr/bin/env python3
"""Tests for the campaign template contract.

A campaign generates its LAMMPS script once and then fills the same script for
every trial. These tests pin the two properties that make that safe:

  1. Filling changes ONLY the lines inside the CAMPAIGN VARIABLES block, so
     trials stay byte-comparable and the objective metric stays meaningful.
  2. A script that cannot be filled is rejected at proposal time, not halfway
     through a 20-iteration run.
"""
import difflib
import json
import os

import pytest

# The sweep logic is stdlib-only by design, so these tests run without the
# langchain/langgraph stack installed.
from orchestrator import campaign_template as tpl

# The on-disk helpers live in campaign_tools, which pulls in langchain and
# langgraph. Skip just those tests when the tool stack is absent rather than
# losing the whole module.
try:
    from orchestrator import campaign_tools as ct
except ImportError:  # pragma: no cover - environment-dependent
    ct = None

needs_tools = pytest.mark.skipif(ct is None, reason="langchain/langgraph not installed")


TEMPLATE = """# Al Griffith crack campaign template
# ==== CAMPAIGN VARIABLES (auto-filled per iteration — do not move) ====
variable a0                equal 4.05
variable temperature       equal 300        # K
variable strain_rate       equal 0.005
variable crack_length_frac equal 0.3
variable crack_width_a0    equal 4
# ==== END CAMPAIGN VARIABLES ====

units metal
boundary p p p
lattice fcc ${a0}
region box block 0 60 0 30 0 3
create_box 1 box
create_atoms 1 box

velocity all create ${temperature} 12345 mom yes rot yes dist gaussian

variable ymid       equal ly/2
variable xbound_low equal lx/2-0.5*${crack_length_frac}*lx
variable ylo_crack  equal ly/2-0.5*${crack_width_a0}*${a0}

fix load all deform 1 y erate ${strain_rate} units box remap x
run 15000
"""

KNOBS = {
    "strain_rate":       [1e-3, 2e-2],
    "temperature":       [100, 700],
    "crack_length_frac": [0.1, 0.6],
    "crack_width_a0":    [2, 8],
}


# ── parse_variable_block ─────────────────────────────────────────────────────

def test_parse_reads_every_declaration():
    values, (begin, end) = tpl.parse_variable_block(TEMPLATE)
    assert values == {
        "a0": 4.05,
        "temperature": 300.0,
        "strain_rate": 0.005,
        "crack_length_frac": 0.3,
        "crack_width_a0": 4.0,
    }
    lines = TEMPLATE.split("\n")
    assert lines[begin].startswith(tpl.VAR_BLOCK_BEGIN)
    assert lines[end].startswith(tpl.VAR_BLOCK_END)


def test_parse_rejects_missing_block():
    with pytest.raises(ValueError, match="no campaign variable block"):
        tpl.parse_variable_block("units metal\nrun 100\n")


def test_parse_rejects_unterminated_block():
    truncated = TEMPLATE.replace("# ==== END CAMPAIGN VARIABLES ====", "")
    with pytest.raises(ValueError, match="never closed"):
        tpl.parse_variable_block(truncated)


def test_parse_rejects_computed_variable_inside_block():
    """`variable ymid equal ly/2` is valid LAMMPS but cannot be swept, so it
    must live below the END marker rather than silently break filling."""
    bad = TEMPLATE.replace(
        "variable crack_width_a0    equal 4\n",
        "variable crack_width_a0    equal 4\nvariable ymid equal ly/2\n",
    )
    with pytest.raises(ValueError, match="not a literal scalar declaration"):
        tpl.parse_variable_block(bad)


def test_parse_rejects_duplicate_declaration():
    bad = TEMPLATE.replace(
        "variable crack_width_a0    equal 4\n",
        "variable crack_width_a0    equal 4\nvariable temperature equal 500\n",
    )
    with pytest.raises(ValueError, match="declared twice"):
        tpl.parse_variable_block(bad)


# ── fill_variable_block ──────────────────────────────────────────────────────

def test_fill_round_trips():
    filled = tpl.fill_variable_block(
        TEMPLATE, {"temperature": 450, "strain_rate": 0.012}
    )
    values, _ = tpl.parse_variable_block(filled)
    assert values["temperature"] == 450.0
    assert values["strain_rate"] == 0.012
    # Untouched knobs keep the template's value.
    assert values["a0"] == 4.05
    assert values["crack_length_frac"] == 0.3


def test_fill_changes_only_lines_inside_the_block():
    """The whole point of the template: two trials differ by knob lines alone."""
    filled = tpl.fill_variable_block(
        TEMPLATE,
        {"temperature": 450, "strain_rate": 0.012,
         "crack_length_frac": 0.55, "crack_width_a0": 7},
    )
    before, after = TEMPLATE.split("\n"), filled.split("\n")
    assert len(before) == len(after)

    _, (begin, end) = tpl.parse_variable_block(TEMPLATE)
    changed = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
    assert changed, "expected the fill to change something"
    assert all(begin < i < end for i in changed), (
        "fill escaped the variable block:\n"
        + "\n".join(difflib.unified_diff(before, after, lineterm="", n=0))
    )


def test_fill_preserves_trailing_comment_alignment():
    filled = tpl.fill_variable_block(TEMPLATE, {"temperature": 450})
    line = next(l for l in filled.split("\n") if "temperature" in l and "equal" in l)
    assert line == "variable temperature       equal 450        # K"


def test_fill_writes_integral_values_without_trailing_zero():
    filled = tpl.fill_variable_block(TEMPLATE, {"temperature": 250.0})
    assert "equal 250 " in filled or "equal 250\n" in filled
    assert "250.0" not in filled


def test_extract_returns_the_block_with_sentinels():
    block = tpl.extract_variable_block(TEMPLATE)
    lines = block.split("\n")
    assert lines[0].startswith(tpl.VAR_BLOCK_BEGIN)
    assert lines[-1].startswith(tpl.VAR_BLOCK_END)
    # It is the whole diff between two trials, so it must carry every knob.
    for knob in KNOBS:
        assert any(f"variable {knob} " in l for l in lines), knob
    # ...and nothing from below the END marker.
    assert "fix load" not in block and "ymid" not in block


def test_extract_reflects_the_filled_values():
    """The card shows what landed, not what was asked for."""
    filled = tpl.fill_variable_block(TEMPLATE, {"temperature": 250, "strain_rate": 0.003})
    block = tpl.extract_variable_block(filled)
    assert "equal 250" in block
    assert "equal 0.003" in block
    assert "equal 300" not in block


def test_fill_rejects_undeclared_variable():
    with pytest.raises(ValueError, match="not declared"):
        tpl.fill_variable_block(TEMPLATE, {"pressure": 1.0})


# ── resolve_placeholders (the OVITO script) ──────────────────────────────────
#
# The OVITO script is Python and is stored once, reused verbatim. A surviving
# ${...} is a SyntaxError that makes the renderer fall back to default colors,
# so the crack silently stops being red.

VARS = {"a0": 3.615, "cna_cutoff": 3.087, "temperature": 400.0}


def test_resolve_substitutes_constants():
    out, problems = tpl.resolve_placeholders(
        "cna_cutoff = ${cna_cutoff}\nlat = ${a0}\n", VARS, swept=["temperature"]
    )
    assert problems == []
    assert out == "cna_cutoff = 3.087\nlat = 3.615\n"
    __import__("ast").parse(out)


def test_resolve_refuses_to_freeze_a_swept_knob():
    """Baking iteration 1's temperature into a script reused by every later
    iteration would be quietly wrong, so it is reported rather than filled."""
    out, problems = tpl.resolve_placeholders(
        "t = ${temperature}\n", VARS, swept=["temperature"]
    )
    assert "${temperature}" in out
    assert problems and "swept knob" in problems[0]


def test_resolve_reports_undeclared_name():
    _, problems = tpl.resolve_placeholders("x = ${nope}\n", VARS, swept=[])
    assert problems and "undeclared" in problems[0]


def test_resolve_leaves_placeholder_free_script_untouched():
    script = "cna_cutoff = 3.087\ncolor = (1.0, 0.0, 0.0)\n"
    out, problems = tpl.resolve_placeholders(script, VARS, swept=["temperature"])
    assert out == script and problems == []


# ── bounds checking ──────────────────────────────────────────────────────────

def test_bounds_accepts_in_range_and_flags_out_of_range():
    plan = {"knobs": KNOBS}
    assert tpl.check_knob_bounds(plan, {"temperature": 300}) == []
    problems = tpl.check_knob_bounds(plan, {"temperature": 900})
    assert len(problems) == 1
    assert "outside the planned range" in problems[0]


def test_bounds_flags_non_numeric():
    problems = tpl.check_knob_bounds({"knobs": KNOBS}, {"temperature": "hot"})
    assert problems and "not a number" in problems[0]


# ── template status on disk ──────────────────────────────────────────────────

@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """A campaign folder rooted in tmp_path, with plan.json written."""
    monkeypatch.setattr(ct, "RUNS_BASE", str(tmp_path))
    campaign_id = "benchmark_20260817_120000"
    os.makedirs(os.path.join(str(tmp_path), campaign_id))
    plan = {
        "objective": {"metric": "delta_cavity_surface_area", "direction": "minimize"},
        "template": {"material": "Al"},
        "knobs": KNOBS,
        "max_iterations": 5,
    }
    with open(ct._plan_path(campaign_id), "w") as f:
        json.dump(plan, f)
    return campaign_id


@needs_tools
def test_template_status_none_then_pending_then_approved(campaign):
    assert ct._template_status(campaign)["status"] == "none"

    with open(ct._template_path(campaign, pending=True), "w") as f:
        f.write(TEMPLATE)
    ct._write_json_atomic(
        ct._template_meta_path(campaign, pending=True),
        {"variables": {"temperature": 300}, "knob_vars": list(KNOBS)},
    )
    assert ct._template_status(campaign)["status"] == "pending"

    with open(ct._template_path(campaign), "w") as f:
        f.write(TEMPLATE)
    ct._write_json_atomic(
        ct._template_meta_path(campaign),
        {"variables": {}, "knob_vars": list(KNOBS), "approved_at": "2026-08-17T12:00:00"},
    )
    status = ct._template_status(campaign)
    assert status["status"] == "approved"
    # Variables are re-read from the script itself, not trusted from meta.
    assert status["variables"]["temperature"] == 300.0
    assert status["knob_vars"] == list(KNOBS)


@needs_tools
def test_read_template_returns_ovito_and_meta(campaign):
    with open(ct._template_path(campaign), "w") as f:
        f.write(TEMPLATE)
    with open(ct._template_ovito_path(campaign), "w") as f:
        f.write("# ovito\n")
    ct._write_json_atomic(ct._template_meta_path(campaign),
                          {"dump_filename_pattern": "dump.*.*"})

    lammps, ovito, meta = ct._read_template(campaign)
    assert lammps == TEMPLATE
    assert ovito == "# ovito\n"
    assert meta["dump_filename_pattern"] == "dump.*.*"


# ── closing a campaign ───────────────────────────────────────────────────────
# Cancelling must be final. The approval endpoint and campaign_propose_template
# both refuse a closed campaign by reading control.json, and cancel moves the
# staged template out of the approval path so there is nothing left to approve.

def _set_status(campaign_id, status):
    ct._write_json_atomic(ct._control_path(campaign_id), {"status": status})


def _stage_template(campaign_id):
    with open(ct._template_path(campaign_id, pending=True), "w") as f:
        f.write(TEMPLATE)
    with open(ct._template_ovito_path(campaign_id, pending=True), "w") as f:
        f.write("# ovito\n")
    ct._write_json_atomic(ct._template_meta_path(campaign_id, pending=True),
                          {"variables": {}, "knob_vars": list(KNOBS)})


@needs_tools
@pytest.mark.parametrize("status", ["running", "paused", "awaiting_template_approval"])
def test_closed_status_none_while_campaign_is_open(campaign, status):
    _set_status(campaign, status)
    assert ct.campaign_closed_status(campaign) is None


@needs_tools
@pytest.mark.parametrize("status", ["cancelled", "done"])
def test_closed_status_reports_terminal_states(campaign, status):
    _set_status(campaign, status)
    assert ct.campaign_closed_status(campaign) == status


@needs_tools
def test_closed_status_treats_unreadable_control_as_open(campaign):
    # Absence of a cancel flag is not evidence of one — a missing or corrupt
    # control.json must never strand a healthy campaign.
    assert ct.campaign_closed_status(campaign) is None
    with open(ct._control_path(campaign), "w") as f:
        f.write("{not json")
    assert ct.campaign_closed_status(campaign) is None


@needs_tools
def test_reject_pending_template_renames_and_clears_pending_status(campaign):
    _stage_template(campaign)
    assert ct._template_status(campaign)["status"] == "pending"

    moved = ct.reject_pending_template(campaign)

    assert len(moved) == 3
    assert all(p.endswith(ct.REJECTED_SUFFIX) for p in moved)
    assert not os.path.exists(ct._template_path(campaign, pending=True))
    # A dead campaign must stop advertising a template awaiting approval.
    assert ct._template_status(campaign)["status"] == "none"
    # Renamed, not deleted: the refused script is still readable.
    with open(ct._template_path(campaign, pending=True)[:-len(ct.PENDING_SUFFIX)]
              + ct.REJECTED_SUFFIX) as f:
        assert f.read() == TEMPLATE


@needs_tools
def test_reject_pending_template_is_noop_when_nothing_staged(campaign):
    # Cancel is also reachable mid-run, where the template was approved long ago.
    assert ct.reject_pending_template(campaign) == []


@needs_tools
@pytest.mark.parametrize("status", ["cancelled", "done"])
def test_propose_template_refuses_closed_campaign(campaign, status):
    _stage_template(campaign)
    _set_status(campaign, status)

    result = ct.campaign_propose_template.func(
        source_iteration=1, state={"campaign_id": campaign}
    )

    assert status in result["error"]
    # The staged copy must NOT be re-shown for approval.
    assert "next_step" not in result
