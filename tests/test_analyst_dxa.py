#!/usr/bin/env python3
"""
Tests for the Analyst agent's dxa_dislocation_density tool.

Covers tool registration, the cached-CSV parser, lattice auto-detection from
input.lammps, the cached read path (which must never re-run OVITO), graceful
failure, and a deliberately tiny compute path.

No mocks for the data: every assertion runs against a real run folder, so a
change that silently alters the numbers fails here rather than in the chat
window. The tests that need a cached DXA result are skipped when that cache is
absent, so a fresh clone still collects cleanly.
"""

import os
import shutil
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from graph.nodes import (  # noqa: E402
    _ANALYST_TOOLS,
    _detect_lattice,
    _dxa_lattice_warning,
    _dxa_rows_from_csv,
    dxa_dislocation_density,
)
from prompts import ANALYST_SYSTEM_PROMPT  # noqa: E402


# A completed Cu crack run with a full 32-frame DXA cache committed alongside it.
CU_RUN = os.path.join(REPO_ROOT, "runs", "run_2026_08_12_102150_Cu_final")
CU_CSV = os.path.join(CU_RUN, "dxa_analysis", "dxa_summary.csv")
# Al uses phase labels 'phase1'/'phase2'/'phase3' where Cu uses
# 'prestress'/'relaxation'/'loading' — the pair proves no logic branches on the
# free-form phase string.
AL_RUN = os.path.join(REPO_ROOT, "runs", "run_2026_08_12_114944_Al_final")
AL_CSV = os.path.join(AL_RUN, "dxa_analysis", "dxa_summary.csv")
# Three dumps only, so the compute path finishes in about a second.
TINY_RUN = os.path.join(REPO_ROOT, "runs", "run_2026_09_24_202721")

needs_cu = pytest.mark.skipif(not os.path.exists(CU_CSV), reason="Cu DXA cache absent")
needs_al = pytest.mark.skipif(not os.path.exists(AL_CSV), reason="Al DXA cache absent")


# =============================================================================
# Registration
# =============================================================================

class TestToolRegistration:

    def test_tool_is_registered(self):
        names = [t.name for t in _ANALYST_TOOLS]
        assert "dxa_dislocation_density" in names
        # The pre-existing tools must survive the addition.
        assert {"list_recent_runs", "get_run_log", "stress_strain_curve"} <= set(names)

    def test_tool_schema_exposes_arguments(self):
        schema = dxa_dislocation_density.args
        assert "run_name" in schema
        assert "lattice" in schema

    def test_lattice_defaults_to_auto(self):
        # The whole point of _detect_lattice is that the agent never chooses.
        assert dxa_dislocation_density.args["lattice"].get("default") == "auto"

    def test_prompt_advertises_the_tool(self):
        # An unlisted tool gets ignored in practice.
        assert "dxa_dislocation_density" in ANALYST_SYSTEM_PROMPT
        assert "Shockley" in ANALYST_SYSTEM_PROMPT

    def test_prompt_tells_the_agent_not_to_pass_lattice(self):
        assert "Do NOT pass `lattice`" in ANALYST_SYSTEM_PROMPT


# =============================================================================
# CSV parser — pure stdlib, no ovito, no run folder
# =============================================================================

@needs_cu
class TestRowsFromCsv:

    @pytest.fixture(scope="class")
    def parsed(self):
        return _dxa_rows_from_csv(CU_CSV)

    def test_row_count_and_ordering(self, parsed):
        rows, _ = parsed
        assert len(rows) == 32
        assert [r["timestep"] for r in rows] == sorted(r["timestep"] for r in rows)
        assert rows[0]["timestep"] == 0
        assert rows[-1]["timestep"] == 120000

    def test_families_come_from_header_order(self, parsed):
        # Header order is what write_summary_csv used, so rows rebuilt here
        # re-plot identically to the ones that produced the file.
        _, families = parsed
        assert families == ["1/2<110>", "1/6<112>", "1/6<110>",
                            "1/3<100>", "1/3<111>", "other"]

    def test_hyphenated_density_columns_are_mapped(self, parsed):
        # The CSV spells these density_m-2 / density_A-2; the dict uses m2 / A2.
        rows, _ = parsed
        assert rows[-1]["dxa"]["density_m2"] == pytest.approx(2.856004e17, rel=1e-5)
        assert rows[-1]["dxa"]["density_A2"] > 0

    def test_counts_and_lengths_round_trip(self, parsed):
        rows, families = parsed
        last = rows[-1]["dxa"]
        assert last["counts"]["HCP"] == 9435
        assert last["n_lines"] == 226
        assert set(last["lengths"]) == set(families)
        assert last["lengths"]["1/6<112>"] == pytest.approx(2736, rel=0.01)

    def test_phase_label_is_preserved_verbatim(self, parsed):
        rows, _ = parsed
        assert rows[-1]["phase"] == "loading"

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "dxa_summary.csv"
        p.write_text("")
        with pytest.raises(ValueError):
            _dxa_rows_from_csv(str(p))

    def test_header_only_raises(self, tmp_path):
        p = tmp_path / "dxa_summary.csv"
        p.write_text("timestep,phase,n_lines,density_m-2\n")
        with pytest.raises(ValueError):
            _dxa_rows_from_csv(str(p))

    def test_foreign_csv_raises(self, tmp_path):
        # A stress_strain.csv accidentally renamed must not parse as DXA data.
        p = tmp_path / "dxa_summary.csv"
        p.write_text("strain,stress_GPa\n0.0,0.0\n0.01,1.2\n")
        with pytest.raises(ValueError):
            _dxa_rows_from_csv(str(p))

    def test_truncated_last_line_keeps_the_good_rows(self, tmp_path):
        # An interrupted write leaves a partial final line; the complete rows
        # above it are still worth reading.
        good = open(CU_CSV).read().splitlines()
        p = tmp_path / "dxa_summary.csv"
        p.write_text("\n".join(good[:5]) + "\n,,,\n")
        rows, _ = _dxa_rows_from_csv(str(p))
        assert len(rows) == 4


# =============================================================================
# Lattice auto-detection
# =============================================================================

class TestDetectLattice:
    """The agent must never be the one guessing FCC vs BCC.

    A wrong lattice does not raise — DXA just finds almost nothing and returns a
    plausible near-zero result, so detection failing silently is the risk.
    """

    @pytest.mark.parametrize("run,want", [
        ("run_2026_08_12_102150_Cu_final", "FCC"),
        ("run_2026_09_24_203323", "BCC"),
        # `lattice hcp ${a} orient x 1 0 0 ... c/a $(v_c/v_a)` — everything
        # after the style token must be ignored, not parsed.
        ("run_2026_09_24_201241", "HCP"),
    ])
    def test_real_runs(self, run, want):
        folder = os.path.join(REPO_ROOT, "runs", run)
        if not os.path.isdir(folder):
            pytest.skip(f"{run} absent")
        assert _detect_lattice(folder) == want

    def test_missing_input_returns_none(self, tmp_path):
        assert _detect_lattice(str(tmp_path)) is None

    def test_unsupported_style_returns_none(self, tmp_path):
        # DXA has no input_crystal_structure for sc or custom.
        (tmp_path / "input.lammps").write_text("units metal\nlattice sc 3.0\n")
        assert _detect_lattice(str(tmp_path)) is None

    def test_ignores_trailing_arguments(self, tmp_path):
        (tmp_path / "input.lammps").write_text(
            "lattice fcc ${a0} orient x 1 1 0 orient y -1 1 0 orient z 0 0 1\n")
        assert _detect_lattice(str(tmp_path)) == "FCC"


# =============================================================================
# Lattice cross-check
# =============================================================================

@needs_cu
class TestLatticeCrossCheck:
    """DXA classifies only the structure it was asked for; everything else
    becomes 'Other'. A wrong lattice therefore shows up as a pristine frame 0
    that is overwhelmingly 'Other', which is the signal this check reads."""

    def test_no_warning_when_lattice_matches(self):
        rows, _ = _dxa_rows_from_csv(CU_CSV)
        assert _dxa_lattice_warning(rows, "FCC") == ""

    def test_warns_when_lattice_disagrees(self):
        rows, _ = _dxa_rows_from_csv(CU_CSV)
        msg = _dxa_lattice_warning(rows, "BCC")
        assert "may be wrong" in msg and "FCC" in msg

    def test_catches_a_cache_poisoned_by_the_cli_default(self):
        # `python dxa_analysis.py <bcc_run>` still defaults to --lattice FCC
        # and writes no lattice into the CSV, so the zeros it produces would
        # otherwise be read back as a genuine defect-free result. Measured
        # signature of that mistake: frame 0 is 100% 'Other'.
        rows = [{"dxa": {"counts": {"Other": 1024, "FCC": 0, "BCC": 0}}}]
        assert "may be wrong" in _dxa_lattice_warning(rows, "FCC")

    def test_silent_when_there_is_nothing_to_compare(self):
        assert _dxa_lattice_warning([{"dxa": {"counts": {}}}], "FCC") == ""


# =============================================================================
# Cached read path
# =============================================================================

@needs_cu
class TestCachedPath:

    @pytest.fixture(scope="class")
    def result(self):
        return dxa_dislocation_density.invoke({"run_name": CU_RUN})

    def test_reports_the_cache_as_its_source(self, result):
        assert "cached CSV" in result
        assert "failed" not in result.lower()

    def test_reports_onset_peak_and_families(self, result):
        assert "step 88000" in result          # nucleation onset
        assert "2.947e+17" in result           # peak density
        assert "Shockley 1/6<112>" in result   # full BURGERS_LABEL, not a stem

    def test_reports_structure_fractions(self, result):
        assert "HCP 9.45%" in result

    def test_does_not_clobber_the_cache(self, result):
        # The CSV may be a full-resolution one the user produced from the CLI.
        before = os.path.getmtime(CU_CSV)
        dxa_dislocation_density.invoke({"run_name": CU_RUN})
        assert os.path.getmtime(CU_CSV) == before

    def test_never_calls_analyze_frame(self, monkeypatch, result):
        # The load-bearing assertion: checking `"ovito" in sys.modules` is
        # unreliable because another test may have imported it.
        import dxa_analysis

        def _boom(*a, **k):
            raise AssertionError("analyze_frame must not run on the cached path")

        monkeypatch.setattr(dxa_analysis, "analyze_frame", _boom)
        out = dxa_dislocation_density.invoke({"run_name": CU_RUN})
        assert "cached CSV" in out

    def test_requested_lattice_against_a_cache_is_caveated(self):
        # The CSV records neither the lattice nor the frame selection that
        # produced it, so echoing the parameter would be a fabrication.
        out = dxa_dislocation_density.invoke({"run_name": CU_RUN, "lattice": "BCC"})
        assert "does not record which lattice" in out


@needs_al
class TestCachedPathOnAluminium:
    """Same code path, different phase vocabulary — nothing may branch on it."""

    @pytest.fixture(scope="class")
    def result(self):
        return dxa_dislocation_density.invoke({"run_name": AL_RUN})

    def test_reports_onset_and_peak(self, result):
        assert "step 76000" in result
        assert "9.885e+16" in result

    def test_passes_the_free_form_phase_through(self, result):
        assert "phase3" in result


# =============================================================================
# Failure paths — a raised exception inside a tool aborts the agent turn
# =============================================================================

class TestGracefulFailure:

    def test_unresolvable_run_returns_string(self):
        out = dxa_dislocation_density.invoke({"run_name": "no_such_run_xyz"})
        assert isinstance(out, str)
        assert "Could not resolve" in out

    def test_empty_run_name_returns_string(self):
        out = dxa_dislocation_density.invoke({"run_name": ""})
        assert isinstance(out, str)
        assert "Could not resolve" in out

    def test_run_without_dumps_explains_itself(self, tmp_path):
        out = dxa_dislocation_density.invoke({"run_name": str(tmp_path)})
        assert isinstance(out, str)
        assert "No LAMMPS dump files" in out

    def test_invalid_lattice_returns_string_and_lists_the_valid_ones(self):
        out = dxa_dislocation_density.invoke(
            {"run_name": CU_RUN, "lattice": "Wurtzite"})
        assert isinstance(out, str)
        assert "Invalid lattice" in out and "FCC" in out

    def test_unreadable_cache_points_at_the_fix(self, tmp_path):
        os.makedirs(tmp_path / "dxa_analysis")
        (tmp_path / "dxa_analysis" / "dxa_summary.csv").write_text("garbage\n")
        out = dxa_dislocation_density.invoke({"run_name": str(tmp_path)})
        assert isinstance(out, str)
        assert "could not read it" in out
        assert "Delete that file" in out


# =============================================================================
# Compute path — kept tiny so the slow branch stays covered
# =============================================================================

@pytest.mark.skipif(not os.path.isdir(TINY_RUN), reason="3-frame fixture absent")
class TestComputePath:

    @pytest.fixture(scope="class")
    def fresh_run(self, tmp_path_factory):
        # Copy into tmp so the real run's cache is neither read nor written.
        dest = tmp_path_factory.mktemp("dxa_compute") / "run"
        shutil.copytree(TINY_RUN, dest,
                        ignore=shutil.ignore_patterns("dxa_analysis"))
        return str(dest)

    @pytest.fixture(scope="class")
    def result(self, fresh_run):
        pytest.importorskip("ovito")
        return dxa_dislocation_density.invoke({"run_name": fresh_run})

    def test_computes_rather_than_reading_a_cache(self, result):
        assert "computed now" in result
        assert "auto-detected from input.lammps" in result

    def test_analyzes_every_frame(self, result):
        assert "frames:            3" in result

    def test_writes_the_cache(self, result, fresh_run):
        csv_path = os.path.join(fresh_run, "dxa_analysis", "dxa_summary.csv")
        assert os.path.exists(csv_path)
        rows, _ = _dxa_rows_from_csv(csv_path)
        assert len(rows) == 3

    def test_second_call_reads_the_cache_it_just_wrote(self, result, fresh_run):
        assert "cached CSV" in dxa_dislocation_density.invoke({"run_name": fresh_run})

    def test_zero_density_run_says_so_explicitly(self, result):
        # This run never nucleates; the agent has no vision, so silence would
        # read as "the tool found nothing to say".
        assert "NONE" in result
        assert "defect-free" in result
