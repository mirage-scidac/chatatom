#!/usr/bin/env python3
# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE
"""
Tests for the Analyst agent's stress_strain_curve tool.

Covers tool registration, the shared run-folder resolver, an end-to-end run
against a real completed tensile simulation, and graceful failure on runs that
have no loading phase.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from graph.nodes import (  # noqa: E402
    _ANALYST_TOOLS,
    _resolve_run_dir,
    stress_strain_curve,
)
from prompts import ANALYST_SYSTEM_PROMPT  # noqa: E402


# A completed Cu tensile run (erate 0.01 along y) whose thermo carries pyy/ly.
FIXTURE_RUN = os.path.join(
    REPO_ROOT, "runs", "benchmark_20260815_160551", "run_2026_08_15_110617"
)


# =============================================================================
# Registration
# =============================================================================

class TestToolRegistration:
    """The tool must be wired into the agent and describe itself correctly."""

    def test_tool_is_registered(self):
        names = [t.name for t in _ANALYST_TOOLS]
        assert "stress_strain_curve" in names
        # The pre-existing tools must survive the addition.
        assert "list_recent_runs" in names
        assert "get_run_log" in names

    def test_tool_schema_exposes_arguments(self):
        schema = stress_strain_curve.args
        assert "run_name" in schema
        assert "axis" in schema
        assert "fit_max_strain" in schema

    def test_prompt_advertises_the_tool(self):
        # An unlisted tool gets ignored in practice.
        assert "stress_strain_curve" in ANALYST_SYSTEM_PROMPT
        assert "Young's modulus" in ANALYST_SYSTEM_PROMPT


# =============================================================================
# Run-folder resolution
# =============================================================================

class TestResolveRunDir:
    """_resolve_run_dir backs both get_run_log and stress_strain_curve."""

    def test_empty_returns_empty(self):
        assert _resolve_run_dir("") == ""

    def test_junk_returns_empty(self):
        assert _resolve_run_dir("no_such_run_xyz") == ""

    def test_out_of_range_number_returns_empty(self):
        assert _resolve_run_dir("99999") == ""

    @pytest.mark.skipif(not os.path.isdir(FIXTURE_RUN), reason="fixture run absent")
    def test_full_path_resolves(self):
        assert _resolve_run_dir(FIXTURE_RUN) == os.path.abspath(FIXTURE_RUN)

    @pytest.mark.skipif(not os.path.isdir(FIXTURE_RUN), reason="fixture run absent")
    def test_analyze_prefix_is_stripped(self):
        # The LLM often passes the user's whole phrase through.
        assert _resolve_run_dir(f"analyze {FIXTURE_RUN}") == os.path.abspath(FIXTURE_RUN)


# =============================================================================
# End-to-end analysis
# =============================================================================

@pytest.mark.skipif(not os.path.isdir(FIXTURE_RUN), reason="fixture run absent")
class TestStressStrainOnRealRun:
    """Run the tool against a real tensile trajectory."""

    @pytest.fixture(scope="class")
    def result(self):
        return stress_strain_curve.invoke({"run_name": FIXTURE_RUN})

    def test_reports_modulus_and_peak(self, result):
        assert "Young's modulus" in result
        assert "peak stress" in result
        assert "GPa" in result
        assert "failed" not in result.lower()

    def test_identifies_loading_axis(self, result):
        assert "loading axis:    y" in result
        assert "-pyy" in result

    def test_writes_outputs(self, result):
        assert os.path.exists(os.path.join(FIXTURE_RUN, "stress_strain.csv"))
        assert os.path.exists(os.path.join(FIXTURE_RUN, "stress_strain.png"))

    def test_strain_is_measured_from_loading_start(self, result):
        # Strain starts at 0 at the first row of the deform block, so the max
        # strain must be positive and physically small (erate 0.01 * 15 ps).
        line = [l for l in result.splitlines() if l.startswith("max strain")][0]
        max_strain = float(line.split(":")[1].strip())
        assert 0.0 < max_strain < 1.0


# =============================================================================
# Failure paths — a raised exception inside a tool aborts the agent turn
# =============================================================================

class TestGracefulFailure:

    def test_unresolvable_run_returns_string(self):
        out = stress_strain_curve.invoke({"run_name": "no_such_run_xyz"})
        assert isinstance(out, str)
        assert "Could not resolve" in out

    def test_missing_log_returns_string(self, tmp_path):
        out = stress_strain_curve.invoke({"run_name": str(tmp_path)})
        assert isinstance(out, str)
        assert "No LAMMPS log file found" in out

    def test_invalid_axis_returns_string(self):
        out = stress_strain_curve.invoke({"run_name": "latest", "axis": "q"})
        assert isinstance(out, str)
        assert "Invalid axis" in out

    @pytest.mark.skipif(not os.path.isdir(FIXTURE_RUN), reason="fixture run absent")
    def test_axis_without_deformation_explains_itself(self):
        # The pull is along y; x never deforms, so this must explain rather
        # than raise or invent a modulus.
        out = stress_strain_curve.invoke({"run_name": FIXTURE_RUN, "axis": "x"})
        assert "no tensile-loading phase" in out
