#!/usr/bin/env python3
# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE
"""
Tests for the general LAMMPS scripting rules snippet.

Guards the content of KnowledgeBase.LAMMPS_ERROR_RULE and its wiring into the
generator and reviewer knowledge contexts.
"""

from prompts import KnowledgeBase


class TestLammpsErrorRule:
    """Verify the snippet exists and keeps its load-bearing content."""

    def test_exists(self):
        assert hasattr(KnowledgeBase, "LAMMPS_ERROR_RULE")

    def test_key_rules_present(self):
        snippet = KnowledgeBase.LAMMPS_ERROR_RULE
        for marker in [
            "$(f_ID)",        # snapshot a fix value before the fix is destroyed
            "units box",      # region/create_atoms/fix deform default to lattice
            "aniso",          # non-cubic cells need shape freedom
            "1.63299",        # iso freezes HCP c/a at the ideal ratio
            "Lost atoms",     # 'f' boundary with no vacuum padding
            "compress no",    # keep atom IDs stable across phases
            "velocity create",
        ]:
            assert marker in snippet, f"missing: {marker}"

    def test_does_not_contradict_griffith_snippet(self):
        """GRIFFITH_CRACK_TECHNIQUE and the ReviewLAMMPSScript docstring both
        specify 'compress no'. Guard against reintroducing the conflict."""
        assert "compress yes" not in KnowledgeBase.LAMMPS_ERROR_RULE
        assert "compress no" in KnowledgeBase.GRIFFITH_CRACK_TECHNIQUE
