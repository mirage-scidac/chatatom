#!/usr/bin/env python3
"""
Tests for Griffith crack and fatigue simulation prompt support.

Verifies that KnowledgeBase snippets, Orchestrator, and DSPy signature
prompts contain the necessary crack/fatigue content for proper script generation
and review.
"""

import pytest
from prompts import (
    KnowledgeBase,
    ORCHESTRATOR_SYSTEM_PROMPT,
    GenerateLAMMPSScript,
    ReviewLAMMPSScript,
)


# =============================================================================
# KnowledgeBase snippet tests
# =============================================================================

class TestKnowledgeBaseCrackSnippets:
    """Verify KnowledgeBase contains crack/fatigue technique snippets."""

    def test_griffith_crack_technique_exists(self):
        assert hasattr(KnowledgeBase, "GRIFFITH_CRACK_TECHNIQUE")
        snippet = KnowledgeBase.GRIFFITH_CRACK_TECHNIQUE
        assert "neigh_modify exclude" in snippet
        assert "crack_lower" in snippet
        assert "crack_upper" in snippet
        assert "region" in snippet
        assert "group" in snippet

    def test_griffith_crack_technique_ordering_constraint(self):
        snippet = KnowledgeBase.GRIFFITH_CRACK_TECHNIQUE
        assert "AFTER create_atoms" in snippet

    def test_griffith_crack_geometry_variables(self):
        snippet = KnowledgeBase.GRIFFITH_CRACK_TECHNIQUE
        for var in ["xmid", "xbound_low", "xbound_high", "ybound_low", "ybound_high"]:
            assert var in snippet, f"Missing geometry variable: {var}"

    def test_cyclic_loading_technique_exists(self):
        assert hasattr(KnowledgeBase, "CYCLIC_LOADING_TECHNIQUE")
        snippet = KnowledgeBase.CYCLIC_LOADING_TECHNIQUE
        assert "fix deform" in snippet
        assert "wiggle" in snippet
        assert "NVT" in snippet

    def test_cyclic_loading_rejects_npt(self):
        snippet = KnowledgeBase.CYCLIC_LOADING_TECHNIQUE
        assert "NOT NPT" in snippet or "not NPT" in snippet

    def test_cyclic_loading_amplitude_formula(self):
        snippet = KnowledgeBase.CYCLIC_LOADING_TECHNIQUE
        assert "strain_fraction" in snippet or "ampL" in snippet

    def test_crack_output_requirements_exists(self):
        assert hasattr(KnowledgeBase, "CRACK_OUTPUT_REQUIREMENTS")
        snippet = KnowledgeBase.CRACK_OUTPUT_REQUIREMENTS
        assert "pe/atom" in snippet
        assert "stress/atom" in snippet

    def test_crack_output_stress_components(self):
        snippet = KnowledgeBase.CRACK_OUTPUT_REQUIREMENTS
        for component in ["stress_atom[1]", "stress_atom[2]", "stress_atom[3]"]:
            assert component in snippet, f"Missing stress component: {component}"

    def test_crack_output_undump_pattern(self):
        snippet = KnowledgeBase.CRACK_OUTPUT_REQUIREMENTS
        assert "undump" in snippet


# =============================================================================
# Orchestrator prompt tests
# =============================================================================

class TestOrchestratorCrackRecognition:
    """Verify Orchestrator prompt recognizes crack/fatigue keywords."""

    def test_crack_keywords_present(self):
        for keyword in ["Griffith crack", "crack propagation", "fracture"]:
            assert keyword in ORCHESTRATOR_SYSTEM_PROMPT, f"Missing keyword: {keyword}"

    def test_fatigue_keywords_present(self):
        for keyword in ["fatigue", "cyclic loading"]:
            assert keyword in ORCHESTRATOR_SYSTEM_PROMPT, f"Missing keyword: {keyword}"

    def test_three_phase_crack_simulation(self):
        assert "3-phase simulation" in ORCHESTRATOR_SYSTEM_PROMPT

    def test_four_phase_fatigue_simulation(self):
        assert "4-phase simulation" in ORCHESTRATOR_SYSTEM_PROMPT

    def test_crack_defaults(self):
        prompt = ORCHESTRATOR_SYSTEM_PROMPT
        assert "20x20x5" in prompt, "Missing minimum system size default"

    def test_crack_specific_output_fields(self):
        prompt = ORCHESTRATOR_SYSTEM_PROMPT
        assert "Simulation type: Griffith Crack" in prompt
        assert "Strain amplitude" in prompt
        assert "Loading period" in prompt

    def test_pre_stress_phase(self):
        assert "Pre-stress" in ORCHESTRATOR_SYSTEM_PROMPT


# =============================================================================
# DSPy signature tests
# =============================================================================

class TestGenerateLAMMPSScriptSignature:
    """Verify GenerateLAMMPSScript DSPy signature has required fields."""

    def test_simulation_details_field_exists(self):
        fields = GenerateLAMMPSScript.model_fields
        assert "simulation_details" in fields, "Missing simulation_details field in GenerateLAMMPSScript"

    def test_knowledge_context_field_exists(self):
        fields = GenerateLAMMPSScript.model_fields
        assert "knowledge_context" in fields, "Missing knowledge_context field in GenerateLAMMPSScript"


class TestReviewLAMMPSScriptSignature:
    """Verify ReviewLAMMPSScript DSPy signature has required fields."""

    def test_knowledge_context_field_exists(self):
        fields = ReviewLAMMPSScript.model_fields
        assert "knowledge_context" in fields, "Missing knowledge_context field in ReviewLAMMPSScript"


# =============================================================================
# KnowledgeBase content tests (replacing prompt-based tests)
# =============================================================================

class TestKnowledgeBaseGeneratorContent:
    """Verify KnowledgeBase contains content needed by the generator."""

    def test_crack_template_phases(self):
        snippet = KnowledgeBase.GRIFFITH_CRACK_TECHNIQUE
        assert "crack_lower" in snippet
        assert "crack_upper" in snippet

    def test_crack_template_uses_kim(self):
        snippet = KnowledgeBase.KIM_MANDATORY_SEQUENCE
        assert "kim init" in snippet
        assert "kim interactions" in snippet

    def test_crack_seeding_in_knowledge(self):
        snippet = KnowledgeBase.GRIFFITH_CRACK_TECHNIQUE
        assert "neigh_modify exclude" in snippet
        assert "crack_lower" in snippet
        assert "crack_upper" in snippet

    def test_fix_deform_wiggle_in_knowledge(self):
        snippet = KnowledgeBase.CYCLIC_LOADING_TECHNIQUE
        assert "fix deform" in snippet
        assert "wiggle" in snippet

    def test_undump_unfix_in_knowledge(self):
        snippet = KnowledgeBase.CRACK_OUTPUT_REQUIREMENTS
        assert "undump" in snippet

    def test_npt_deform_conflict_in_knowledge(self):
        snippet = KnowledgeBase.CYCLIC_LOADING_TECHNIQUE
        assert "NOT NPT" in snippet or "NVT" in snippet


class TestKnowledgeBaseReviewerContent:
    """Verify KnowledgeBase contains content needed by the reviewer."""

    def test_kim_ordering_rules(self):
        snippet = KnowledgeBase.KIM_MANDATORY_SEQUENCE
        assert "BEFORE create_box" in snippet
        assert "AFTER create_box" in snippet

    def test_crack_ordering_constraint(self):
        snippet = KnowledgeBase.GRIFFITH_CRACK_TECHNIQUE
        assert "AFTER create_atoms" in snippet

    def test_pe_stress_in_crack_output(self):
        snippet = KnowledgeBase.CRACK_OUTPUT_REQUIREMENTS
        assert "pe/atom" in snippet
        assert "stress/atom" in snippet


# =============================================================================
# Integration test: DSPy Reviewer catches NPT + fix deform conflict
# =============================================================================

class TestReviewerCrackIntegration:
    """Integration tests using the DSPy ReviewLAMMPSScript (requires LLM API)."""

    @pytest.mark.slow
    def test_reviewer_rejects_npt_with_fix_deform(self):
        """Reviewer should FAIL a script that combines NPT with fix deform."""
        import dspy
        from agents import configure_dspy_lm

        bad_script = """units metal
dimension 3
boundary p p p
atom_style atomic

kim init EAM_Dynamo_MishinMehlPapaconstantopoulos_2001_Cu__MO_346334655118_005 metal

variable a0 equal 3.615
lattice fcc ${a0}
region whole block 0 20 0 20 0 5 units lattice
create_box 1 whole
create_atoms 1 region whole

kim interactions Cu
mass 1 63.546

velocity all create 300.0 12345 rot yes dist gaussian

neighbor 0.3 bin
neigh_modify delay 0 every 100
timestep 0.001

compute peratom all pe/atom
compute stress_atom all stress/atom NULL

thermo 1000
thermo_style custom step time temp pe press pxx pyy pzz vol lx ly lz

# BAD: NPT with fix deform - these conflict!
fix 1 all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0
variable ampL equal 0.0025*lx
fix 2 all deform 1 x wiggle ${ampL} 100.0

dump 1 all custom 1000 dump.*.cycled id type x y z c_peratom c_stress_atom[1] c_stress_atom[2] c_stress_atom[3]

run 500000"""

        configure_dspy_lm()
        reviewer = dspy.ChainOfThought(ReviewLAMMPSScript)
        result = reviewer(
            lammps_script=bad_script,
            intended_parameters="Cu at 300K, NVT with cyclic loading, 500ps"
        )

        # is_valid should be False (or a string representation of false)
        is_valid = result.is_valid
        if isinstance(is_valid, str):
            is_valid = is_valid.lower() in ('true', 'yes')

        assert not is_valid, (
            f"Reviewer should FAIL NPT + fix deform script, got is_valid={result.is_valid}, issues={result.issues}"
        )
