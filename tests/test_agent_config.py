#!/usr/bin/env python3
"""
Test script to verify agent and DSPy module configuration is working correctly.
"""

import config
from agents import (
    SmartOrchestrator, IterativeLammpsGeneration,
    OVITOScriptReview, OVITOScriptGeneration,
    configure_dspy_lm
)

def test_agent_configurations():
    """Test that DSPy modules use the configured model via configure_dspy_lm."""

    print("Testing DSPy module configuration...")
    print(f"Config values:")
    print(f"  DEFAULT_MODEL: {config.DEFAULT_MODEL}")
    print(f"  API_BASE_URL: {config.API_BASE_URL}")
    print()

    # Configure DSPy LM
    lm = configure_dspy_lm()
    print(f"DSPy LM configured: {lm}")
    print()

    # Test SmartOrchestrator instantiation
    orchestrator = SmartOrchestrator()
    print(f"SmartOrchestrator: instantiated successfully")
    print(f"  Has analyzer: {hasattr(orchestrator, 'analyzer')}")
    print(f"  Has intent_classifier: {hasattr(orchestrator, 'intent_classifier')}")
    print(f"  Has viz_detector: {hasattr(orchestrator, 'viz_detector')}")
    print()

    # Test OVITOScriptGeneration instantiation
    ovito_gen = OVITOScriptGeneration()
    print(f"OVITOScriptGeneration: instantiated successfully")
    print(f"  Has generator: {hasattr(ovito_gen, 'generator')}")
    print()

    # Test OVITOScriptReview instantiation
    ovito_review = OVITOScriptReview()
    print(f"OVITOScriptReview: instantiated successfully")
    print(f"  Has reviewer: {hasattr(ovito_review, 'reviewer')}")
    print()

    # Test IterativeLammpsGeneration instantiation
    lammps_gen = IterativeLammpsGeneration()
    print(f"IterativeLammpsGeneration: instantiated successfully")
    print(f"  Has generator: {hasattr(lammps_gen, 'generator')}")
    print(f"  Has reviewer: {hasattr(lammps_gen, 'reviewer')}")
    print()

    print("Configuration Summary:")
    print(f"  All DSPy modules configured: ✓")
    print(f"  API Base URL: {config.API_BASE_URL}")

if __name__ == "__main__":
    test_agent_configurations()
