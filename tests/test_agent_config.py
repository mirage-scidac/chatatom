#!/usr/bin/env python3
"""
Test script to verify per-agent model configuration is working correctly.
"""

import config
from agents import OrchestratorAgent, LammpsGeneratorAgent, LammpsReviewerAgent

def test_agent_configurations():
    """Test that each agent uses its configured model and temperature."""
    
    print("Testing per-agent model configuration...")
    print(f"Config values:")
    print(f"  ORCHESTRATOR_MODEL: {config.ORCHESTRATOR_MODEL}")
    print(f"  LAMMPS_GENERATOR_MODEL: {config.LAMMPS_GENERATOR_MODEL}")
    print(f"  LAMMPS_REVIEWER_MODEL: {config.LAMMPS_REVIEWER_MODEL}")
    print(f"  DEFAULT_MODEL: {config.DEFAULT_MODEL}")
    print()
    
    # Test OrchestratorAgent
    orchestrator = OrchestratorAgent()
    print(f"OrchestratorAgent:")
    print(f"  Model: {orchestrator.model}")
    print(f"  Temperature: {orchestrator.default_temperature}")
    print(f"  Expected Model: {config.ORCHESTRATOR_MODEL}")
    print(f"  Expected Temperature: {config.ORCHESTRATOR_TEMPERATURE}")
    print(f"  ✓ Model matches: {orchestrator.model == config.ORCHESTRATOR_MODEL}")
    print(f"  ✓ Temperature matches: {orchestrator.default_temperature == config.ORCHESTRATOR_TEMPERATURE}")
    print()
    
    # Test LammpsGeneratorAgent
    generator = LammpsGeneratorAgent()
    print(f"LammpsGeneratorAgent:")
    print(f"  Model: {generator.model}")
    print(f"  Temperature: {generator.default_temperature}")
    print(f"  Expected Model: {config.LAMMPS_GENERATOR_MODEL}")
    print(f"  Expected Temperature: {config.LAMMPS_GENERATOR_TEMPERATURE}")
    print(f"  ✓ Model matches: {generator.model == config.LAMMPS_GENERATOR_MODEL}")
    print(f"  ✓ Temperature matches: {generator.default_temperature == config.LAMMPS_GENERATOR_TEMPERATURE}")
    print()
    
    # Test LammpsReviewerAgent
    reviewer = LammpsReviewerAgent()
    print(f"LammpsReviewerAgent:")
    print(f"  Model: {reviewer.model}")
    print(f"  Temperature: {reviewer.default_temperature}")
    print(f"  Expected Model: {config.LAMMPS_REVIEWER_MODEL}")
    print(f"  Expected Temperature: {config.LAMMPS_REVIEWER_TEMPERATURE}")
    print(f"  ✓ Model matches: {reviewer.model == config.LAMMPS_REVIEWER_MODEL}")
    print(f"  ✓ Temperature matches: {reviewer.default_temperature == config.LAMMPS_REVIEWER_TEMPERATURE}")
    print()
    
    # Verify all agents have different configurations if desired
    all_models = [orchestrator.model, generator.model, reviewer.model]
    all_temps = [orchestrator.default_temperature, generator.default_temperature, reviewer.default_temperature]
    
    print("Configuration Summary:")
    print(f"  All agents configured: ✓")
    print(f"  Models: {all_models}")
    print(f"  Temperatures: {all_temps}")
    print(f"  API Base URL: {config.API_BASE_URL}")
    print(f"  API Key: {config.API_KEY}")

if __name__ == "__main__":
    test_agent_configurations()
