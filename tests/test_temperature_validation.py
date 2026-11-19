#!/usr/bin/env python3
# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE
"""
Test script to verify that the DSPy-based ReviewLAMMPSScript can catch
the missing velocity initialization issue that causes 0K simulations.
"""

from agents import IterativeLammpsGeneration, configure_dspy_lm
import dspy
from prompts import ReviewLAMMPSScript

def test_missing_velocity_detection():
    """Test that reviewer catches missing velocity initialization in NVT simulation"""

    # Create a problematic script (like the one that caused 0K simulation)
    problematic_script = """units metal
dimension 3
boundary p p p
atom_style atomic

kim init SW_StillingerWeber_1985_Si__MO_405512056662_005 metal

lattice diamond 5.431
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 box

kim interactions Si

mass 1 28.0855

neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

# NVT ensemble WITHOUT velocity initialization - this causes 0K simulation!
fix 1 all nvt temp 300.0 300.0 0.1

thermo 100
thermo_style custom step temp pe ke etotal

timestep 0.001
run 1000"""

    # Test the DSPy reviewer
    configure_dspy_lm()
    reviewer = dspy.ChainOfThought(ReviewLAMMPSScript)
    result = reviewer(
        lammps_script=problematic_script,
        intended_parameters="Si at 300K, NVT, 1ps"
    )

    print("=== TESTING PROBLEMATIC SCRIPT (Missing Velocity) ===")
    print(f"is_valid: {result.is_valid}")
    print(f"issues: {result.issues}")
    print(f"severity: {result.severity}")
    print("="*60)

    # Check if reviewer caught the issue
    is_valid = result.is_valid
    if isinstance(is_valid, str):
        is_valid = is_valid.lower() not in ('true', 'yes')
    else:
        is_valid = not is_valid

    if is_valid:
        print("✅ SUCCESS: Reviewer correctly identified missing velocity initialization!")
        return True
    else:
        print("❌ FAILURE: Reviewer did not catch the missing velocity initialization issue!")
        return False

def test_correct_script_validation():
    """Test that reviewer passes a correct script with proper velocity initialization"""

    correct_script = """units metal
dimension 3
boundary p p p
atom_style atomic

kim init SW_StillingerWeber_1985_Si__MO_405512056662_005 metal

lattice diamond 5.431
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 box

kim interactions Si

mass 1 28.0855

# Proper velocity initialization for 300K
velocity all create 300.0 12345

neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

fix 1 all nvt temp 300.0 300.0 0.1

thermo 100
thermo_style custom step temp pe ke etotal

timestep 0.001
run 1000"""

    configure_dspy_lm()
    reviewer = dspy.ChainOfThought(ReviewLAMMPSScript)
    result = reviewer(
        lammps_script=correct_script,
        intended_parameters="Si at 300K, NVT, 1ps"
    )

    print("\n=== TESTING CORRECT SCRIPT (With Velocity) ===")
    print(f"is_valid: {result.is_valid}")
    print(f"issues: {result.issues}")
    print(f"severity: {result.severity}")
    print("="*60)

    # Check if reviewer passes the correct script
    is_valid = result.is_valid
    if isinstance(is_valid, str):
        is_valid = is_valid.lower() in ('true', 'yes')

    if is_valid:
        print("✅ SUCCESS: Reviewer correctly passed the script with proper velocity initialization!")
        return True
    else:
        print("❌ FAILURE: Reviewer incorrectly failed a correct script!")
        return False

def test_generator_improvement():
    """Test that the DSPy generator includes velocity initialization"""

    configure_dspy_lm()
    generator = IterativeLammpsGeneration(max_attempts=1, use_memory=False)

    result = generator.forward(
        material="Si",
        kim_model="SW_StillingerWeber_1985_Si__MO_405512056662_005",
        temperature=300.0,
        ensemble="NVT",
        duration_ps=1.0,
        system_size=3,
    )

    print("\n=== TESTING DSPy GENERATOR ===")
    print("Generated Script:")
    print(result.script[:500] + "..." if len(result.script) > 500 else result.script)
    print("="*60)

    # Check if generator includes velocity initialization
    if "velocity all create" in result.script and "300" in result.script:
        print("✅ SUCCESS: Generator now includes velocity initialization!")
        return True
    else:
        print("❌ FAILURE: Generator still missing velocity initialization!")
        return False

if __name__ == "__main__":
    print("Testing DSPy-Based LAMMPS Agents for Temperature Validation")
    print("="*60)

    # Run all tests
    test1_passed = test_missing_velocity_detection()
    test2_passed = test_correct_script_validation()
    test3_passed = test_generator_improvement()

    print(f"\n=== FINAL RESULTS ===")
    print(f"Missing Velocity Detection: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Correct Script Validation: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"Generator Enhancement: {'✅ PASS' if test3_passed else '❌ FAIL'}")

    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 ALL TESTS PASSED! The agents should now catch the 0K temperature issue.")
    else:
        print("\n⚠️  Some tests failed. The agents may need further refinement.")
