#!/usr/bin/env python3
"""
Test script to verify that the enhanced LammpsReviewerAgent can catch
the missing velocity initialization issue that causes 0K simulations.
"""

from agents import LammpsGeneratorAgent, LammpsReviewerAgent

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

    # Test the reviewer
    reviewer = LammpsReviewerAgent()
    review_result = reviewer.execute(problematic_script)
    
    print("=== TESTING PROBLEMATIC SCRIPT (Missing Velocity) ===")
    print("Script:")
    print(problematic_script)
    print("\n" + "="*60)
    print("Reviewer Response:")
    print(review_result)
    print("="*60)
    
    # Check if reviewer caught the issue
    if "FAIL" in review_result and ("velocity" in review_result.lower() or "0k" in review_result.lower()):
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

    reviewer = LammpsReviewerAgent()
    review_result = reviewer.execute(correct_script)
    
    print("\n=== TESTING CORRECT SCRIPT (With Velocity) ===")
    print("Reviewer Response:")
    print(review_result)
    print("="*60)
    
    # Check if reviewer passes the correct script
    if "PASS" in review_result:
        print("✅ SUCCESS: Reviewer correctly passed the script with proper velocity initialization!")
        return True
    else:
        print("❌ FAILURE: Reviewer incorrectly failed a correct script!")
        return False

def test_generator_improvement():
    """Test that the enhanced generator now includes velocity initialization"""
    
    generator = LammpsGeneratorAgent()
    user_request = "Create a LAMMPS script for silicon at 300K using NVT ensemble"
    
    generated_script = generator.execute(user_request)
    
    print("\n=== TESTING ENHANCED GENERATOR ===")
    print("User Request:", user_request)
    print("\nGenerated Script:")
    print(generated_script)
    print("="*60)
    
    # Check if generator includes velocity initialization
    if "velocity all create" in generated_script and "300" in generated_script:
        print("✅ SUCCESS: Generator now includes velocity initialization!")
        return True
    else:
        print("❌ FAILURE: Generator still missing velocity initialization!")
        return False

if __name__ == "__main__":
    print("Testing Enhanced LAMMPS Agents for Temperature Validation")
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
