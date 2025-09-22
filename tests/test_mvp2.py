import pytest
import os
from tools import run_lammps_simulation, validate_lammps_script


def test_validate_lammps_script():
    """Test the LAMMPS script validation function."""
    # Read the test script
    test_script_path = os.path.join("tests", "test_assets", "in.test_lj")
    with open(test_script_path, 'r') as f:
        test_script = f.read()
    
    # Test that our test script is valid
    assert validate_lammps_script(test_script), "Test script should be valid"
    
    # Test invalid script (missing kim command)
    invalid_script = """
    units lj
    dimension 3
    boundary p p p
    """
    assert not validate_lammps_script(invalid_script), "Script without kim command should be invalid"


def test_run_lammps_simulation():
    """
    Test the LAMMPS simulation execution with real LAMMPS library.
    This is an integration test that runs a real (but small) simulation.
    """
    # Read the test script
    test_script_path = os.path.join("tests", "test_assets", "in.test_lj")
    with open(test_script_path, 'r') as f:
        test_script = f.read()
    
    # Run the simulation
    output_files, thermo_data = run_lammps_simulation(test_script)
    
    # Assertions
    # 1. Check that we got some output files
    assert isinstance(output_files, dict), "Output files should be a dictionary"
    
    # 2. Check that we got thermodynamic data
    assert isinstance(thermo_data, list), "Thermo data should be a list"
    assert len(thermo_data) > 0, "Should have at least one data point"
    
    # 3. Check that the first data point has expected structure
    first_data_point = thermo_data[0]
    assert isinstance(first_data_point, dict), "Each data point should be a dictionary"
    assert "step" in first_data_point, "Data point should contain step information"
    
    # 4. If dump file was created, check it exists
    if 'dump' in output_files:
        dump_path = output_files['dump']
        assert os.path.exists(dump_path), f"Dump file should exist at {dump_path}"
        
        # Clean up the dump file after test
        try:
            os.remove(dump_path)
        except:
            pass
    
    # 5. Clean up any log files
    if 'log' in output_files:
        log_path = output_files['log']
        try:
            os.remove(log_path)
        except:
            pass
    
    print(f"Test completed successfully. Got {len(thermo_data)} data points.")
    print(f"Output files: {list(output_files.keys())}")
    print(f"Sample data point: {first_data_point}")


if __name__ == "__main__":
    # Allow running this test directly
    test_validate_lammps_script()
    test_run_lammps_simulation()
    print("All MVP 2 tests passed!")
