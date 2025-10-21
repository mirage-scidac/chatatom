#!/usr/bin/env python3
"""
Test script to verify run folder functionality.
"""
import run_manager
import os

def test_run_folder_creation():
    """Test that run folders are created correctly."""
    print("Testing run folder creation...")
    
    # Create a run folder
    run_folder = run_manager.create_run_folder()
    print(f"✓ Created run folder: {run_folder}")
    
    # Verify structure
    assert os.path.exists(run_folder), "Run folder should exist"
    assert os.path.exists(os.path.join(run_folder, "frames")), "Frames subfolder should exist"
    print("✓ Run folder structure verified")
    
    # Test saving LAMMPS script
    test_script = "# Test LAMMPS script\nunits metal\n"
    script_path = run_manager.save_lammps_script(run_folder, test_script)
    assert os.path.exists(script_path), "LAMMPS script should be saved"
    print(f"✓ LAMMPS script saved: {script_path}")
    
    # Test saving OVITO script
    test_ovito = "# Test OVITO script\nfrom ovito.modifiers import *\n"
    ovito_path = run_manager.save_ovito_script(run_folder, test_ovito)
    assert os.path.exists(ovito_path), "OVITO script should be saved"
    print(f"✓ OVITO script saved: {ovito_path}")
    
    # Test saving metadata
    metadata = {
        'test': True,
        'num_cores': 4,
        'material': 'Al'
    }
    run_manager.save_simulation_metadata(run_folder, metadata)
    metadata_path = os.path.join(run_folder, "simulation_info.json")
    assert os.path.exists(metadata_path), "Metadata should be saved"
    print(f"✓ Metadata saved: {metadata_path}")
    
    # Test updating status
    run_manager.update_simulation_status(run_folder, 'completed', frames=100)
    print("✓ Status updated")
    
    # Test getting summary
    summary = run_manager.get_run_folder_summary(run_folder)
    print(f"✓ Run folder summary:")
    print(f"  - Has input script: {summary['has_input_script']}")
    print(f"  - Has OVITO script: {summary['has_ovito_script']}")
    print(f"  - Frame count: {summary['frame_count']}")
    print(f"  - Status: {summary['metadata']['status']}")
    
    print(f"\n✅ All tests passed!")
    print(f"\nRun folder location: {run_folder}")
    print("You can inspect the folder to see the organized structure.")

if __name__ == "__main__":
    test_run_folder_creation()
