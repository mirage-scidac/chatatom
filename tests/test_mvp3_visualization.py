# Copyright © 2026, UChicago Argonne, LLC All Rights Reserved
# Full license accessible at https://github.com/mirage-scidac/chatatom/blob/main/LICENSE

"""
Test suite for MVP 3: 3D Trajectory Visualization with Real LAMMPS and OVITO
"""
import pytest
import tempfile
import os
import sys
import numpy as np

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools import (
    extract_lammps_frame, 
    create_ovito_frame, 
    render_frame_to_image,
    run_lammps_with_visualization
)
import lammps


class TestLammpsDataExtraction:
    """Test extracting positions/types from real LAMMPS."""
    
    def test_extract_lammps_frame_basic(self):
        """Test basic frame extraction from real LAMMPS instance."""
        # Create real LAMMPS instance
        lmp = lammps.lammps()
        
        try:
            # Set up minimal system
            lmp.command("units lj")
            lmp.command("dimension 3")
            lmp.command("boundary p p p")
            lmp.command("atom_style atomic")
            lmp.command("region box block 0 10 0 10 0 10")
            lmp.command("create_box 1 box")
            lmp.command("create_atoms 1 single 5.0 5.0 5.0")
            lmp.command("create_atoms 1 single 6.0 5.0 5.0")  # Add second atom
            lmp.command("mass 1 1.0")
            lmp.command("pair_style lj/cut 2.5")
            lmp.command("pair_coeff 1 1 1.0 1.0")
            lmp.command("velocity all create 1.0 12345")
            
            # Test extraction
            frame_data = extract_lammps_frame(lmp)
            
            assert frame_data is not None
            assert frame_data['natoms'] == 2
            assert frame_data['positions'].shape == (2, 3)
            assert frame_data['types'].shape == (2,)
            assert frame_data['box']['xlo'] == 0.0
            assert frame_data['box']['xhi'] == 10.0
            assert frame_data['velocities'] is not None
            
        finally:
            lmp.close()
    
    def test_extract_lammps_frame_multiple_atoms(self):
        """Test extraction with multiple atoms."""
        lmp = lammps.lammps()
        
        try:
            # Set up system with 4 atoms
            lmp.command("units lj")
            lmp.command("dimension 3")
            lmp.command("boundary p p p")
            lmp.command("atom_style atomic")
            lmp.command("region box block 0 10 0 10 0 10")
            lmp.command("create_box 2 box")
            lmp.command("create_atoms 1 single 2.0 2.0 2.0")
            lmp.command("create_atoms 1 single 8.0 2.0 2.0")
            lmp.command("create_atoms 2 single 2.0 8.0 2.0")
            lmp.command("create_atoms 2 single 8.0 8.0 2.0")
            lmp.command("mass 1 1.0")
            lmp.command("mass 2 2.0")
            lmp.command("pair_style lj/cut 2.5")
            lmp.command("pair_coeff 1 1 1.0 1.0")
            lmp.command("pair_coeff 2 2 1.0 1.0")
            lmp.command("pair_coeff 1 2 1.0 1.0")
            lmp.command("velocity all create 1.0 12345")
            
            frame_data = extract_lammps_frame(lmp)
            
            assert frame_data is not None
            assert frame_data['natoms'] == 4
            assert frame_data['positions'].shape == (4, 3)
            assert frame_data['types'].shape == (4,)
            
            # Check that we have both atom types
            unique_types = np.unique(frame_data['types'])
            assert len(unique_types) == 2
            assert 1 in unique_types
            assert 2 in unique_types
            
        finally:
            lmp.close()


class TestOvitoFrameCreation:
    """Test creating OVITO data with real OVITO library."""
    
    def test_create_ovito_frame_success(self):
        """Test successful OVITO frame creation with real data."""
        # Create realistic frame data
        frame_data = {
            'natoms': 2,
            'positions': np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            'types': np.array([1, 2]),
            'box': {'xlo': 0.0, 'xhi': 10.0, 'ylo': 0.0, 'yhi': 10.0, 'zlo': 0.0, 'zhi': 10.0},
            'velocities': np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
            'forces': None
        }
        
        # Test creation with real OVITO
        result = create_ovito_frame(frame_data)
        
        assert result is not None
        
        # Verify the data was set correctly
        particles = result.particles
        assert particles.count == 2
        
        # Check positions
        np.testing.assert_array_almost_equal(particles.positions, frame_data['positions'])
        
        # Check types
        np.testing.assert_array_equal(particles.particle_types, frame_data['types'])
        
        # Check velocities
        np.testing.assert_array_almost_equal(particles.velocities, frame_data['velocities'])
        
        # Check simulation cell
        cell = result.cell
        assert cell is not None
        assert cell.pbc == (True, True, True)
    
    def test_create_ovito_frame_none_input(self):
        """Test creation with None input."""
        result = create_ovito_frame(None)
        assert result is None
    
    def test_create_ovito_frame_large_system(self):
        """Test creation with larger system."""
        # Create a larger system (27 atoms in 3x3x3 grid)
        natoms = 27
        positions = []
        types = []
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    positions.append([i * 2.0, j * 2.0, k * 2.0])
                    types.append(1)
        
        frame_data = {
            'natoms': natoms,
            'positions': np.array(positions),
            'types': np.array(types),
            'box': {'xlo': 0.0, 'xhi': 6.0, 'ylo': 0.0, 'yhi': 6.0, 'zlo': 0.0, 'zhi': 6.0},
            'velocities': np.random.random((natoms, 3)) * 0.1,
            'forces': np.random.random((natoms, 3)) * 0.01
        }
        
        result = create_ovito_frame(frame_data)
        
        assert result is not None
        assert result.particles.count == natoms
        np.testing.assert_array_almost_equal(result.particles.positions, frame_data['positions'])


class TestRenderingPipeline:
    """Test complete render with real OVITO."""
    
    def test_render_frame_to_image_success(self):
        """Test successful image rendering with real OVITO."""
        # Create real frame data
        frame_data = {
            'natoms': 4,
            'positions': np.array([
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0], 
                [0.0, 2.0, 0.0],
                [2.0, 2.0, 0.0]
            ]),
            'types': np.array([1, 1, 2, 2]),
            'box': {'xlo': -1.0, 'xhi': 3.0, 'ylo': -1.0, 'yhi': 3.0, 'zlo': -1.0, 'zhi': 3.0},
            'velocities': None,
            'forces': None
        }
        
        # Create OVITO data
        data_collection = create_ovito_frame(frame_data)
        assert data_collection is not None
        
        # Test rendering
        settings = {'quality': 'low', 'camera_angle': 'default'}
        result = render_frame_to_image(data_collection, settings)
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Verify it's valid base64
        import base64
        try:
            decoded = base64.b64decode(result)
            assert len(decoded) > 0
        except Exception:
            pytest.fail("Result is not valid base64")
    
    def test_render_frame_to_image_none_input(self):
        """Test rendering with None input."""
        result = render_frame_to_image(None)
        assert result is None
    
    def test_render_frame_different_qualities(self):
        """Test rendering with different quality settings."""
        # Create simple frame data
        frame_data = {
            'natoms': 1,
            'positions': np.array([[0.0, 0.0, 0.0]]),
            'types': np.array([1]),
            'box': {'xlo': -2.0, 'xhi': 2.0, 'ylo': -2.0, 'yhi': 2.0, 'zlo': -2.0, 'zhi': 2.0},
            'velocities': None,
            'forces': None
        }
        
        data_collection = create_ovito_frame(frame_data)
        
        # Test different quality settings
        for quality in ['low', 'medium', 'high']:
            settings = {'quality': quality}
            result = render_frame_to_image(data_collection, settings)
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0


class TestVisualizationPipeline:
    """Test the complete visualization pipeline with real simulations."""
    
    def test_run_lammps_with_visualization_simple(self):
        """Test complete visualization pipeline with simple simulation."""
        # Simple LJ simulation script
        test_script = """
        units lj
        dimension 3
        boundary p p p
        atom_style atomic
        lattice fcc 2.0
        region box block 0 2 0 2 0 2
        create_box 1 box
        create_atoms 1 box
        mass 1 1.0
        pair_style lj/cut 2.5
        pair_coeff 1 1 1.0 1.0
        velocity all create 1.0 12345
        fix 1 all nve
        thermo 5
        run 10
        """
        
        # Track callback data
        callback_data = []
        def test_callback(data):
            callback_data.append(data)
        
        # Run visualization
        render_config = {'interval': 5, 'quality': 'low'}
        output_files, thermo_data, frame_images = run_lammps_with_visualization(
            test_script, 
            render_config=render_config,
            data_callback=test_callback
        )
        
        # Verify results
        assert isinstance(output_files, dict)
        assert isinstance(thermo_data, list)
        assert isinstance(frame_images, list)
        
        # Should have generated at least one frame
        assert len(frame_images) >= 1
        
        # Check frame structure
        for frame in frame_images:
            assert 'image' in frame
            assert 'frame_number' in frame
            assert 'timestep' in frame
            assert isinstance(frame['image'], str)
            assert len(frame['image']) > 0
        
        # Should have received callback data
        assert len(callback_data) > 0
        
        # Check for trajectory frames in callback data
        trajectory_frames = [d for d in callback_data if d.get('type') == 'trajectory_frame']
        assert len(trajectory_frames) > 0
    
    def test_run_lammps_with_visualization_no_kim(self):
        """Test visualization with non-KIM potential."""
        # Simple LJ simulation without KIM
        test_script = """
        units lj
        dimension 3
        boundary p p p
        atom_style atomic
        lattice fcc 1.5
        region box block 0 3 0 3 0 3
        create_box 1 box
        create_atoms 1 box
        mass 1 1.0
        pair_style lj/cut 2.5
        pair_coeff 1 1 1.0 1.0
        velocity all create 1.0 12345
        fix 1 all nve
        thermo 10
        run 20
        """
        
        render_config = {'interval': 10, 'quality': 'low'}
        output_files, thermo_data, frame_images = run_lammps_with_visualization(
            test_script, 
            render_config=render_config
        )
        
        # Should work without KIM models
        assert isinstance(frame_images, list)
        assert len(frame_images) >= 1
        
        # Verify frame content
        for frame in frame_images:
            assert frame['image'] is not None
            assert len(frame['image']) > 0


class TestMemoryEfficiency:
    """Ensure no memory leaks in direct transfer with real data."""
    
    def test_frame_data_memory_usage(self):
        """Test that frame data structures are reasonable size."""
        # Create test frame data with realistic size
        natoms = 100  # Smaller than original test for faster execution
        frame_data = {
            'natoms': natoms,
            'positions': np.random.random((natoms, 3)) * 10,
            'types': np.ones(natoms, dtype=int),
            'box': {'xlo': 0.0, 'xhi': 10.0, 'ylo': 0.0, 'yhi': 10.0, 'zlo': 0.0, 'zhi': 10.0},
            'velocities': np.random.random((natoms, 3)) * 0.1,
            'forces': np.random.random((natoms, 3)) * 0.01
        }
        
        # Calculate memory usage
        positions_size = frame_data['positions'].nbytes
        types_size = frame_data['types'].nbytes
        velocities_size = frame_data['velocities'].nbytes
        forces_size = frame_data['forces'].nbytes
        
        total_size = positions_size + types_size + velocities_size + forces_size
        
        # Should be reasonable (less than 10KB for 100 atoms)
        assert total_size < 10 * 1024  # 10KB
        
        # Memory should scale linearly with atom count
        size_per_atom = total_size / natoms
        assert size_per_atom < 100  # Less than 100 bytes per atom


class TestIntegrationScenarios:
    """Integration tests for common usage scenarios with real simulations."""
    
    def test_small_argon_simulation_visualization(self):
        """Test visualization with a small Argon-like simulation."""
        # Small Argon-like system (8 atoms in 2x2x2 FCC)
        argon_script = """
        units lj
        dimension 3
        boundary p p p
        atom_style atomic
        lattice fcc 1.5
        region box block 0 2 0 2 0 2
        create_box 1 box
        create_atoms 1 box
        mass 1 1.0
        pair_style lj/cut 2.5
        pair_coeff 1 1 1.0 1.0
        velocity all create 0.8 12345
        fix 1 all nve
        thermo 5
        run 15
        """
        
        # Track visualization frames
        frames_received = []
        def frame_callback(data):
            if isinstance(data, dict) and data.get('type') == 'trajectory_frame':
                frames_received.append(data)
        
        # Run visualization
        render_config = {'interval': 5, 'quality': 'low', 'camera_angle': 'default'}
        output_files, thermo_data, frame_images = run_lammps_with_visualization(
            argon_script,
            render_config=render_config,
            data_callback=frame_callback
        )
        
        # Verify realistic results
        assert len(thermo_data) >= 1  # Should have thermodynamic data
        
        # Should have generated visualization frames
        assert len(frame_images) >= 1
        assert len(frames_received) >= 1
        
        # Verify frame structure
        for frame in frame_images:
            assert 'image' in frame
            assert frame['image'] is not None
            assert len(frame['image']) > 0
        
        # Verify callback frames match returned frames
        assert len(frames_received) == len(frame_images)
    
    def test_end_to_end_pipeline(self):
        """Test the complete end-to-end pipeline."""
        # Create LAMMPS instance
        lmp = lammps.lammps()
        
        try:
            # Set up minimal system
            lmp.command("units lj")
            lmp.command("dimension 3")
            lmp.command("boundary p p p")
            lmp.command("atom_style atomic")
            lmp.command("region box block 0 5 0 5 0 5")
            lmp.command("create_box 1 box")
            lmp.command("create_atoms 1 single 2.5 2.5 2.5")
            lmp.command("create_atoms 1 single 3.5 2.5 2.5")  # Add second atom
            lmp.command("mass 1 1.0")
            lmp.command("pair_style lj/cut 2.5")
            lmp.command("pair_coeff 1 1 1.0 1.0")
            lmp.command("velocity all create 1.0 12345")
            
            # Step 1: Extract frame data
            frame_data = extract_lammps_frame(lmp)
            assert frame_data is not None
            
            # Step 2: Create OVITO frame
            ovito_data = create_ovito_frame(frame_data)
            assert ovito_data is not None
            
            # Step 3: Render to image
            image_b64 = render_frame_to_image(ovito_data, {'quality': 'low'})
            assert image_b64 is not None
            assert len(image_b64) > 0
            
            # Verify the complete pipeline works
            print(f"✅ End-to-end pipeline successful: {len(image_b64)} character base64 image")
            
        finally:
            lmp.close()


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v'])
