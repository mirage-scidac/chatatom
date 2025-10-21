"""
Test script for dynamic OVITO visualization system.

This script tests the dynamic script generation approach for
visualizing "only crystalline atoms" and other complex requests.

Tests the new architecture where OrchestratorAgent handles
visualization parsing.
"""

from agents import OrchestratorAgent
from ovito_script_generator import OVITOScriptGenerator
from ovito_executor import DynamicOVITOExecutor


def test_script_generation():
    """Test OVITO script generation for various requests."""
    
    print("=" * 60)
    print("Testing Dynamic OVITO Script Generation")
    print("=" * 60)
    
    # Initialize components
    generator = OVITOScriptGenerator()
    executor = DynamicOVITOExecutor()
    
    # Test cases
    test_requests = [
        "Show only crystalline atoms",
        "Highlight defects using centro-symmetry",
        "Show only FCC atoms colored by structure type",
        "Visualize surface atoms",
    ]
    
    simulation_context = {
        'material': 'Cu',
        'natoms': 10000,
        'temperature': 300,
        'structure': 'FCC'
    }
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n{i}. Testing request: '{request}'")
        print("-" * 60)
        
        try:
            # Generate script
            script = generator.generate_ovito_script(request, simulation_context)
            
            print("Generated script:")
            print(script)
            print()
            
            # Validate script
            try:
                executor.validate_and_sanitize(script)
                print("✅ Script validation: PASSED")
            except Exception as e:
                print(f"❌ Script validation: FAILED - {e}")
        
        except Exception as e:
            print(f"❌ Script generation failed: {e}")
    
    print("\n" + "=" * 60)


def test_orchestrator_agent():
    """Test the OrchestratorAgent with visualization parsing."""
    
    print("\n" + "=" * 60)
    print("Testing OrchestratorAgent (New Architecture)")
    print("=" * 60)
    
    orchestrator = OrchestratorAgent()
    
    test_requests = [
        "Show only crystalline atoms",
        "Color by temperature",
    ]
    
    simulation_context = {
        'material': 'Al',
        'natoms': 5000,
        'temperature': 500,
        'structure': 'FCC'
    }
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n{i}. Testing request: '{request}'")
        print("-" * 60)
        
        # Test script mode using OrchestratorAgent
        result = orchestrator.parse_visualization_request(
            request,
            simulation_context
        )
        
        print(f"Mode: {result['mode']}")
        print(f"Rationale: {result['rationale']}")
        
        if result.get('warnings'):
            print(f"Warnings: {result['warnings']}")
        
        if result.get('script'):
            print("\nGenerated script:")
            print(result['script'][:200] + "..." if len(result['script']) > 200 else result['script'])
            print("✅ Script mode: SUCCESS")
        else:
            print("❌ Script mode: FAILED")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DYNAMIC OVITO VISUALIZATION SYSTEM TEST")
    print("=" * 60)
    
    try:
        # Test 1: Script generation
        test_script_generation()
        
        # Test 2: OrchestratorAgent (new architecture)
        test_orchestrator_agent()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
