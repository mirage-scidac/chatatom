"""
Test script for dynamic OVITO visualization system.

This script tests the DSPy-based script generation approach for
visualizing "only crystalline atoms" and other complex requests.

Tests the architecture where SmartOrchestrator handles visualization
detection and OVITOScriptGeneration handles script creation.
"""

from agents import SmartOrchestrator, IterativeOVITOGeneration, configure_dspy_lm
from ovito_executor import DynamicOVITOExecutor


def test_script_generation():
    """Test iterative OVITO script generation with review for various requests."""

    print("=" * 60)
    print("Testing Iterative OVITO Script Generation (DSPy)")
    print("=" * 60)

    # Initialize components
    configure_dspy_lm()
    generator = IterativeOVITOGeneration(max_attempts=3)
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
            # Generate script via iterative DSPy module
            result = generator.forward(request, simulation_context)

            print(f"Review passed: {result.review_passed}")
            print(f"Attempts: {result.attempts}")
            if result.issues:
                print(f"Issues: {result.issues}")

            if result.script:
                print("Generated script:")
                print(result.script)
                print()

                # Validate script
                try:
                    executor.validate_and_sanitize(result.script)
                    print("✅ Script validation: PASSED")
                except Exception as e:
                    print(f"❌ Script validation: FAILED - {e}")
            else:
                print("❌ No script generated (all attempts failed review)")

        except Exception as e:
            print(f"❌ Script generation failed: {e}")

    print("\n" + "=" * 60)


def test_visualization_detection():
    """Test the SmartOrchestrator visualization detection."""

    print("\n" + "=" * 60)
    print("Testing SmartOrchestrator Visualization Detection")
    print("=" * 60)

    configure_dspy_lm()
    orchestrator = SmartOrchestrator()

    test_requests = [
        ("Show only crystalline atoms", True),
        ("Color by temperature", True),
        ("Run a simulation of copper", False),
    ]

    for i, (request, expected) in enumerate(test_requests, 1):
        print(f"\n{i}. Testing request: '{request}'")
        print("-" * 60)

        result = orchestrator.detect_visualization(request)

        print(f"Detected custom viz: {result}")
        print(f"Expected: {expected}")

        if result == expected:
            print("✅ PASS")
        else:
            print("❌ FAIL")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DYNAMIC OVITO VISUALIZATION SYSTEM TEST (DSPy)")
    print("=" * 60)

    try:
        # Test 1: Script generation
        test_script_generation()

        # Test 2: Visualization detection
        test_visualization_detection()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
