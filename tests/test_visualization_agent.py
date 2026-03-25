#!/usr/bin/env python3
"""
Test script to verify visualization agent integration using DSPy modules.
"""

import sys
import json
from agents import SmartOrchestrator, OVITOScriptGeneration, configure_dspy_lm

def test_visualization_parsing():
    """Test that visualization requests are properly detected and scripts generated."""

    print("Testing Visualization Agent Integration (DSPy)")
    print("=" * 60)

    # Configure DSPy
    configure_dspy_lm()
    orchestrator = SmartOrchestrator()
    ovito_gen = OVITOScriptGeneration()

    # Test cases
    test_cases = [
        {
            'request': 'show only crystalline atoms',
            'expected_custom': True,
            'description': 'Filter to crystalline atoms'
        },
        {
            'request': 'visualize the defects',
            'expected_custom': True,
            'description': 'Highlight defects'
        },
        {
            'request': 'color by structure type',
            'expected_custom': True,
            'description': 'Color coding by structure'
        },
        {
            'request': 'run a simulation of copper at 300K',
            'expected_custom': False,
            'description': 'No visualization request (should use default)'
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"Request: '{test['request']}'")

        try:
            is_custom = orchestrator.detect_visualization(test['request'])

            print(f"Custom viz detected: {is_custom}")
            print(f"Expected custom: {test['expected_custom']}")

            if is_custom == test['expected_custom']:
                print("✅ PASS")
            else:
                print("❌ FAIL - Detection mismatch")

            if is_custom:
                script = ovito_gen.forward(
                    test['request'],
                    {'material': 'Cu', 'structure': 'FCC'}
                )
                print(f"Script generated: {len(script)} characters")

        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Visualization Agent Integration Test Complete")

if __name__ == '__main__':
    test_visualization_parsing()
