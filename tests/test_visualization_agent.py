#!/usr/bin/env python3
"""
Test script to verify visualization agent integration.
"""

import sys
import json
from agents import OrchestratorAgent

def test_visualization_parsing():
    """Test that visualization requests are properly parsed."""
    
    print("Testing Visualization Agent Integration")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = OrchestratorAgent()
    
    # Test cases
    test_cases = [
        {
            'request': 'show only crystalline atoms',
            'expected_mode': 'script',
            'description': 'Filter to crystalline atoms'
        },
        {
            'request': 'visualize the defects',
            'expected_mode': 'script',
            'description': 'Highlight defects'
        },
        {
            'request': 'color by structure type',
            'expected_mode': 'script',
            'description': 'Color coding by structure'
        },
        {
            'request': 'run a simulation',
            'expected_mode': 'config',
            'description': 'No visualization request (should use basic config)'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"Request: '{test['request']}'")
        
        try:
            result = orchestrator.parse_visualization_request(
                test['request'],
                {'material': 'Cu', 'structure': 'FCC'}
            )
            
            print(f"Mode: {result.get('mode')}")
            print(f"Expected: {test['expected_mode']}")
            
            if result.get('mode') == test['expected_mode']:
                print("✅ PASS")
            else:
                print("❌ FAIL - Mode mismatch")
            
            if result.get('rationale'):
                print(f"Rationale: {result['rationale'][:100]}...")
            
            if result.get('script'):
                print(f"Script generated: {len(result['script'])} characters")
            
            if result.get('warnings'):
                print(f"Warnings: {result['warnings']}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Visualization Agent Integration Test Complete")

if __name__ == '__main__':
    test_visualization_parsing()
