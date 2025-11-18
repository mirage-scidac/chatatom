# Dynamic Visualization Refactoring

**Date**: 2025-10-07  
**Status**: Completed

## Overview

Refactored the visualization system to use only dynamic OVITO script generation, removing the dual-path confusion between static and dynamic visualization modes.

## Problems Addressed

### 1. Dual Execution Paths
The system previously had two parallel visualization approaches:
- **Static Mode**: Hardcoded logic in `apply_static_visualization()` with predefined checks
- **Dynamic Mode**: LLM-generated OVITO scripts from the Visualization Agent

This created confusion and maintenance issues, with the static path often being used even when good dynamic scripts were generated.

### 2. Frame Limit Hardcoded
The Orchestrator agent's system prompt mentioned "typically 10" frames, causing it to ignore the configured `MAX_TRAJECTORY_FRAMES = 20` value.

### 3. CNA Coloring Not Working
The static visualization path didn't properly handle pre-computed CNA values from LAMMPS dump files, even though the dynamic scripts were correctly designed to use them.

## Changes Made

### 1. Simplified `visualization_rendering.py`

**Removed**:
- `apply_static_visualization()` function with hardcoded CNA/CSP logic
- Complex nested config extraction logic
- Multiple fallback paths that prevented dynamic scripts from executing

**Kept**:
- `apply_basic_visualization()` as a simple fallback for when scripts fail
- Dynamic script execution as the primary path
- Proper error handling and logging

**New Behavior**:
```python
if mode == 'script' and script exists:
    Execute dynamic OVITO script
    If fails → fallback to basic visualization
else:
    Use basic visualization (simple particle type coloring)
```

### 2. Updated Orchestrator Agent Prompt

**Changed in `agents.py`**:
```python
# OLD:
2. Use config.MAX_TRAJECTORY_FRAMES (typically 10) as the target number of frames

# NEW:
2. Import config and use config.MAX_TRAJECTORY_FRAMES as the target number of frames

IMPORTANT: You MUST import config at the start of your calculation to get the actual MAX_TRAJECTORY_FRAMES value.
Do NOT assume or hardcode the number of frames - always use the configured value.
```

**Updated Example**:
- Old: 10 frames hardcoded in example
- New: 20 frames using config.MAX_TRAJECTORY_FRAMES

## Benefits

### 1. Single Source of Truth
The Visualization Agent is now the only source for visualization logic. No more confusion about which path is being used.

### 2. Proper CNA Handling
Dynamic scripts correctly detect and use pre-computed CNA values from LAMMPS:
```python
# Generated script properly maps c_cna to Structure Type
mapper = ComputePropertyModifier()
mapper.output_property = 'Structure Type'
mapper.expressions = ['c_cna']  # Use pre-computed values
pipeline.modifiers.append(mapper)
```

### 3. Configurable Frame Limits
The system now respects `config.MAX_TRAJECTORY_FRAMES` instead of hardcoding 10 frames.

### 4. Cleaner Codebase
- Removed ~80 lines of unused static visualization code
- Simplified execution logic
- Easier to maintain and debug

## Testing Recommendations

1. **Test CNA Coloring**:
   - Run a simulation with structure-based visualization request
   - Verify atoms are colored by CNA structure type (FCC, BCC, HCP, etc.)
   - Check terminal output for "[RENDER] Executing dynamic OVITO script"

2. **Test Frame Limit**:
   - Verify simulations capture 20 frames (or whatever MAX_TRAJECTORY_FRAMES is set to)
   - Check terminal output for correct number of target timesteps

3. **Test Fallback**:
   - Intentionally break a visualization script
   - Verify system falls back to basic visualization without crashing

## Migration Notes

### For Users
No changes needed - the system now works more intuitively with visualization requests.

### For Developers
- Remove any references to `apply_static_visualization()`
- All visualization customization should go through the Visualization Agent
- Update tests to expect dynamic script execution

## Related Files

- `visualization_rendering.py` - Simplified rendering logic
- `agents.py` - Updated Orchestrator prompt for frame sampling
- `config.py` - MAX_TRAJECTORY_FRAMES now properly used
- `tools.py` - Calls visualization with dynamic scripts

## Future Improvements

1. Add validation for generated OVITO scripts before execution
2. Implement caching for frequently used visualization patterns
3. Add user-facing visualization templates
4. Improve error messages when scripts fail
