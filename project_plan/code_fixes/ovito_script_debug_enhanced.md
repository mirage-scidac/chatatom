# Enhanced OVITO Script Debug Plan

## Debug Logging Added

We've added comprehensive debug logging at all key points in the OVITO script pipeline to identify where the script is being lost.

### 1. WebSocket Handler (app.py - handle_streaming_simulation)

**Location**: Lines 179-186

**Debug Messages**:
```
[APP DEBUG] About to parse visualization request
[APP DEBUG] user_command: <command>
[APP DEBUG] simulation_context: <context>
[APP DEBUG] Generated OVITO script: True/False (X chars)
[APP DEBUG] Script preview: <first 200 chars>
[APP DEBUG] No OVITO script generated (returned None)
[APP DEBUG] Render config created with ovito_script: True/False
```

### 2. Tools Module (tools.py - run_lammps_simulation)

**Location**: Lines 1084-1092

**Debug Messages**:
```
[TOOLS DEBUG] render_config type: <type>
[TOOLS DEBUG] render_config keys: <keys>
[TOOLS DEBUG] Extracted ovito_script from render_config: True/False (X chars)
[TOOLS DEBUG] Script preview: <first 200 chars>
[TOOLS DEBUG] ovito_script is None or empty
[TOOLS DEBUG] Starting rendering worker with ovito_script: True/False
```

### 3. Rendering Module (visualization_rendering.py)

**Location**: Line 47

**Debug Messages**:
```
[RENDER DEBUG] Called with ovito_script: True/False (X chars)
```

## Expected Debug Flow for Successful Script Execution

When a custom visualization is requested (e.g., "Color atoms by atom type, FCC, BCC etc. in Ovito"), you should see:

```
[APP DEBUG] About to parse visualization request
[APP DEBUG] user_command: Simulate melting and recrystallization of Al...
[APP DEBUG] simulation_context: {'material': 'Al'}
[APP DEBUG] Generated OVITO script: True (XXX chars)
[APP DEBUG] Script preview: from ovito.io import import_file...
[APP DEBUG] Render config created with ovito_script: True
[TOOLS DEBUG] render_config type: <class 'dict'>
[TOOLS DEBUG] render_config keys: dict_keys(['ovito_script', 'target_timesteps'])
[TOOLS DEBUG] Extracted ovito_script from render_config: True (XXX chars)
[TOOLS DEBUG] Script preview: from ovito.io import import_file...
[TOOLS DEBUG] Starting rendering worker with ovito_script: True
[RENDER DEBUG] Called with ovito_script: True (XXX chars)
```

## Current Issue

Based on the console output provided, we're seeing:
```
[RENDER DEBUG] Called with ovito_script: False (0 chars)
```

But NOT seeing the APP DEBUG or TOOLS DEBUG messages, which suggests:

1. **Possible Issue #1**: The WebSocket handler might not be executing the visualization parsing code
2. **Possible Issue #2**: The `parse_visualization_request()` function is returning `None`
3. **Possible Issue #3**: The render_config is not being created or passed correctly

## What to Look For in Next Test

When you run the simulation again, watch for:

1. **Do you see APP DEBUG messages?**
   - If NO: The WebSocket handler isn't executing the visualization code
   - If YES but script is None: The Visualization Agent isn't generating the script

2. **Do you see TOOLS DEBUG messages?**
   - If NO: The render_config isn't being passed to run_lammps_simulation
   - If YES but script is None: The script was lost between app.py and tools.py

3. **What does render_config look like?**
   - Check the keys - should include 'ovito_script'
   - Check if ovito_script value is None or a string

## Next Steps Based on Debug Output

### If APP DEBUG shows script generated but TOOLS DEBUG shows None:
- Issue is in how render_config is passed to run_lammps_simulation
- Check the WebSocket handler's call to run_lammps_simulation

### If APP DEBUG shows script is None:
- Issue is in parse_visualization_request or the Visualization Agent
- Need to debug why the agent isn't generating the script

### If no APP DEBUG messages appear:
- Issue is that the WebSocket handler isn't being called
- Or the visualization parsing code isn't executing
- Check if the frontend is using the correct endpoint

## Testing Command

Use the same command that failed before:
```
Simulate melting and recrystallization of Al in 100 ps at 1 atm. 
Heat to 4000K. Use 8000 atoms and longer on 1 side (horizontal). 
Equilibrate for 10 ps first. Hold at 300K at the end for 10 ps.
Color atoms by atom type, FCC, BCC etc. in Ovito
```

The phrase "Color atoms by atom type, FCC, BCC etc. in Ovito" should trigger custom visualization script generation.
