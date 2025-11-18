# OVITO Script Execution Debug Plan

## Problem
Custom OVITO scripts are being generated but not executed. The system falls back to default visualization instead.

## Debug Logging Added

We've added strategic debug logging at each step of the pipeline to trace where the OVITO script might be getting lost:

### 1. app.py (WebSocket Handler)
**Location**: `handle_streaming_simulation` function, around line 179-180

**Debug Output**:
```
[APP DEBUG] Generated OVITO script: True/False (X chars)
[APP DEBUG] Script preview: <first 200 chars>...
[APP DEBUG] Render config created with ovito_script: True/False
```

**What to check**: 
- Is the script being generated? (should be True with >0 chars)
- Does the render_config contain the script?

### 2. tools.py (run_lammps_simulation)
**Location**: Around line 1084

**Debug Output**:
```
[TOOLS DEBUG] Extracted ovito_script from render_config: True/False (X chars)
[TOOLS DEBUG] Starting rendering worker with ovito_script: True/False
```

**What to check**:
- Is the script successfully extracted from render_config?
- Is it being passed to the rendering worker?

### 3. visualization_rendering.py (render_frame_to_image_custom)
**Location**: First line of function

**Debug Output**:
```
[RENDER DEBUG] Called with ovito_script: True/False (X chars)
```

**What to check**:
- Does the rendering function receive the script?
- If it receives None, the problem is in the parameter passing

## Expected Flow

When everything works correctly, you should see:

```
[APP DEBUG] Generated OVITO script: True (XXX chars)
[APP DEBUG] Script preview: from ovito.modifiers import...
[APP DEBUG] Render config created with ovito_script: True
[TOOLS DEBUG] Extracted ovito_script from render_config: True (XXX chars)
[TOOLS DEBUG] Starting rendering worker with ovito_script: True
[RENDER DEBUG] Called with ovito_script: True (XXX chars)
[RENDER] Executing custom OVITO script (XXX chars)
[RENDER] Custom OVITO script executed successfully
```

## Possible Failure Points

### Scenario 1: Script Not Generated
```
[APP DEBUG] Generated OVITO script: False (0 chars)
```
**Problem**: Visualization Agent not generating script
**Solution**: Check parse_visualization_request in tools.py

### Scenario 2: Script Lost in render_config
```
[APP DEBUG] Generated OVITO script: True (XXX chars)
[APP DEBUG] Render config created with ovito_script: False
```
**Problem**: Script not being added to render_config
**Solution**: Check render_config creation in app.py

### Scenario 3: Script Lost in Extraction
```
[TOOLS DEBUG] Extracted ovito_script from render_config: False (0 chars)
```
**Problem**: Script not in render_config or extraction failing
**Solution**: Check render_config.get('ovito_script') in tools.py

### Scenario 4: Script Lost in Thread Creation
```
[TOOLS DEBUG] Starting rendering worker with ovito_script: True
[RENDER DEBUG] Called with ovito_script: False (0 chars)
```
**Problem**: Thread args not passing script correctly
**Solution**: Check threading.Thread args in tools.py line ~1095

## Next Steps

1. Run the simulation with the user's command:
   ```
   Simulate melting and recrystallization of Al in 100 ps at 1 atm. 
   Heat to 4000K. Use 8000 atoms and longer on 1 side (horizontal). 
   Equilibrate for 10 ps first. Hold at 300K at the end for 10 ps.
   Color atoms by atom type, FCC, BCC etc. in Ovito
   ```

2. Watch the console output for the debug messages

3. Identify which step shows the script becoming None

4. Fix the specific issue at that step

## Current Hypothesis

Based on code review, the most likely issue is in the threading.Thread creation where the ovito_script parameter might not be passed correctly to the rendering_worker function. The debug logging will confirm this.
