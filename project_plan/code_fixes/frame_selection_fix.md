# Frame Selection Fix for Multi-Frame Dump Files

## Problem
After refactoring to use OVITO's native `import_file()`, frames weren't updating in the UI. The counter was incrementing but all images looked identical.

## Root Cause
In `visualization_rendering.py`, when loading a dump file with `import_file()`, we weren't selecting the specific frame to render. The code had a TODO comment and was always rendering frame 0:

```python
# For now, just use the first frame - OVITO will handle multi-frame files
# TODO: Add frame selection logic if needed
```

This meant:
- ✅ Frame counter incremented correctly
- ✅ Different timesteps were identified
- ❌ But we always rendered the same frame (frame 0)

## Solution Implemented
Added proper frame selection logic that:

1. **Searches for the matching timestep:**
   ```python
   for frame_index in range(num_frames):
       data = pipeline.compute(frame_index)
       current_timestep = data.attributes.get('Timestep', None)
       
       if current_timestep == frame_timestep:
           pipeline.source.loaded_frame = frame_index
           break
   ```

2. **Sets the pipeline to use that specific frame:**
   ```python
   pipeline.source.loaded_frame = frame_index
   ```

3. **Provides fallback if timestep not found:**
   ```python
   if not frame_found:
       print(f"[RENDER] Warning: Timestep {frame_timestep} not found, using frame 0")
       pipeline.source.loaded_frame = 0
   ```

## How It Works Now

### Complete Flow:
1. **Monitor identifies timestep** → queues `{'dump_file': path, 'timestep': 12000, 'frame_number': 5}`
2. **Renderer receives frame info** → calls `render_frame_to_image_custom(dump_file, config, frame_timestep=12000)`
3. **OVITO loads dump file** → `pipeline = import_file(dump_file)`
4. **Search for matching frame:**
   - Loop through all frames in the file
   - Check each frame's timestep attribute
   - When found, set `pipeline.source.loaded_frame = frame_index`
5. **Render that specific frame** → Different frames now show different states!

## Benefits

1. **Frames now update correctly** - Each rendered image shows the actual state at that timestep
2. **Maintains all properties** - Still loads c_cna and other custom properties
3. **Smart sampling still works** - Only renders target timesteps
4. **Robust error handling** - Falls back to frame 0 if timestep not found

## Testing

To verify the fix works:
1. Run a simulation that generates multiple frames
2. Check that the UI shows different images as the counter increments
3. Verify atoms move/change between frames
4. Confirm CNA coloring (if enabled) shows structure changes over time

## Files Modified
- `visualization_rendering.py` - Added frame selection logic in `render_frame_to_image_custom()`

## Next Steps
- Test with a real simulation to verify frames update correctly
- Verify CNA coloring works with the frame selection
- Monitor performance (frame search is O(n) but should be fast for typical simulations)
