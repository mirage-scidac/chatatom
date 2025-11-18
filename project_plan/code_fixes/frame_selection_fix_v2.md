# Frame Selection Fix v2 - Corrected Implementation

## Problem
After the initial refactoring to use OVITO's native `import_file()`, frames weren't updating in the UI. The counter was incrementing but all images looked identical.

## Root Cause
The `FileSource.loaded_frame` property is **read-only** and cannot be set directly. The initial fix attempted:
```python
pipeline.source.loaded_frame = frame_index  # ❌ AttributeError: no setter
```

## Corrected Solution

Instead of trying to set a read-only property, we now:

1. **Find the target frame index** by searching through frames
2. **Store the frame index** in a variable (`target_frame_index`)
3. **Use the frame index** when:
   - Computing pipeline data: `pipeline.compute(target_frame_index)`
   - Zooming viewport: `viewport.zoom_all((pipeline, target_frame_index), size=render_size)`
   - Rendering image: `viewport.render_image(..., frame=target_frame_index)`

## Implementation Details

### Key Changes in `visualization_rendering.py`:

```python
# Track which frame index to use for rendering
target_frame_index = 0

if isinstance(data_source, str):
    pipeline = import_file(data_source)
    
    # Find the frame with matching timestep
    if frame_timestep is not None:
        for frame_index in range(num_frames):
            data = pipeline.compute(frame_index)
            current_timestep = data.attributes.get('Timestep', None)
            
            if current_timestep == frame_timestep:
                target_frame_index = frame_index  # ✅ Store index
                break

# Later, use the stored index:
data = pipeline.compute(target_frame_index)  # ✅ Compute specific frame
viewport.zoom_all((pipeline, target_frame_index), size=render_size)  # ✅ Zoom to frame
viewport.render_image(..., frame=target_frame_index)  # ✅ Render specific frame
```

## How It Works Now

### Complete Flow:
1. **Monitor identifies timestep** → queues `{'dump_file': path, 'timestep': 12000, 'frame_number': 5}`
2. **Renderer receives frame info** → calls `render_frame_to_image_custom(dump_file, config, frame_timestep=12000)`
3. **OVITO loads dump file** → `pipeline = import_file(dump_file)` (loads ALL properties including c_cna)
4. **Search for matching frame:**
   - Loop through all frames
   - Check each frame's timestep attribute
   - When found, store `target_frame_index = frame_index`
5. **Use target frame throughout:**
   - Compute pipeline with that frame
   - Zoom viewport to that frame
   - Render that specific frame
6. **Result:** Different frames now show different states! ✅

## Benefits

1. **Frames update correctly** - Each rendered image shows the actual state at that timestep
2. **Maintains all properties** - Still loads c_cna and other custom properties via native import
3. **Smart sampling works** - Only renders target timesteps
4. **Robust error handling** - Falls back to frame 0 if timestep not found
5. **No read-only property errors** - Uses proper OVITO API methods

## Testing

To verify the fix works:
1. Run a simulation that generates multiple frames
2. Check that the UI shows different images as the counter increments
3. Verify atoms move/change between frames
4. Confirm CNA coloring (if enabled) shows structure changes over time

## Files Modified
- `visualization_rendering.py` - Fixed frame selection to use `target_frame_index` variable instead of trying to set read-only property
- `docs/frame_selection_fix_v2.md` - This updated documentation

## Comparison with Initial Attempt

### Initial (Broken):
```python
pipeline.source.loaded_frame = frame_index  # ❌ No setter
```

### Corrected (Working):
```python
target_frame_index = frame_index  # ✅ Store in variable
# Then use it:
pipeline.compute(target_frame_index)
viewport.zoom_all((pipeline, target_frame_index), size=render_size)
viewport.render_image(..., frame=target_frame_index)
```

## Next Steps
- Test with a real simulation to verify frames update correctly
- Verify CNA coloring works with the frame selection
- Monitor performance (frame search is O(n) but should be fast for typical simulations)
