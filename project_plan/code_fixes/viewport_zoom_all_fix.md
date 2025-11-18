# Viewport zoom_all() Fix - Correcting OVITO API Usage

## Problem
After implementing frame selection in v2, rendering failed with:
```
TypeError: _Viewport_zoom_all() got multiple values for argument 'size'
```

This error occurred at line 182 in `visualization_rendering.py`:
```python
viewport.zoom_all((pipeline, target_frame_index), size=render_size)
```

## Root Cause
The `viewport.zoom_all()` method in OVITO does not accept a tuple of `(pipeline, frame_index)` as its first positional argument. The method signature is simpler than we assumed.

### Incorrect Usage (What We Had):
```python
viewport.zoom_all((pipeline, target_frame_index), size=render_size)
```

This attempted to pass:
1. A tuple `(pipeline, target_frame_index)` as the first positional argument
2. `size=render_size` as a keyword argument

But OVITO's `zoom_all()` method doesn't expect these arguments in this way, causing the "multiple values for argument 'size'" error.

## Solution
The `viewport.zoom_all()` method should be called without arguments. It automatically zooms to fit all objects currently in the scene.

### Correct Usage:
```python
viewport.zoom_all()
```

## Why This Works

The frame context is already properly established through:

1. **Pipeline computation at target frame:**
   ```python
   data = pipeline.compute(target_frame_index)
   ```

2. **Pipeline added to scene:**
   ```python
   pipeline.add_to_scene()
   ```

3. **Frame specified in render call:**
   ```python
   viewport.render_image(
       filename=tmp_path,
       size=render_size,
       alpha=True,
       renderer=renderer,
       frame=target_frame_index  # Frame context here
   )
   ```

The `zoom_all()` method operates on the current scene state, which already includes the pipeline at the correct frame. The actual frame rendering is controlled by the `frame` parameter in `render_image()`.

## Implementation

### Changed Code (Line 182):
```python
# BEFORE (incorrect):
viewport.zoom_all((pipeline, target_frame_index), size=render_size)

# AFTER (correct):
viewport.zoom_all()
```

### Complete Rendering Flow:
```python
# 1. Load dump file with all frames
pipeline = import_file(data_source)

# 2. Find target frame by timestep
for frame_index in range(num_frames):
    data = pipeline.compute(frame_index)
    if data.attributes.get('Timestep') == frame_timestep:
        target_frame_index = frame_index
        break

# 3. Apply visualization modifiers
apply_static_visualization(pipeline, config)

# 4. Add to scene
pipeline.add_to_scene()

# 5. Setup viewport
viewport = Viewport()
viewport.type = Viewport.Type.Perspective
viewport.camera_pos = (30, 30, 30)
viewport.camera_dir = (-1, -1, -1)

# 6. Zoom to fit (no arguments needed)
viewport.zoom_all()  # ✅ Correct

# 7. Render specific frame
viewport.render_image(
    filename=tmp_path,
    size=render_size,
    alpha=True,
    renderer=renderer,
    frame=target_frame_index  # Frame selection happens here
)
```

## Key Insights

1. **zoom_all() is scene-based**: It zooms to fit all objects in the current scene, regardless of frame
2. **Frame selection is render-time**: The specific frame is selected via the `frame` parameter in `render_image()`
3. **Pipeline state is frame-agnostic**: The pipeline can compute different frames, but zoom_all() doesn't need to know which one

## Testing Checklist

After this fix, verify:
- [x] No TypeError when rendering frames
- [ ] Frames render successfully
- [ ] Different frames show different atomic positions
- [ ] CNA coloring works correctly (if enabled)
- [ ] Smart sampling renders only target timesteps
- [ ] Viewport properly zooms to fit all particles

## Files Modified
- `visualization_rendering.py` - Line 182: Simplified `viewport.zoom_all()` call
- `docs/viewport_zoom_all_fix.md` - This documentation

## Related Documentation
- `docs/frame_selection_fix_v2.md` - Previous fix for frame selection
- `docs/ovito_native_reader_refactoring.md` - Initial refactoring to use native import_file()

## OVITO API Reference
The correct usage follows OVITO's standard API pattern where:
- `zoom_all()` operates on the current scene state
- Frame selection is handled separately via `frame` parameters in compute/render methods
- No need to pass pipeline or frame information to zoom_all()
