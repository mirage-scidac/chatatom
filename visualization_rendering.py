"""
Visualization Rendering - OVITO-based molecular dynamics visualization.

This module provides functions to render molecular dynamics frames using OVITO,
with support for both dynamic script execution and static configuration.
"""

from typing import Dict, Any, Optional
import tempfile
import base64
import os
from ovito_executor import DynamicOVITOExecutor, VisualizationScriptError


def render_frame_to_image_custom(
    data_collection,
    viz_config: Dict[str, Any],
    client_info: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Enhanced rendering with dynamic OVITO script support.
    
    This function can handle both:
    1. Dynamic script mode: Execute LLM-generated OVITO Python scripts
    2. Static config mode: Use predefined visualization configurations
    
    IMPORTANT: This function is designed to NEVER raise exceptions that could
    stop the simulation. All errors are caught and logged, with graceful fallback
    to basic visualization or returning None.
    
    Args:
        data_collection: OVITO DataCollection with simulation data
        viz_config: Either:
            - {'mode': 'script', 'script': '...', ...} for dynamic scripts
            - {'mode': 'config', 'config': {...}, ...} for static configs
            - Legacy dict with direct config parameters
        client_info: Optional client information for logging
        
    Returns:
        str: Base64-encoded PNG image, or None on error (simulation continues)
    """
    try:
        from ovito.pipeline import Pipeline, StaticSource
        from ovito.vis import Viewport, TachyonRenderer
        from ovito.modifiers import ColorCodingModifier
        import platform
        import threading
        
        if data_collection is None:
            print("Error: data_collection is None")
            return None
        
        # Create pipeline from data
        pipeline = Pipeline(source=StaticSource(data=data_collection))
        
        # Determine visualization mode
        mode = viz_config.get('mode', 'config')
        
        if mode == 'script' and 'script' in viz_config and viz_config['script']:
            # Dynamic script mode - execute LLM-generated OVITO script
            # Wrap in try-catch to ensure errors don't propagate
            try:
                executor = DynamicOVITOExecutor()
                pipeline = executor.execute_script(
                    viz_config['script'],
                    pipeline,
                    data_collection
                )
                
            except VisualizationScriptError as e:
                print(f"Visualization script execution failed: {e}")
                print("Falling back to basic visualization")
                try:
                    apply_basic_visualization(pipeline)
                except Exception as fallback_error:
                    print(f"Fallback visualization also failed: {fallback_error}")
                    return None
                
            except Exception as e:
                print(f"Unexpected error executing visualization script: {type(e).__name__}: {e}")
                print("Falling back to basic visualization")
                try:
                    apply_basic_visualization(pipeline)
                except Exception as fallback_error:
                    print(f"Fallback visualization also failed: {fallback_error}")
                    return None
        
        else:
            # Static configuration mode (backward compatible)
            try:
                # Extract config if nested
                if 'config' in viz_config:
                    config = viz_config['config']
                else:
                    config = viz_config
                
                apply_static_visualization(pipeline, config)
            except Exception as e:
                print(f"Error applying static visualization: {type(e).__name__}: {e}")
                print("Falling back to basic visualization")
                try:
                    apply_basic_visualization(pipeline)
                except Exception as fallback_error:
                    print(f"Fallback visualization also failed: {fallback_error}")
                    return None
        
        # Add pipeline to scene
        pipeline.add_to_scene()
        
        # Create viewport for rendering
        viewport = Viewport()
        viewport.type = Viewport.Type.Perspective
        
        # Set camera position based on config
        camera_angle = viz_config.get('camera_angle', 'default')
        if isinstance(viz_config.get('config'), dict):
            camera_angle = viz_config['config'].get('camera_angle', camera_angle)
        
        if camera_angle == 'top':
            viewport.camera_pos = (0, 0, 50)
            viewport.camera_dir = (0, 0, -1)
        elif camera_angle == 'side':
            viewport.camera_pos = (50, 0, 0)
            viewport.camera_dir = (-1, 0, 0)
        else:  # default perspective
            viewport.camera_pos = (30, 30, 30)
            viewport.camera_dir = (-1, -1, -1)
        
        # Determine render size based on quality
        quality = viz_config.get('quality', 'medium')
        if isinstance(viz_config.get('config'), dict):
            quality = viz_config['config'].get('quality', quality)
        
        if quality == 'high':
            render_size = (800, 600)
        elif quality == 'low':
            render_size = (400, 300)
        else:  # medium
            render_size = (600, 450)
        
        # Zoom to fit all particles
        viewport.zoom_all(size=render_size)
        
        # Render to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Use TachyonRenderer for thread safety (especially on macOS)
            renderer = TachyonRenderer()
            renderer.antialiasing = True
            renderer.direct_light_intensity = 0.9
            renderer.shadows = True
            renderer.ambient_occlusion = True
            renderer.depth_of_field = False
            
            # Check threading context
            is_main_thread = threading.current_thread() is threading.main_thread()
            is_macos = platform.system() == 'Darwin'
            
            if not is_main_thread and is_macos:
                print("Non-main thread on macOS - using TachyonRenderer")
            
            # Render the image
            viewport.render_image(
                filename=tmp_path,
                size=render_size,
                alpha=True,
                renderer=renderer
            )
            
            # Read and encode as base64
            with open(tmp_path, 'rb') as img_file:
                img_data = img_file.read()
                base64_image = base64.b64encode(img_data).decode('utf-8')
            
            return base64_image
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        print(f"Error rendering frame: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None


def apply_basic_visualization(pipeline) -> None:
    """
    Apply basic visualization (fallback when script execution fails).
    
    Args:
        pipeline: OVITO Pipeline to modify
    """
    from ovito.modifiers import ColorCodingModifier
    
    # Simple color coding by particle type
    color_modifier = ColorCodingModifier(
        property='Particle Type',
        gradient=ColorCodingModifier.Hot()
    )
    pipeline.modifiers.append(color_modifier)


def apply_static_visualization(pipeline, config: Dict[str, Any]) -> None:
    """
    Apply static visualization configuration.
    
    This is the backward-compatible mode that uses predefined
    visualization patterns.
    
    Args:
        pipeline: OVITO Pipeline to modify
        config: Configuration dictionary
    """
    from ovito.modifiers import (
        ColorCodingModifier,
        CommonNeighborAnalysisModifier,
        CentroSymmetryModifier
    )
    
    # Check if analysis is needed
    analysis = config.get('analysis', None)
    
    if analysis == 'CNA':
        # Apply Common Neighbor Analysis
        cna = CommonNeighborAnalysisModifier(cutoff=3.5)
        pipeline.modifiers.append(cna)
    elif analysis == 'CSP':
        # Apply Centro-Symmetry Parameter
        csp = CentroSymmetryModifier(num_neighbors=12)
        pipeline.modifiers.append(csp)
    
    # Apply color coding
    color_by = config.get('color_by', 'Particle Type')
    color_scheme = config.get('color_scheme', 'viridis')
    
    # Map color scheme names to OVITO gradients
    gradient_map = {
        'viridis': ColorCodingModifier.Viridis(),
        'hot': ColorCodingModifier.Hot(),
        'jet': ColorCodingModifier.Jet(),
        'rainbow': ColorCodingModifier.Rainbow(),
    }
    
    gradient = gradient_map.get(color_scheme.lower(), ColorCodingModifier.Viridis())
    
    color_modifier = ColorCodingModifier(
        property=color_by,
        gradient=gradient
    )
    pipeline.modifiers.append(color_modifier)
