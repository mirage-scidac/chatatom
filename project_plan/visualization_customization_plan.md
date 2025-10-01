# User-Customizable Visualization Plan

## Executive Summary
This plan outlines the implementation of user-customizable visualization features for the LLM Molecular Dynamics Studio, enabling users to request specific visualization styles through natural language, similar to how they currently specify simulation parameters.

## Current State Analysis

### Existing Capabilities
- **Fixed visualization pipeline**: Colors atoms by particle type only
- **Limited customization**: Quality (low/medium/high), camera angle (default/top/side)
- **Automatic frame sampling**: Smart interval calculation based on simulation length
- **Base64 image streaming**: Real-time visualization updates via WebSocket

### Key Limitations
1. No natural language parsing for visualization preferences
2. Hardcoded OVITO modifiers (only ColorCodingModifier with "Hot" gradient)
3. No support for structural analysis visualizations (CNA, CSP, etc.)
4. No user control over color schemes, particle sizes, or rendering styles

## Proposed Architecture

### 1. Visualization Request Parser
Create a new component within the Orchestrator Agent to extract visualization preferences:

```python
class VisualizationParser:
    """Parse natural language visualization requests."""
    
    def extract_viz_params(self, user_request: str) -> Dict:
        """
        Extract visualization parameters from user request.
        
        Returns dict with:
        - color_by: 'type', 'cna', 'csp', 'coordination', 'velocity', 'force', etc.
        - color_scheme: 'default', 'viridis', 'plasma', 'custom', etc.
        - particle_size: 'auto', 'uniform', 'by_type', 'by_property'
        - show_bonds: bool
        - show_box: bool
        - analysis_modifiers: List of OVITO modifiers to apply
        """
```

### 2. OVITO Modifier Pipeline System

#### Supported Analysis Types
Implement support for common OVITO analysis modifiers:

| Analysis Type | OVITO Modifier | Use Case |
|--------------|----------------|----------|
| Common Neighbor Analysis (CNA) | `CommonNeighborAnalysisModifier` | Identify crystal structures (FCC, HCP, BCC) |
| Centro-symmetry Parameter | `CentroSymmetryParameterModifier` | Detect defects and dislocations |
| Coordination Analysis | `CoordinationAnalysisModifier` | Count nearest neighbors |
| Ackland-Jones Analysis | `AcklandJonesModifier` | Identify phases in alloys |
| Atomic Strain | `AtomicStrainModifier` | Visualize local deformation |
| Displacement Vectors | `DisplacementVectorsModifier` | Show atomic motion |
| Voronoi Analysis | `VoronoiAnalysisModifier` | Analyze local atomic environment |
| Cluster Analysis | `ClusterAnalysisModifier` | Identify atomic clusters |

#### Dynamic Modifier Application
```python
def apply_visualization_modifiers(pipeline, viz_params):
    """
    Dynamically apply OVITO modifiers based on user preferences.
    
    Args:
        pipeline: OVITO Pipeline object
        viz_params: Parsed visualization parameters
    """
    # Clear existing modifiers
    pipeline.modifiers.clear()
    
    # Apply analysis modifiers first
    if viz_params['color_by'] == 'cna':
        cna_modifier = CommonNeighborAnalysisModifier()
        pipeline.modifiers.append(cna_modifier)
        
        # Color by structure type
        color_modifier = ColorCodingModifier(
            property='Structure Type',
            gradient=ColorCodingModifier.Viridis()
        )
        pipeline.modifiers.append(color_modifier)
    
    elif viz_params['color_by'] == 'coordination':
        coord_modifier = CoordinationAnalysisModifier(
            cutoff=3.5,  # Could be user-configurable
            number_of_bins=50
        )
        pipeline.modifiers.append(coord_modifier)
        
        color_modifier = ColorCodingModifier(
            property='Coordination',
            gradient=get_color_gradient(viz_params['color_scheme'])
        )
        pipeline.modifiers.append(color_modifier)
    
    # Add more modifier combinations...
```

### 3. Natural Language Examples

Map common user requests to visualization configurations:

| User Request | Parsed Configuration |
|-------------|---------------------|
| "Color atoms by crystal structure" | `color_by='cna'` |
| "Show me defects in the crystal" | `color_by='csp'`, highlight high CSP values |
| "Visualize atomic strain" | `color_by='strain'`, apply strain calculation |
| "Color by coordination number" | `color_by='coordination'` |
| "Show velocity vectors" | `show_vectors='velocity'`, arrow glyphs |
| "Highlight grain boundaries" | `color_by='cna'`, emphasize non-crystalline |
| "Show only surface atoms" | Apply `ConstructSurfaceModifier` |
| "Color by temperature" | `color_by='kinetic_energy'` (proxy for temperature) |

### 4. Implementation Phases

#### Phase 1: Core Infrastructure (Week 1)
- [ ] Extend Orchestrator Agent with visualization parsing
- [ ] Create `VisualizationConfig` class for storing preferences
- [ ] Modify `render_frame_to_image()` to accept visualization config
- [ ] Add basic modifier pipeline support

#### Phase 2: Analysis Modifiers (Week 2)
- [ ] Implement CNA (Common Neighbor Analysis) support
- [ ] Add Centro-symmetry Parameter analysis
- [ ] Implement Coordination Analysis
- [ ] Add Voronoi Analysis support
- [ ] Create modifier preset system

#### Phase 3: Advanced Features (Week 3)
- [ ] Add vector visualization (velocity, force, displacement)
- [ ] Implement slice/clip modifiers for cross-sections
- [ ] Add transparency and particle size controls
- [ ] Support for custom color maps
- [ ] Implement selection modifiers (by region, type, property)

#### Phase 4: User Experience (Week 4)
- [ ] Create visualization preset library
- [ ] Add real-time visualization parameter updates
- [ ] Implement visualization history/undo
- [ ] Add export options (high-res images, animations)
- [ ] Create visualization templates for common analyses

## Technical Implementation Details

### Modified Render Function
```python
def render_frame_to_image_custom(data_collection, viz_config):
    """
    Enhanced rendering with user-customizable visualization.
    
    Args:
        data_collection: OVITO DataCollection
        viz_config: VisualizationConfig object with user preferences
    
    Returns:
        str: Base64-encoded image with custom visualization
    """
    # Create pipeline
    pipeline = Pipeline(source=StaticSource(data=data_collection))
    
    # Apply user-requested modifiers
    apply_visualization_modifiers(pipeline, viz_config)
    
    # Configure viewport settings
    viewport = setup_viewport(viz_config)
    
    # Render with appropriate quality
    renderer = setup_renderer(viz_config.quality)
    
    # Generate image
    return render_to_base64(pipeline, viewport, renderer)
```

### WebSocket Message Format
```json
{
  "type": "visualization_update",
  "config": {
    "color_by": "cna",
    "color_scheme": "default",
    "show_bonds": false,
    "particle_size": "auto",
    "camera": {
      "position": [30, 30, 30],
      "target": [0, 0, 0],
      "up": [0, 0, 1]
    },
    "analysis": {
      "type": "cna",
      "parameters": {
        "cutoff": 3.5
      }
    }
  }
}
```

## Testing Strategy

### Unit Tests
- Test visualization parameter parsing
- Verify modifier pipeline construction
- Validate color scheme application
- Test error handling for invalid configurations

### Integration Tests
- End-to-end visualization with various modifiers
- Performance testing with large systems
- Memory usage validation
- Thread safety verification

### User Acceptance Tests
- Natural language request accuracy
- Visualization quality assessment
- Response time measurements
- User feedback on color schemes and clarity

## Performance Considerations

### Optimization Strategies
1. **Modifier Caching**: Cache analysis results for static frames
2. **Progressive Rendering**: Start with low quality, upgrade progressively
3. **Selective Analysis**: Only analyze visible atoms for large systems
4. **GPU Acceleration**: Use OVITO's GPU-accelerated modifiers where available
5. **Frame Skipping**: Skip frames if rendering can't keep up with simulation

### Benchmarks
- Target: < 500ms per frame for systems up to 10,000 atoms
- Memory usage: < 100MB per frame for typical visualizations
- Queue size: Maximum 10 frames in render queue

## Risk Mitigation

### Potential Risks
1. **Performance degradation** with complex modifiers
   - Mitigation: Implement quality levels and automatic degradation
   
2. **Thread safety issues** with OVITO on macOS
   - Mitigation: Continue using TachyonRenderer, test thoroughly
   
3. **User confusion** with too many options
   - Mitigation: Provide sensible defaults and presets
   
4. **Memory leaks** with multiple modifiers
   - Mitigation: Proper cleanup and resource management

## Success Metrics

### Quantitative Metrics
- Support for at least 8 different analysis types
- < 1 second response time for visualization updates
- 90% accuracy in parsing user visualization requests
- Memory usage stays within 2x of current implementation

### Qualitative Metrics
- Users can intuitively request complex visualizations
- Visualizations provide scientific insight
- System remains responsive during analysis
- Documentation is clear and comprehensive

## Future Enhancements

### Potential Extensions
1. **Machine Learning Integration**: Automatic defect detection and highlighting
2. **Multi-frame Analysis**: Track properties across trajectory
3. **Collaborative Features**: Share visualization configurations
4. **Export Capabilities**: Generate publication-quality figures
5. **VR/AR Support**: Immersive molecular visualization
6. **Custom Shaders**: User-defined rendering effects

## Conclusion

This plan transforms the current fixed visualization system into a flexible, user-customizable platform that responds to natural language requests. By leveraging OVITO's extensive modifier system and implementing intelligent parsing, users will be able to explore their molecular simulations with the same ease they currently enjoy when setting up simulations.

The phased approach ensures steady progress while maintaining system stability, and the comprehensive testing strategy guarantees reliability. Upon completion, users will be able to request visualizations like "color atoms by common neighbor analysis" or "highlight defects using centro-symmetry parameter" and receive appropriate, scientifically meaningful visualizations in real-time.
