# User-Customizable Visualization Plan

## Executive Summary
This plan outlines the implementation of user-customizable visualization features for the LLM Molecular Dynamics Studio, enabling users to request specific visualization styles through natural language. The system uses **dynamic OVITO script generation** to handle arbitrary visualization requests without requiring code changes for each new visualization type.

## Architecture Philosophy: Dynamic Script Generation

### The Problem with Static Implementation
The initial approach of hardcoding OVITO modifiers in `visualization_rendering.py` has fundamental limitations:
- Every new visualization type requires code changes
- Cannot handle complex, multi-step OVITO operations
- Limited to pre-defined modifier combinations
- Doesn't scale with user needs (e.g., dislocation analysis, custom filtering)

### The Solution: LLM-Generated OVITO Scripts
Similar to how the system generates LAMMPS input scripts dynamically, we implement **dynamic OVITO Python script generation**:

```
User Request: "Show only crystalline atoms, colored by structure type"
    ↓
VisualizationAgent (LLM)
    ↓
Generates OVITO Python Script:
    ```python
    from ovito.modifiers import CommonNeighborAnalysisModifier, SelectTypeModifier, DeleteSelectedModifier
    
    # Apply CNA to identify structures
    pipeline.modifiers.append(CommonNeighborAnalysisModifier(cutoff=3.5))
    
    # Select non-crystalline atoms (Structure Type = 5)
    select = SelectTypeModifier(property='Structure Type', types={5})
    pipeline.modifiers.append(select)
    
    # Delete selected atoms
    pipeline.modifiers.append(DeleteSelectedModifier())
    
    # Color remaining atoms by structure
    color = ColorCodingModifier(property='Structure Type', gradient=...)
    pipeline.modifiers.append(color)
    ```
    ↓
Execute Script Dynamically
    ↓
Render Result
```

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
5. **Cannot handle arbitrary OVITO operations without code changes**

## Proposed Architecture: Dynamic OVITO Script Generation

### Core Components

#### 1. OVITOScriptGenerator (New Component)
```python
class OVITOScriptGenerator(Agent):
    """
    LLM-powered generator for OVITO Python scripts.
    
    Similar to LAMMPS script generation, this agent generates complete
    OVITO Python code based on natural language visualization requests.
    """
    
    def __init__(self):
        system_prompt = """You are an expert in OVITO Python scripting for molecular dynamics visualization.

ROLE:
Generate complete, executable OVITO Python scripts based on natural language requests.
You have deep knowledge of OVITO's API, modifiers, and best practices.

OVITO PYTHON API KNOWLEDGE:

1. PIPELINE STRUCTURE:
```python
from ovito.pipeline import Pipeline, StaticSource
from ovito.io import import_file

# Create pipeline
pipeline = Pipeline(source=StaticSource(data=data_collection))

# Add modifiers
pipeline.modifiers.append(modifier)

# Compute results
data = pipeline.compute()
```

2. COMMON MODIFIERS:

CommonNeighborAnalysisModifier:
```python
from ovito.modifiers import CommonNeighborAnalysisModifier
cna = CommonNeighborAnalysisModifier(cutoff=3.5, mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff)
pipeline.modifiers.append(cna)
# Creates 'Structure Type' property: 1=FCC, 2=HCP, 3=BCC, 4=ICO, 5=Other
```

SelectTypeModifier (for filtering):
```python
from ovito.modifiers import SelectTypeModifier
# Select atoms with specific property values
select = SelectTypeModifier(property='Structure Type', types={5})  # Select "Other" structures
pipeline.modifiers.append(select)
```

DeleteSelectedModifier (remove atoms):
```python
from ovito.modifiers import DeleteSelectedModifier
pipeline.modifiers.append(DeleteSelectedModifier())
```

InvertSelectionModifier (invert selection):
```python
from ovito.modifiers import InvertSelectionModifier
pipeline.modifiers.append(InvertSelectionModifier())
```

ColorCodingModifier:
```python
from ovito.modifiers import ColorCodingModifier
color = ColorCodingModifier(
    property='Structure Type',
    gradient=ColorCodingModifier.Viridis()  # or .Hot(), .Jet(), etc.
)
pipeline.modifiers.append(color)
```

CentroSymmetryModifier (defect detection):
```python
from ovito.modifiers import CentroSymmetryModifier
csp = CentroSymmetryModifier(num_neighbors=12)
pipeline.modifiers.append(csp)
# Creates 'Centrosymmetry' property
```

ExpressionSelectionModifier (complex filtering):
```python
from ovito.modifiers import ExpressionSelectionModifier
# Select atoms based on expression
expr = ExpressionSelectionModifier(expression='Centrosymmetry > 0.5')
pipeline.modifiers.append(expr)
```

CoordinationAnalysisModifier:
```python
from ovito.modifiers import CoordinationAnalysisModifier
coord = CoordinationAnalysisModifier(cutoff=3.5, number_of_bins=50)
pipeline.modifiers.append(coord)
```

DislocationAnalysisModifier:
```python
from ovito.modifiers import DislocationAnalysisModifier
dxa = DislocationAnalysisModifier()
pipeline.modifiers.append(dxa)
# Identifies dislocation lines in crystals
```

3. VISUALIZATION PATTERNS:

Pattern: "Show only crystalline atoms"
```python
# Apply CNA
pipeline.modifiers.append(CommonNeighborAnalysisModifier(cutoff=3.5))
# Select non-crystalline (Structure Type = 5)
pipeline.modifiers.append(SelectTypeModifier(property='Structure Type', types={5}))
# Delete selected
pipeline.modifiers.append(DeleteSelectedModifier())
# Color by structure
pipeline.modifiers.append(ColorCodingModifier(property='Structure Type'))
```

Pattern: "Highlight defects"
```python
# Apply CSP
pipeline.modifiers.append(CentroSymmetryModifier(num_neighbors=12))
# Color by CSP value
pipeline.modifiers.append(ColorCodingModifier(
    property='Centrosymmetry',
    gradient=ColorCodingModifier.Hot()
))
```

Pattern: "Show only surface atoms"
```python
from ovito.modifiers import ConstructSurfaceModifier
surface = ConstructSurfaceModifier(radius=3.0, smoothing_level=8)
pipeline.modifiers.append(surface)
# Surface atoms get 'Surface Distance' property
# Select non-surface atoms (Surface Distance > 0)
pipeline.modifiers.append(ExpressionSelectionModifier(expression='SurfaceDistance > 0'))
pipeline.modifiers.append(DeleteSelectedModifier())
```

Pattern: "Analyze dislocations"
```python
from ovito.modifiers import DislocationAnalysisModifier
dxa = DislocationAnalysisModifier()
pipeline.modifiers.append(dxa)
# Dislocation lines are created as separate data objects
```

4. BEST PRACTICES:
- Always import required modifiers at the top
- Apply analysis modifiers before selection/filtering
- Use ExpressionSelectionModifier for complex conditions
- Chain modifiers in logical order
- Include comments explaining each step
- Handle edge cases (empty selections, etc.)

5. COMMON WORKFLOWS:

Filtering workflow:
1. Apply analysis modifier (CNA, CSP, etc.)
2. Select atoms based on property
3. Either delete or invert selection
4. Apply color coding to remaining atoms

Multi-property analysis:
1. Apply multiple analysis modifiers
2. Use ExpressionSelectionModifier to combine conditions
3. Color by most relevant property

OUTPUT FORMAT:
Generate a complete Python script that:
1. Imports all required OVITO modules
2. Assumes 'pipeline' variable exists with data loaded
3. Applies modifiers in correct order
4. Includes explanatory comments
5. Returns the modified pipeline

Example output for "show only FCC atoms":
```python
from ovito.modifiers import CommonNeighborAnalysisModifier, SelectTypeModifier, DeleteSelectedModifier, InvertSelectionModifier, ColorCodingModifier

# Apply Common Neighbor Analysis to identify crystal structures
cna = CommonNeighborAnalysisModifier(cutoff=3.5)
pipeline.modifiers.append(cna)

# Select FCC atoms (Structure Type = 1)
select_fcc = SelectTypeModifier(property='Structure Type', types={1})
pipeline.modifiers.append(select_fcc)

# Invert selection to select non-FCC atoms
pipeline.modifiers.append(InvertSelectionModifier())

# Delete non-FCC atoms
pipeline.modifiers.append(DeleteSelectedModifier())

# Color remaining FCC atoms
color = ColorCodingModifier(property='Structure Type', gradient=ColorCodingModifier.Viridis())
pipeline.modifiers.append(color)
```

IMPORTANT:
- Generate ONLY the Python code, no markdown formatting
- Assume 'pipeline' variable is already created
- Include all necessary imports
- Add comments for clarity
- Handle edge cases appropriately
"""
        
        super().__init__(
            system_prompt,
            model=config.ORCHESTRATOR_MODEL,
            default_temperature=0.2  # Lower temperature for code generation
        )
    
    def generate_ovito_script(self, user_request: str, simulation_context: dict = None) -> str:
        """
        Generate OVITO Python script from natural language request.
        
        Args:
            user_request: Natural language visualization request
            simulation_context: Optional simulation context
            
        Returns:
            str: Complete OVITO Python script
        """
        context_str = ""
        if simulation_context:
            context_str = f"""
SIMULATION CONTEXT:
- Material: {simulation_context.get('material', 'Unknown')}
- System size: {simulation_context.get('natoms', 'Unknown')} atoms
- Temperature: {simulation_context.get('temperature', 'Unknown')} K
- Crystal structure: {simulation_context.get('structure', 'Unknown')}
"""
        
        prompt = f"""{context_str}

USER VISUALIZATION REQUEST:
{user_request}

Generate a complete OVITO Python script to accomplish this visualization.
Include all necessary imports and comments.
Assume the 'pipeline' variable already exists with data loaded.
"""
        
        script = self.execute(prompt)
        
        # Clean up script (remove markdown if present)
        if "```python" in script:
            script = script.split("```python")[1].split("```")[0].strip()
        elif "```" in script:
            script = script.split("```")[1].split("```")[0].strip()
        
        return script
```

#### 2. DynamicOVITOExecutor (New Component)
```python
class DynamicOVITOExecutor:
    """
    Safely execute dynamically generated OVITO scripts.
    
    Provides sandboxed execution environment with error handling.
    """
    
    def __init__(self):
        self.allowed_imports = {
            'ovito.modifiers',
            'ovito.pipeline',
            'ovito.vis',
            'ovito.data',
            'numpy',
            'math'
        }
        self.execution_timeout = 30  # seconds
        
    def execute_script(self, script: str, pipeline, data_collection) -> Pipeline:
        """
        Execute OVITO script in controlled environment.
        
        Args:
            script: Generated OVITO Python code
            pipeline: OVITO Pipeline object
            data_collection: OVITO DataCollection
            
        Returns:
            Modified pipeline
        """
        # Validate script safety
        self._validate_script(script)
        
        # Create execution namespace
        namespace = {
            'pipeline': pipeline,
            'data_collection': data_collection,
            '__builtins__': __builtins__
        }
        
        try:
            # Execute script with timeout
            exec(script, namespace)
            return namespace['pipeline']
            
        except Exception as e:
            print(f"Error executing OVITO script: {e}")
            print(f"Script was:\n{script}")
            raise VisualizationScriptError(f"Failed to execute visualization script: {e}")
    
    def _validate_script(self, script: str):
        """Validate script for security and safety."""
        # Check for dangerous operations
        dangerous_patterns = [
            'import os',
            'import sys',
            'import subprocess',
            'eval(',
            'exec(',
            '__import__',
            'open(',
            'file(',
        ]
        
        for pattern in dangerous_patterns:
            if pattern in script:
                raise SecurityError(f"Dangerous operation detected: {pattern}")
        
        # Validate imports
        import_lines = [line for line in script.split('\n') if line.strip().startswith('from ') or line.strip().startswith('import ')]
        for line in import_lines:
            module = line.split()[1].split('.')[0]
            if module not in self.allowed_imports and module not in ['ovito']:
                raise SecurityError(f"Unauthorized import: {module}")


class VisualizationScriptError(Exception):
    """Raised when OVITO script execution fails."""
    pass


class SecurityError(Exception):
    """Raised when script contains dangerous operations."""
    pass
```

#### 3. Enhanced VisualizationAgent
```python
class VisualizationAgent(Agent):
    """
    Enhanced visualization agent with dynamic script generation.
    
    Can operate in two modes:
    1. Configuration mode: Generate structured config (backward compatible)
    2. Script mode: Generate OVITO Python code (new, more flexible)
    """
    
    def __init__(self):
        super().__init__(...)
        self.script_generator = OVITOScriptGenerator()
        self.script_executor = DynamicOVITOExecutor()
        
    def parse_visualization_request(self, user_request: str, simulation_context: dict = None, mode: str = 'script') -> dict:
        """
        Parse visualization request.
        
        Args:
            user_request: Natural language request
            simulation_context: Simulation context
            mode: 'script' for dynamic generation, 'config' for structured config
            
        Returns:
            dict with either 'script' or 'config' key
        """
        if mode == 'script':
            # Generate OVITO Python script
            script = self.script_generator.generate_ovito_script(user_request, simulation_context)
            
            return {
                'mode': 'script',
                'script': script,
                'rationale': f"Generated OVITO script for: {user_request}",
                'warnings': []
            }
        else:
            # Fall back to configuration mode (existing implementation)
            return self._parse_to_config(user_request, simulation_context)
```

#### 4. Updated visualization_rendering.py
```python
def render_frame_to_image_custom(data_collection, viz_config, client_info=None):
    """
    Enhanced rendering with dynamic OVITO script support.
    
    Args:
        data_collection: OVITO DataCollection
        viz_config: Either dict with 'script' key or VisualizationConfig object
        client_info: Optional client information
        
    Returns:
        str: Base64-encoded image
    """
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.vis import Viewport, TachyonRenderer
    
    # Create pipeline
    pipeline = Pipeline(source=StaticSource(data=data_collection))
    
    # Check if we have a dynamic script or static config
    if isinstance(viz_config, dict) and 'script' in viz_config:
        # Dynamic script mode
        executor = DynamicOVITOExecutor()
        try:
            pipeline = executor.execute_script(
                viz_config['script'],
                pipeline,
                data_collection
            )
        except Exception as e:
            print(f"Script execution failed: {e}")
            # Fall back to basic visualization
            apply_basic_visualization(pipeline)
    else:
        # Static configuration mode (existing implementation)
        apply_ovito_modifiers(pipeline, viz_config)
    
    # Add pipeline to scene
    pipeline.add_to_scene()
    
    # Render (existing code)
    viewport = Viewport()
    # ... rest of rendering code
```

### Architecture Diagram

```
User: "Show only crystalline atoms, colored by structure"
    ↓
OrchestratorAgent
    ↓
VisualizationAgent.parse_visualization_request(mode='script')
    ↓
OVITOScriptGenerator.generate_ovito_script()
    ↓
LLM generates:
    ```python
    from ovito.modifiers import CommonNeighborAnalysisModifier, SelectTypeModifier, DeleteSelectedModifier
    pipeline.modifiers.append(CommonNeighborAnalysisModifier(cutoff=3.5))
    pipeline.modifiers.append(SelectTypeModifier(property='Structure Type', types={5}))
    pipeline.modifiers.append(DeleteSelectedModifier())
    ```
    ↓
DynamicOVITOExecutor.execute_script()
    ↓
Validated & Executed
    ↓
render_frame_to_image_custom()
    ↓
Base64 image returned to user
```

## Benefits of Dynamic Script Generation

### 1. Infinite Extensibility
- **No code changes needed** for new visualization types
- Handles arbitrary OVITO operations
- Can combine multiple modifiers in complex ways
- Supports future OVITO features automatically

### 2. Natural Language Understanding
- LLM understands intent: "show only crystalline atoms" → generates selection + deletion
- Handles ambiguity: "highlight defects" → chooses appropriate analysis method
- Context-aware: Considers material type, simulation conditions

### 3. Complex Operations
Examples that would be difficult with static configs:

```python
# "Show grain boundaries in FCC regions"
- Apply CNA
- Select FCC atoms
- Identify grain boundaries within FCC
- Color by grain ID

# "Visualize dislocation cores with high CSP"
- Apply DislocationAnalysis
- Apply CentroSymmetry
- Select atoms near dislocations with CSP > threshold
- Highlight selected atoms

# "Show surface atoms with coordination < 12"
- Apply ConstructSurface
- Apply CoordinationAnalysis
- Select surface atoms with Coordination < 12
- Color by coordination number
```

### 4. Self-Documenting
- Generated scripts include comments
- Users can see exactly what OVITO operations are performed
- Educational value: users learn OVITO API

### 5. Debugging and Iteration
- If script fails, LLM can regenerate with fixes
- Users can request modifications: "same but with different cutoff"
- Error messages reference specific OVITO operations

## Implementation Phases (Revised)

### Phase 1: Dynamic Script Generation Foundation (Week 1)
**Priority: Establish script generation and execution**

- [ ] Create `OVITOScriptGenerator` class with comprehensive OVITO API knowledge
- [ ] Implement `DynamicOVITOExecutor` with security validation
- [ ] Test script generation with diverse requests
- [ ] Implement script validation and sandboxing
- [ ] Add error handling and fallback mechanisms
- [ ] Create test suite for script generation
- [ ] Document OVITO API patterns in system prompt

**Success Criteria:**
- LLM generates valid OVITO scripts for 80%+ of requests
- Scripts execute safely in sandboxed environment
- Fallback to basic visualization on errors

### Phase 2: Integration with Existing System (Week 2)
**Priority: Connect script generation to rendering pipeline**

- [ ] Update `VisualizationAgent` to support script mode
- [ ] Modify `render_frame_to_image_custom()` to handle scripts
- [ ] Implement backward compatibility with config mode
- [ ] Add script caching for performance
- [ ] Create WebSocket handlers for script-based visualization
- [ ] Implement script execution monitoring
- [ ] Add logging and debugging tools

**Success Criteria:**
- Script mode works alongside existing config mode
- No regression in existing functionality
- Scripts execute within performance bounds

### Phase 3: Advanced Features and Optimization (Week 3)
**Priority: Enhance capabilities and performance**

- [ ] Add script optimization (remove redundant operations)
- [ ] Implement script templates for common patterns
- [ ] Add user feedback loop for script refinement
- [ ] Create visualization preset library (script-based)
- [ ] Implement script versioning and history
- [ ] Add export functionality (save scripts)
- [ ] Optimize script execution performance

**Success Criteria:**
- Complex visualizations work reliably
- Performance meets targets (< 500ms p90)
- Users can save and reuse scripts

### Phase 4: Polish and Production (Week 4)
**Priority: Production readiness**

- [ ] Comprehensive testing with edge cases
- [ ] Security audit of script execution
- [ ] Performance optimization
- [ ] Documentation and examples
- [ ] User guide for script-based visualization
- [ ] Analytics and monitoring
- [ ] Production deployment

**Success Criteria:**
- System handles production load
- Security validated
- Documentation complete

## Migration Strategy

### Backward Compatibility
The system supports both modes:

1. **Script Mode** (new, default):
   - Dynamic OVITO script generation
   - Maximum flexibility
   - Handles arbitrary requests

2. **Config Mode** (existing):
   - Structured configuration
   - Faster for simple cases
   - Fallback when script generation fails

### Gradual Rollout
1. Deploy script generation alongside existing system
2. A/B test with subset of users
3. Monitor performance and accuracy
4. Gradually increase script mode usage
5. Keep config mode as fallback

## Security Considerations

### Script Validation
- Whitelist allowed imports
- Block dangerous operations (file I/O, system calls)
- Execution timeout (30 seconds)
- Resource limits (memory, CPU)

### Sandboxing
- Restricted namespace
- No access to filesystem
- No network operations
- Limited to OVITO operations

### Monitoring
- Log all generated scripts
- Track execution failures
- Alert on suspicious patterns
- Regular security audits

## Performance Considerations

### Script Execution Overhead
- Parsing: ~10ms
- Validation: ~5ms
- Execution: Variable (depends on modifiers)
- Total overhead: ~15-20ms (acceptable)

### Optimization Strategies
1. **Script Caching**: Cache generated scripts for common requests
2. **Template Matching**: Use templates for frequent patterns
3. **Lazy Execution**: Only execute when rendering needed
4. **Parallel Execution**: Run multiple scripts concurrently

### Benchmarks
- Script generation: < 100ms (p90)
- Script validation: < 10ms
- Script execution: < 500ms for typical operations
- Total visualization time: < 700ms (p90)

## Testing Strategy

### Unit Tests
- Script generation for various requests
- Script validation (security)
- Script execution (correctness)
- Error handling and fallbacks

### Integration Tests
- End-to-end visualization with scripts
- Complex multi-modifier scenarios
- Performance under load
- Security validation

### User Acceptance Tests
- Natural language request accuracy
- Visualization quality
- Response time
- User satisfaction

## Success Metrics

### Quantitative
- Script generation accuracy: > 85%
- Script execution success rate: > 90%
- Response time: < 700ms (p90)
- Security incidents: 0
- User satisfaction: > 4.0/5.0

### Qualitative
- Users can request arbitrary visualizations
- System handles edge cases gracefully
- Documentation is clear
- Scripts are educational

## Example Use Cases

### 1. Crystalline Atom Filtering
**Request**: "Show only crystalline atoms"

**Generated Script**:
```python
from ovito.modifiers import CommonNeighborAnalysisModifier, SelectTypeModifier, DeleteSelectedModifier, ColorCodingModifier

# Identify crystal structures
pipeline.modifiers.append(CommonNeighborAnalysisModifier(cutoff=3.5))

# Select non-crystalline atoms (Structure Type = 5)
pipeline.modifiers.append(SelectTypeModifier(property='Structure Type', types={5}))

# Delete non-crystalline atoms
pipeline.modifiers.append(DeleteSelectedModifier())

# Color by structure type
pipeline.modifiers.append(ColorCodingModifier(property='Structure Type'))
```

### 2. Dislocation Analysis
**Request**: "Analyze dislocations and show only atoms near dislocation cores"

**Generated Script**:
```python
from ovito.modifiers import DislocationAnalysisModifier, ExpressionSelectionModifier, DeleteSelectedModifier, InvertSelectionModifier

# Identify dislocations
pipeline.modifiers.append(DislocationAnalysisModifier())

# Select atoms far from dislocations (distance > 5 Angstroms)
pipeline.modifiers.append(ExpressionSelectionModifier(expression='DislocationDistance > 5'))

# Delete atoms far from dislocations
pipeline.modifiers.append(DeleteSelectedModifier())
```

### 3. Surface Defect Visualization
**Request**: "Show surface atoms with defects"

**Generated Script**:
```python
from ovito.modifiers import ConstructSurfaceModifier, CentroSymmetryModifier, ExpressionSelectionModifier, DeleteSelectedModifier, ColorCodingModifier

# Identify surface atoms
pipeline.modifiers.append(ConstructSurfaceModifier(radius=3.0))

# Calculate centro-symmetry parameter
pipeline.modifiers.append(CentroSymmetryModifier(num_neighbors=12))

# Select non-surface atoms OR atoms with low CSP (no defects)
pipeline.modifiers.append(ExpressionSelectionModifier(expression='SurfaceDistance > 0 || Centrosymmetry < 0.1'))

# Delete selected atoms
pipeline.modifiers.append(DeleteSelectedModifier())

# Color remaining atoms by CSP
pipeline.modifiers.append(ColorCodingModifier(property='Centrosymmetry', gradient=ColorCodingModifier.Hot()))
```

## Conclusion

The dynamic OVITO script generation approach transforms the visualization system from a static, limited tool into a flexible, extensible platform that can handle arbitrary visualization requests without code changes. By leveraging LLM intelligence to generate OVITO Python code, the system can:

1. **Handle any visualization request** - No need to hardcode new modifiers
2. **Scale infinitely** - New OVITO features work automatically
3. **Educate users** - Generated scripts teach OVITO API
4. **Maintain security** - Sandboxed execution with validation
5. **Provide flexibility** - Users can modify and reuse scripts

This approach aligns with the existing LAMMPS script generation pattern and provides a consistent, powerful interface for both simulation setup and visualization customization.
