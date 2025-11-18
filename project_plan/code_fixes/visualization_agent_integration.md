# Visualization Agent Integration

## Overview

The Visualization agent has been integrated into the chat workflow to provide visible explanations of visualization settings, similar to how the Orchestrator, Generator, and Reviewer agents work.

## Architecture

### Components

1. **OVITOScriptGenerator** (`ovito_script_generator.py`)
   - LLM-powered agent that generates OVITO Python scripts
   - Inherits from base `Agent` class
   - Has deep knowledge of OVITO API and modifiers

2. **OrchestratorAgent.parse_visualization_request()** (`agents.py`)
   - Wrapper method that uses OVITOScriptGenerator
   - Parses natural language visualization requests
   - Returns structured visualization configuration

3. **Workflow Integration** (`app.py`)
   - Added "Visualization" agent phase in streaming endpoints
   - Streams visualization rationale and generated scripts to chat
   - Displays warnings and configuration details

### Workflow Sequence

```
User Request
    ↓
Orchestrator (analyzes request)
    ↓
Generator (creates LAMMPS script)
    ↓
Reviewer (validates script)
    ↓
Visualization (if viz keywords detected) ← NEW
    ↓
System (executes simulation)
```

## Implementation Details

### Detection Logic

The system detects visualization requests by checking for keywords:
- `visualize`, `visualization`, `show`, `display`, `render`
- `crystalline`, `crystal`, `structure`, `defect`, `surface`
- `color by`, `highlight`, `filter`, `only`

### Response Format

When a visualization request is detected, the Visualization agent:

1. **Streams rationale**: Explains the visualization approach
   ```
   Visualization approach: Using CommonNeighborAnalysisModifier to identify 
   crystal structures, then filtering to show only FCC atoms...
   ```

2. **Streams OVITO script**: Shows the generated Python code
   ```python
   from ovito.modifiers import CommonNeighborAnalysisModifier, SelectTypeModifier
   
   # Apply CNA to identify structures
   cna = CommonNeighborAnalysisModifier(cutoff=3.5)
   pipeline.modifiers.append(cna)
   ...
   ```

3. **Streams warnings**: Alerts about potential issues
   ```
   ⚠️ Script validation warning: Complex selection may be slow
   ```

### Frontend Integration

The frontend (`index.html`) displays the Visualization agent with:
- Icon: 🎨
- Color: Purple theme (`bg-purple-800 border-l-4 border-purple-400`)
- Message formatting: Code blocks for scripts, regular text for explanations

## Usage Examples

### Example 1: Filter to Crystalline Atoms

**User Request:**
```
Simulate copper at 300K and show only crystalline atoms
```

**Visualization Agent Response:**
```
🎨 Visualization
Visualization approach: Using CommonNeighborAnalysisModifier to identify 
crystal structures (FCC, HCP, BCC), then filtering out non-crystalline atoms.

Generated OVITO script:
```python
from ovito.modifiers import CommonNeighborAnalysisModifier, SelectTypeModifier, DeleteSelectedModifier, InvertSelectionModifier

# Apply CNA to identify crystal structures
cna = CommonNeighborAnalysisModifier(cutoff=3.5)
pipeline.modifiers.append(cna)

# Select crystalline atoms (FCC=1, HCP=2, BCC=3)
select = SelectTypeModifier(property='Structure Type', types={1, 2, 3})
pipeline.modifiers.append(select)

# Invert to select non-crystalline
pipeline.modifiers.append(InvertSelectionModifier())

# Delete non-crystalline atoms
pipeline.modifiers.append(DeleteSelectedModifier())
```
```

### Example 2: Highlight Defects

**User Request:**
```
Run aluminum simulation and highlight defects
```

**Visualization Agent Response:**
```
🎨 Visualization
Visualization approach: Using CentroSymmetryModifier to detect defects, 
then applying color coding to highlight high CSP values.

Generated OVITO script:
```python
from ovito.modifiers import CentroSymmetryModifier, ColorCodingModifier

# Apply centrosymmetry analysis
csp = CentroSymmetryModifier(num_neighbors=12)
pipeline.modifiers.append(csp)

# Color by CSP value (high values = defects)
color = ColorCodingModifier(
    property='Centrosymmetry',
    gradient=ColorCodingModifier.Hot()
)
pipeline.modifiers.append(color)
```
```

## Benefits

1. **Transparency**: Users see exactly what visualization settings are applied
2. **Education**: Users learn OVITO scripting through examples
3. **Debugging**: Generated scripts can be inspected and modified
4. **Consistency**: Visualization agent follows same pattern as other agents

## Configuration

The Visualization agent uses these config settings:

```python
# config.py
OVITO_SCRIPT_MODEL = "gpt4o"  # Model for OVITO script generation
OVITO_SCRIPT_TEMPERATURE = 0.1  # Low temperature for consistent scripts
```

## Code Changes

### app.py - Both Streaming Endpoints

Added visualization phase after reviewer in both `/api/llm-stream` and `/api/llm-stream-live`:

```python
# Check for visualization request and generate OVITO script if needed
viz_keywords = [
    'visualize', 'visualization', 'show', 'display', 'render',
    'crystalline', 'crystal', 'structure', 'defect', 'surface',
    'color by', 'highlight', 'filter', 'only'
]

has_viz_request = any(keyword in user_command.lower() for keyword in viz_keywords)

if has_viz_request:
    yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Visualization'})}\n\n"
    
    # Parse visualization request using orchestrator
    viz_result = orchestrator.parse_visualization_request(user_command, {
        'material': 'Unknown',
        'structure': 'Unknown'
    })
    
    # Stream rationale, script, and warnings...
```

### index.html - Frontend Support

Added Visualization agent case to all switch statements:

```javascript
case 'Visualization':
    colorClass = 'bg-purple-800 border-l-4 border-purple-400';
    agentIcon = '🎨';
    break;
```

## Testing

Run the test suite:
```bash
# Activate conda environment first
conda activate chatmd

# Run test
python test_visualization_agent.py
```

This validates:
- Visualization request detection
- Script generation for common patterns
- Proper handling of non-visualization requests
- Error handling and warnings

## Future Enhancements

Potential improvements:
1. Interactive script editing before execution
2. Visualization preset library
3. Multi-step visualization workflows
4. Real-time preview of visualization settings
5. Visualization performance metrics
6. User-customizable visualization templates
