"""
OVITO Script Generator - LLM-powered dynamic OVITO Python script generation.

This module provides an agent that generates complete OVITO Python scripts
from natural language visualization requests, similar to how LAMMPS scripts
are generated dynamically.
"""

from agents import Agent
import config


class OVITOScriptGenerator(Agent):
    """
    LLM-powered generator for OVITO Python scripts.
    
    This agent generates complete, executable OVITO Python code based on 
    natural language visualization requests. It has deep knowledge of OVITO's 
    API, modifiers, and best practices for molecular dynamics visualization.
    """
    
    def __init__(self):
        system_prompt = """You are an expert in OVITO Python scripting for molecular dynamics visualization.

ROLE:
Generate complete, executable OVITO Python scripts based on natural language requests.
You have deep knowledge of OVITO's API, modifiers, and best practices.

CRITICAL: AVOID REDUNDANT COMPUTATION
When the LAMMPS simulation has already computed analysis properties (like CNA), you MUST use those pre-computed values instead of re-computing them in OVITO. This is more efficient and ensures consistency.

OVITO PYTHON API KNOWLEDGE:

0. USING PRE-COMPUTED LAMMPS PROPERTIES:

If LAMMPS has already computed properties (indicated by context), use them directly:

Example - Using pre-computed CNA from LAMMPS:
```python
from ovito.modifiers import PythonModifier, ColorCodingModifier, CommonNeighborAnalysisModifier
import numpy as np

# Mapping documented in CommonNeighborAnalysisModifier (docs/ovito_docs/modifiers.md)
structure_names = {
    1: "FCC",
    2: "HCP",
    3: "BCC",
    4: "ICO",
    5: "Other",
}

# Guard modifier (runs every frame) to ensure 'Structure Type' exists before other modifiers
def ensure_structure_type(frame, data):
    particle_property_names = set(data.particles.keys())  # uses PropertyContainer.keys() (docs/ovito_docs/data_model.md)
    if 'Structure Type' in particle_property_names:
        return

    if 'c_cna' in particle_property_names:
        cna_values = np.array(data.particles['c_cna'])
        data.particles_.create_property('Structure Type', data=cna_values)
        return

    # Fallback: compute CNA inside OVITO for this frame only (docs/ovito_docs/user_modifier.md)
    cna_modifier = CommonNeighborAnalysisModifier(
        mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff,
        cutoff=3.8,
    )
    data.apply(cna_modifier)  # In-place application keeps pipeline state per-frame

pipeline.modifiers.append(PythonModifier(function=ensure_structure_type))


def create_structure_color_modifier():
    # Fixed mapping keeps colors stable across frames (docs/ovito_docs/modifiers.md)
    modifier = ColorCodingModifier(
        property='Structure Type',
        gradient=ColorCodingModifier.Viridis(),
    )
    modifier.auto_adjust_range = False  # keep legend stable across frames
    modifier.start_value = 1
    modifier.end_value = 5
    modifier.discretize_color_map = True
    return modifier


# Downstream modifiers can rely on Structure Type being present each frame
pipeline.modifiers.append(create_structure_color_modifier())

# Reuse create_structure_color_modifier() anywhere you need consistent structure coloring.
```

IMPORTANT: When context indicates "CNA_PRECOMPUTED" or similar, ALWAYS use the pre-computed values instead of applying CommonNeighborAnalysisModifier.

If the dump file is missing the expected 'c_cna' column, the PythonModifier guard shown above must invoke CommonNeighborAnalysisModifier (or an equivalent structure identification step) so that downstream modifiers still receive a valid 'Structure Type' property without emitting warnings.

PER-FRAME PROPERTY GUARDS (CRITICAL):

OVITO recomputes the entire modifier stack for every requested animation frame. Any property you create just once (e.g., by calling pipeline.compute() outside the modifier chain) will not persist across frames. To avoid “property missing” fallbacks, wrap every derived-property requirement in a modifier that executes per frame:

1. Use PythonModifier (docs/ovito_docs/user_modifier.md) when you simply need to copy, rename, or aggregate existing per-atom data (e.g., copying c_cna into Structure Type, computing a selection flag, calculating statistics, snapshot logging).
2. Use built-in analysis modifiers (CommonNeighborAnalysisModifier, CentroSymmetryModifier, DislocationAnalysisModifier, etc.) inside that PythonModifier via `data.apply(...)` (in-place) or by appending them directly to the pipeline so they execute each frame.
3. Never rely on a one-off `pipeline.compute()` side-effect to add properties, because the executor will call compute() again for each frame and those properties will vanish.
4. When multiple downstream modifiers depend on the same derived property, install exactly one guard modifier at the *start* of the pipeline to prepare that property, and let later steps assume it exists.

This pattern applies broadly: if you need per-frame selections, classifications, statistics, or color channels, ensure a modifier establishes the required data during every pipeline evaluation.

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
color = create_structure_color_modifier()  # Helper keeps palette stable across frames
pipeline.modifiers.append(color)
```
Disabling `auto_adjust_range` and choosing fixed `start_value`/`end_value` keeps the legend and colors stable across frames instead of re-scaling per timestep (see docs/ovito_docs/modifiers.md, ColorCodingModifier.auto_adjust_range).

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

ConstructSurfaceModifier:
```python
from ovito.modifiers import ConstructSurfaceModifier
surface = ConstructSurfaceModifier(radius=3.0, smoothing_level=8)
pipeline.modifiers.append(surface)
# Creates 'Surface Distance' property
```

3. VISUALIZATION PATTERNS:

Pattern: "Show only crystalline atoms":
```python
from ovito.modifiers import PythonModifier, SelectTypeModifier, DeleteSelectedModifier

pipeline.modifiers.append(PythonModifier(function=ensure_structure_type))
pipeline.modifiers.append(SelectTypeModifier(property='Structure Type', types={5}))
pipeline.modifiers.append(DeleteSelectedModifier())
pipeline.modifiers.append(create_structure_color_modifier())
```
The guard handles both pre-computed and on-demand CNA scenarios, so no duplicate workflows are needed.

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
6. CHECKS CONTEXT for pre-computed properties and uses them when available

Example: "Show only FCC atoms":
```python
from ovito.modifiers import PythonModifier, SelectTypeModifier, DeleteSelectedModifier, InvertSelectionModifier

pipeline.modifiers.append(PythonModifier(function=ensure_structure_type))
select_fcc = SelectTypeModifier(property='Structure Type', types={1})
pipeline.modifiers.append(select_fcc)
pipeline.modifiers.append(InvertSelectionModifier())
pipeline.modifiers.append(DeleteSelectedModifier())
pipeline.modifiers.append(create_structure_color_modifier())
```
Chain additional modifiers after these core steps as required (e.g., logging, further selections, or alternative color schemes built off the guard-prepared properties).

IMPORTANT:
- Generate ONLY the Python code, no markdown formatting
- Assume 'pipeline' variable is already created
- Include all necessary imports
- Add comments for clarity
- Handle edge cases appropriately
"""
        
        super().__init__(
            system_prompt,
            model=config.OVITO_SCRIPT_MODEL,  # Uses dedicated OVITO script model
            default_temperature=config.OVITO_SCRIPT_TEMPERATURE
        )
    
    def generate_ovito_script(self, user_request: str, simulation_context: dict = None) -> str:
        """
        Generate OVITO Python script from natural language request.
        
        Args:
            user_request: Natural language visualization request
            simulation_context: Optional simulation context (material, structure, etc.)
            
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
            
            # Check if CNA is pre-computed in LAMMPS
            if simulation_context.get('cna_precomputed', False):
                context_str += "\n⚠️ IMPORTANT: CNA_PRECOMPUTED - LAMMPS already computed CNA values (available as 'c_cna' property)\n"
                context_str += "You MUST use the pre-computed c_cna values instead of applying CommonNeighborAnalysisModifier.\n"
        
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
