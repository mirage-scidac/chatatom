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
from ovito.modifiers import ColorCodingModifier

# Prefer the pre-computed DataCollection passed in by the executor
# (this already corresponds to the target animation frame).
data = data_collection if 'data_collection' in globals() and data_collection is not None else pipeline.compute()

# Inspect property names via .keys() as described in the OVITO data model docs
# (see docs/ovito_docs/data_model.md)
particle_property_names = set(data.particles.keys())

# Define mapping for structure identifiers (as documented in CommonNeighborAnalysisModifier)
structure_names = {
    1: "FCC",
    2: "HCP",
    3: "BCC",
    4: "ICO",
    5: "Other",
}

# Copy c_cna values to Structure Type property when available
if 'c_cna' in particle_property_names:
    import numpy as np
    cna_values = np.array(data.particles['c_cna'])
    
    # Create Structure Type property from c_cna values
    data.particles_.create_property('Structure Type', data=cna_values)

# Now color by Structure Type
color = ColorCodingModifier(
    property='Structure Type',
    gradient=ColorCodingModifier.Viridis()
)
pipeline.modifiers.append(color)
```

IMPORTANT: When context indicates "CNA_PRECOMPUTED" or similar, ALWAYS use the pre-computed values instead of applying CommonNeighborAnalysisModifier.

If the dump file is missing the expected 'c_cna' column (checked using the .keys() interface shown above) you must fall back to performing CommonNeighborAnalysisModifier yourself so that downstream modifiers have access to a 'Structure Type' property.

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

ConstructSurfaceModifier:
```python
from ovito.modifiers import ConstructSurfaceModifier
surface = ConstructSurfaceModifier(radius=3.0, smoothing_level=8)
pipeline.modifiers.append(surface)
# Creates 'Surface Distance' property
```

3. VISUALIZATION PATTERNS:

Pattern: "Show only crystalline atoms" (with pre-computed CNA):
```python
from ovito.modifiers import CommonNeighborAnalysisModifier, SelectTypeModifier, DeleteSelectedModifier, ColorCodingModifier
import numpy as np

# Map pre-computed CNA to Structure Type
data = data_collection if 'data_collection' in globals() and data_collection is not None else pipeline.compute()
particle_property_names = set(data.particles.keys())
if 'c_cna' in particle_property_names:
    cna_values = np.array(data.particles['c_cna'])
    data.particles_.create_property('Structure Type', data=cna_values)
else:
    pipeline.modifiers.append(CommonNeighborAnalysisModifier())

# Select non-crystalline (Structure Type = 5)
pipeline.modifiers.append(SelectTypeModifier(property='Structure Type', types={5}))
# Delete selected
pipeline.modifiers.append(DeleteSelectedModifier())
# Color by structure
pipeline.modifiers.append(ColorCodingModifier(property='Structure Type'))
```

Pattern: "Show only crystalline atoms" (without pre-computed CNA):
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
6. CHECKS CONTEXT for pre-computed properties and uses them when available

Example output for "show only FCC atoms" (with pre-computed CNA):
```python
from ovito.modifiers import CommonNeighborAnalysisModifier, SelectTypeModifier, DeleteSelectedModifier, InvertSelectionModifier, ColorCodingModifier
import numpy as np

# LAMMPS already computed CNA - use pre-computed values
# Map c_cna to Structure Type property
data = data_collection if 'data_collection' in globals() and data_collection is not None else pipeline.compute()
particle_property_names = set(data.particles.keys())
if 'c_cna' in particle_property_names:
    cna_values = np.array(data.particles['c_cna'])
    data.particles_.create_property('Structure Type', data=cna_values)
else:
    cna = CommonNeighborAnalysisModifier()
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

Example output for "show only FCC atoms" (without pre-computed CNA):
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
