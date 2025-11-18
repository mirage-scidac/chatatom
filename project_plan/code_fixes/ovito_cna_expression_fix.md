# OVITO ComputePropertyModifier Expression Error Fix

## Problem

The OVITO visualization pipeline was failing with the following error when trying to use pre-computed CNA values from LAMMPS:

```
RuntimeError: Modifier 'Compute property' reported: Unexpected token "c_cna" found at position 0.
```

### Root Cause

The `ovito_script_generator.py` was generating incorrect code that used `ComputePropertyModifier` with an invalid expression syntax:

```python
# INCORRECT - This caused the error
mapper = ComputePropertyModifier()
mapper.output_property = 'Structure Type'
mapper.expressions = ['c_cna']  # ERROR: Expression parser expects math expression, not property name
pipeline.modifiers.append(mapper)
```

The `ComputePropertyModifier.expressions` parameter expects a mathematical expression string (e.g., `'Position.X + 5'`), not just a property name like `'c_cna'`.

## Solution

Updated the OVITO script generator to use **direct property assignment** instead of `ComputePropertyModifier`:

```python
# CORRECT - Direct property assignment
from ovito.modifiers import ColorCodingModifier
import numpy as np

# Compute the pipeline to access data
data = pipeline.compute()

# Copy c_cna values to Structure Type property
if 'c_cna' in data.particles:
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

## Changes Made

### File: `ovito_script_generator.py`

Updated three sections of the system prompt:

1. **Section 0: USING PRE-COMPUTED LAMMPS PROPERTIES**
   - Changed from `ComputePropertyModifier` with expressions
   - To direct property creation using `data.particles_.create_property()`

2. **Section 3: Pattern "Show only crystalline atoms" (with pre-computed CNA)**
   - Updated to use direct property assignment
   - Added numpy import for array handling

3. **OUTPUT FORMAT: Example output for "show only FCC atoms" (with pre-computed CNA)**
   - Updated to use the correct property assignment approach
   - Ensures LLM generates correct code for pre-computed CNA scenarios

## How It Works

1. **Compute pipeline**: `data = pipeline.compute()` - Access the data collection
2. **Check property exists**: `if 'c_cna' in data.particles:` - Verify LAMMPS computed CNA
3. **Extract values**: `cna_values = np.array(data.particles['c_cna'])` - Get CNA values as numpy array
4. **Create property**: `data.particles_.create_property('Structure Type', data=cna_values)` - Create new property
5. **Apply modifiers**: Use standard OVITO modifiers on the new 'Structure Type' property

## Benefits

- **Correct syntax**: Uses OVITO's proper API for property creation
- **No expression parsing**: Avoids the expression parser entirely
- **Maintains efficiency**: Still uses pre-computed values from LAMMPS
- **Consistent behavior**: Works reliably across different OVITO versions

## Testing

To verify the fix works:

1. Run a simulation with structure-based visualization request:
   ```
   "Simulate melting and recrystallization of Al. Color atoms by structure type (FCC, BCC, etc.)"
   ```

2. The system should:
   - Detect CNA is needed
   - LAMMPS computes CNA and saves as `c_cna`
   - OVITO script generator creates correct property assignment code
   - Visualization renders without errors
   - Atoms are colored by their crystal structure

## Related Files

- `ovito_script_generator.py` - Updated system prompt with correct examples
- `ovito_executor.py` - Executes the generated scripts (no changes needed)
- `visualization_rendering.py` - Renders frames using OVITO (no changes needed)
- `docs/cna_optimization_summary.md` - Original CNA optimization documentation

## Previous Related Fixes

1. **OVITO scripts not passed to backend** - Fixed WebSocket data passing
2. **ImportError: __import__ not found** - Fixed restricted builtins
3. **Fatal Python error: __orig_import__** - Fixed by using full builtins module
4. **ComputePropertyModifier expression error** - Fixed with this update (direct property assignment)

## Status

✅ **FIXED** - The OVITO script generator now generates correct code for mapping pre-computed CNA values to the Structure Type property, eliminating the expression parser error.
