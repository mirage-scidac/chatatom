# Parallel LAMMPS Implementation with MPI

## Overview

This document describes the implementation of parallel LAMMPS execution using the Python MPI interface with `mpi4py` and `gather_atoms()` for real-time visualization support.

## Problem Statement

The original implementation had a critical limitation: when MPI was enabled for parallel execution, visualization was disabled because:

1. **Subprocess MPI** (`mpirun -np N lmp`) couldn't access LAMMPS internal data structures
2. **No gather_atoms()** capability meant frame data couldn't be collected from distributed ranks
3. **Forced serial fallback** whenever visualization was needed (lines 1327-1333 in tools.py)

## Solution: Python MPI Interface

We implemented a new execution path that uses the LAMMPS Python API with MPI communicator passing:

```python
from mpi4py import MPI
from lammps import lammps

comm = MPI.COMM_WORLD
lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none'], comm=comm)
```

This enables:
- Direct MPI communicator passing to LAMMPS
- `gather_atoms()` for collecting distributed data
- Real-time visualization with parallel execution
- Proper rank synchronization

## Implementation Details

### Core Functions

#### 1. `extract_lammps_frame_mpi(lmp, comm)`
Extracts frame data using MPI gather operations:
- All ranks participate in `gather_atoms()`
- Only rank 0 processes and returns complete data
- Gathers positions, types, velocities, forces

```python
positions = lmp.gather_atoms("x", 1, 3)  # All ranks participate
types = lmp.gather_atoms("type", 0, 1)

if me == 0:
    # Only rank 0 processes data
    frame_data = {
        'natoms': natoms,
        'positions': np.array(positions).reshape((natoms, 3)),
        'types': np.array(types, dtype=int),
        ...
    }
    return frame_data
else:
    return None
```

#### 2. `run_lammps_mpi_python(script_string, render_config, data_callback, num_cores)`
Main parallel execution function:
- Initializes MPI communicator
- Passes communicator to LAMMPS
- Executes simulation in chunks for visualization
- Uses `extract_lammps_frame_mpi()` to gather data
- Rank 0 handles all visualization and callbacks
- Proper synchronization with `comm.Barrier()`

#### 3. Integration with `run_lammps_with_async_visualization()`
Automatically selects execution mode:
```python
if use_mpi:
    print("Using Python MPI interface for parallel execution with visualization")
    return run_lammps_mpi_python(script_string, render_config, data_callback, cores)
```

## Architecture

```
User Request (cores > 1)
    ↓
run_lammps_with_async_visualization()
    ↓
Detects MPI needed
    ↓
run_lammps_mpi_python()
    ↓
┌─────────────────────────────────────┐
│ All Ranks:                          │
│ - Initialize MPI communicator       │
│ - Create LAMMPS with comm           │
│ - Execute simulation commands       │
│ - Participate in gather_atoms()     │
│ - Synchronize with Barrier()        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Rank 0 Only:                        │
│ - Process gathered data             │
│ - Create OVITO frames               │
│ - Render visualizations             │
│ - Stream to WebSocket               │
│ - Handle file operations            │
└─────────────────────────────────────┘
```

## Key Features

### 1. Automatic Mode Selection
- Serial execution (cores = 1): Uses Python LAMMPS interface
- Parallel execution (cores > 1): Uses Python MPI interface
- Fallback to serial if MPI unavailable

### 2. Real-Time Visualization
- Frame extraction at regular intervals
- Rank 0 handles all rendering
- WebSocket streaming to browser
- No performance impact on compute ranks

### 3. Proper Synchronization
- `comm.Barrier()` after each chunk
- Ensures all ranks stay in sync
- Prevents race conditions
- Clean shutdown on all ranks

### 4. Error Handling
- Graceful fallback if mpi4py unavailable
- Proper cleanup on all ranks
- Error messages only from rank 0

## Usage Examples

### Basic Parallel Execution
```python
from tools import run_lammps_with_async_visualization

# Automatically uses MPI if cores > 1
output_files, thermo_data, frames = run_lammps_with_async_visualization(
    script_string=lammps_script,
    render_config={'quality': 'medium'},
    data_callback=my_callback,
    num_cores=4  # Uses Python MPI interface
)
```

### Direct MPI Function
```python
from tools import run_lammps_mpi_python

output_files, thermo_data, frames = run_lammps_mpi_python(
    script_string=lammps_script,
    render_config={'quality': 'high'},
    data_callback=my_callback,
    num_cores=4
)
```

### Example Script
See `examples/parallel_gather_example.py` for a standalone demonstration:
```bash
mpirun -np 4 python examples/parallel_gather_example.py
```

## Performance Considerations

### Benefits
- **Parallel speedup**: Simulation runs on multiple cores
- **Efficient data gathering**: `gather_atoms()` is optimized
- **Minimal overhead**: Only rank 0 handles visualization
- **Scalable**: Works with any number of ranks

### Limitations
- **Memory on rank 0**: Complete atom arrays stored on rank 0
- **Gather overhead**: Data collection has communication cost
- **Visualization serial**: Rendering only on rank 0

### Optimization Tips
1. **Reduce frame frequency**: Use smart interval calculation
2. **Limit max frames**: Default 100 frames prevents memory issues
3. **Adjust quality**: Lower quality for faster rendering
4. **Scale appropriately**: More cores for larger systems

## Configuration

### Enable MPI
In `config.py`:
```python
LAMMPS_USE_MPI = True
LAMMPS_CORES = 4
LAMMPS_MPI_COMMAND = "mpirun"
```

### Check MPI Availability
```python
from tools import check_mpi_available

if check_mpi_available():
    print("MPI is available")
```

## Testing

### Unit Tests
See `tests/test_parallel_mpi.py` (to be created):
- Test gather_atoms() functionality
- Verify rank synchronization
- Check data integrity
- Test fallback behavior

### Integration Tests
- Run with different core counts
- Verify visualization works
- Check performance scaling
- Test error handling

## Troubleshooting

### Common Issues

**1. mpi4py not found**
```
Error: mpi4py not available. Falling back to serial execution.
```
Solution: Install mpi4py: `pip install mpi4py`

**2. MPI not available**
```
Warning: MPI not available. Falling back to serial execution.
```
Solution: Install MPI (e.g., `brew install open-mpi` on macOS)

**3. Visualization fails**
- Check that only rank 0 is rendering
- Verify OVITO is installed
- Check render_config parameters

**4. Synchronization issues**
- Ensure all ranks execute same commands
- Check Barrier() placement
- Verify file operations only on rank 0

## Future Enhancements

### Planned Features
1. **Distributed rendering**: Render on multiple ranks
2. **Async gather**: Non-blocking data collection
3. **Compression**: Reduce data transfer overhead
4. **Checkpointing**: Save/restore parallel state

### Performance Improvements
1. **Selective gathering**: Only gather needed properties
2. **Lazy evaluation**: Defer processing until needed
3. **Caching**: Reuse gathered data when possible
4. **Streaming**: Progressive data transfer

## References

- [LAMMPS Python Interface](https://docs.lammps.org/Python_head.html)
- [mpi4py Documentation](https://mpi4py.readthedocs.io/)
- [LAMMPS gather_atoms()](https://docs.lammps.org/Python_module.html#lammps.lammps.gather_atoms)
- [MPI Communicators](https://www.mpi-forum.org/docs/)

## Changelog

### 2025-01-06
- Initial implementation of Python MPI interface
- Added `run_lammps_mpi_python()` function
- Implemented `extract_lammps_frame_mpi()`
- Integrated with async visualization pipeline
- Created example script
- Removed MPI fallback that disabled visualization
