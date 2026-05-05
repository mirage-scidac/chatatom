"""
Standalone helpers for generating and validating Atomsk structures.

This module is intentionally not wired into the ChatMD request harness yet.
It provides a small, high-level wrapper surface for Atomsk workflows that can
be called by future skills/tools or by tests.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Iterable, Optional, Sequence


class AtomskError(RuntimeError):
    """Raised when Atomsk is unavailable or an Atomsk workflow fails."""


@dataclass(frozen=True)
class AtomskCommandResult:
    """Captured result from an Atomsk subprocess invocation."""

    command: Annotated[
        tuple[str, ...],
        "Full command tuple that was executed, including the Atomsk executable.",
    ]
    cwd: Annotated[
        Path,
        "Resolved working directory used for the Atomsk subprocess.",
    ]
    stdout: Annotated[
        str,
        "Captured Atomsk standard output.",
    ]
    stderr: Annotated[
        str,
        "Captured Atomsk standard error.",
    ]


@dataclass(frozen=True)
class GrainNode:
    """One Atomsk polycrystal grain node."""

    x: Annotated[
        str | float,
        "Atomsk node x coordinate, either absolute Angstrom value or expression such as 0.5*box.",
    ]
    y: Annotated[
        str | float,
        "Atomsk node y coordinate, either absolute Angstrom value or expression such as 0.5*box.",
    ]
    z: Annotated[
        str | float,
        "Atomsk node z coordinate, either absolute Angstrom value or expression such as 0.5*box.",
    ]
    orientation: Annotated[
        str,
        "Atomsk grain orientation string, typically three crystallographic directions.",
    ]

    def to_atomsk_line(self) -> str:
        return (
            f"node {_format_atomsk_value(self.x)} "
            f"{_format_atomsk_value(self.y)} "
            f"{_format_atomsk_value(self.z)} "
            f"{self.orientation}"
        )


@dataclass(frozen=True)
class AtomskStructureValidation:
    """Validation summary for an Atomsk-generated LAMMPS data file."""

    data_file: Annotated[
        Path,
        "LAMMPS data file that was validated.",
    ]
    atom_count: Annotated[
        int,
        "Atom count parsed from the LAMMPS data header.",
    ]
    atom_type_count: Annotated[
        int,
        "Atom type count parsed from the LAMMPS data header.",
    ]
    box_bounds: Annotated[
        dict[str, tuple[float, float]],
        "Parsed x, y, and z box bounds keyed by axis.",
    ]
    masses: Annotated[
        dict[int, float],
        "Masses parsed from the LAMMPS data file keyed by atom type.",
    ]
    grain_count: Annotated[
        Optional[int],
        "Grain count parsed from Atomsk auxiliary files, if available.",
    ]
    auxiliary_files: Annotated[
        tuple[Path, ...],
        "Existing Atomsk auxiliary files discovered next to the data file.",
    ]
    issues: Annotated[
        tuple[str, ...],
        "Validation issue messages; empty when the structure is valid.",
    ]

    @property
    def is_valid(self) -> bool:
        return not self.issues


DEFAULT_HEX_GRAIN_ORIENTATIONS = (
    "[2_2_-5] [5_5_4] [1_-1_0]",
    "[4_4_-1] [1_1_8] [1_-1_0]",
    "[1_1_1] [1_1_-2] [1_-1_0]",
    "[-1_-1_1] [1_1_2] [1_-1_0]",
)


def find_atomsk_executable() -> Path:
    """
    Locate a usable Atomsk executable from ATOMSK_EXECUTABLE.

    We intentionally do not embed project-local or machine-local fallback paths
    here. Runtime setup must make the executable path explicit.
    """
    env_path = os.environ.get("ATOMSK_EXECUTABLE")
    if not env_path:
        raise AtomskError("ATOMSK_EXECUTABLE is not set. Set it to the Atomsk executable path.")

    resolved = Path(env_path).expanduser().resolve()
    if _is_usable_atomsk(resolved):
        return resolved
    raise AtomskError(
        f"ATOMSK_EXECUTABLE is not a runnable Atomsk executable: {resolved}. "
        "Check that the file exists, has execute permission, and supports '--version'."
    )


def run_atomsk_command(
    args: Annotated[
        Sequence[str | os.PathLike[str]],
        "Atomsk command arguments excluding the executable path.",
    ],
    cwd: Annotated[
        str | os.PathLike[str],
        "Working directory where Atomsk should run and write outputs.",
    ],
    atomsk_executable: Annotated[
        str | os.PathLike[str] | None,
        "Optional explicit Atomsk executable path; defaults to ATOMSK_EXECUTABLE discovery.",
    ] = None,
) -> AtomskCommandResult:
    """
    Run Atomsk with a safe argument list and captured output.

    The command is executed without a shell, so callers should pass each Atomsk
    flag and value as a separate item in ``args``. The working directory is
    created if needed. A non-zero Atomsk exit code raises ``AtomskError`` with
    captured stdout and stderr.
    """
    cwd_path = Path(cwd)
    cwd_path.mkdir(parents=True, exist_ok=True)

    executable = Path(atomsk_executable).resolve() if atomsk_executable else find_atomsk_executable()
    command = tuple([str(executable), *[str(arg) for arg in args]])

    completed = subprocess.run(
        command,
        cwd=str(cwd_path),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AtomskError(
            f"Atomsk failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    return AtomskCommandResult(
        command=command,
        cwd=cwd_path.resolve(),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def create_unit_cell(
    output_dir: Annotated[
        str | os.PathLike[str],
        "Directory where the Atomsk seed structure should be written.",
    ],
    lattice: Annotated[
        str,
        "Atomsk lattice keyword, such as fcc, bcc, hcp, diamond, or rocksalt.",
    ],
    lattice_parameter: Annotated[
        float,
        "Primary lattice parameter in Angstrom.",
    ],
    species: Annotated[
        str | Sequence[str],
        "Atomic species symbols required by the selected Atomsk lattice.",
    ],
    output_filename: Annotated[
        str | None,
        "Seed output filename, usually an XSF file; defaults to species name plus .xsf.",
    ] = None,
    orientation: Annotated[
        Sequence[str] | None,
        "Optional three crystallographic direction strings following Atomsk orient syntax.",
    ] = None,
    atomsk_executable: Annotated[
        str | os.PathLike[str] | None,
        "Optional explicit Atomsk executable path; defaults to ATOMSK_EXECUTABLE discovery.",
    ] = None,
) -> Path:
    """
    Create an Atomsk seed unit cell, typically an XSF file.

    This wraps ``atomsk --create`` for the common seed-generation step used
    before polycrystal construction. The returned path points to the generated
    seed file.
    """
    species_list = [species] if isinstance(species, str) else list(species)
    if not species_list:
        raise ValueError("At least one species is required")

    output_name = output_filename or f"{''.join(species_list)}.xsf"
    args: list[str | os.PathLike[str]] = [
        "--create",
        lattice,
        str(lattice_parameter),
        *species_list,
    ]
    if orientation is not None:
        if len(orientation) != 3:
            raise ValueError("orientation must contain exactly three direction strings")
        args.extend(["orient", *orientation])
    args.extend([output_name, "-ow"])

    run_atomsk_command(args, output_dir, atomsk_executable=atomsk_executable)
    return Path(output_dir).resolve() / output_name


def write_polycrystal_node_file(
    path: Annotated[
        str | os.PathLike[str],
        "Destination path for the Atomsk polycrystal node parameter file.",
    ],
    box: Annotated[
        tuple[float, float, float],
        "Final polycrystal box lengths in Angstrom along x, y, and z.",
    ],
    nodes: Annotated[
        Sequence[GrainNode],
        "Explicit Atomsk grain nodes with normalized or absolute coordinates and orientations.",
    ],
) -> Path:
    """
    Write an Atomsk polycrystal parameter file with explicit grain nodes.

    The file contains one ``box`` line followed by one ``node`` line per grain,
    matching Atomsk's ``--polycrystal`` parameter-file format.
    """
    if not nodes:
        raise ValueError("At least one grain node is required")

    param_path = Path(path)
    param_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"box {box[0]} {box[1]} {box[2]}"]
    lines.extend(node.to_atomsk_line() for node in nodes)
    param_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return param_path


def create_polycrystal_from_nodes(
    output_dir: Annotated[
        str | os.PathLike[str],
        "Directory where Atomsk should create the polycrystal outputs.",
    ],
    seed_file: Annotated[
        str | os.PathLike[str],
        "Atomsk-readable seed structure, typically produced by create_unit_cell.",
    ],
    box: Annotated[
        tuple[float, float, float],
        "Final polycrystal box lengths in Angstrom along x, y, and z.",
    ],
    nodes: Annotated[
        Sequence[GrainNode],
        "Explicit grain nodes used to define the Voronoi polycrystal.",
    ],
    output_prefix: Annotated[
        str,
        "Output prefix for Atomsk-generated files before the .lmp extension.",
    ] = "data.hexslab",
    duplicate: Annotated[
        tuple[int, int, int],
        "Integer repetitions to apply along x, y, and z after polycrystal generation.",
    ] = (1, 1, 1),
    param_filename: Annotated[
        str,
        "Filename for the generated Atomsk polycrystal node parameter file.",
    ] = "atomsk_box_commands.txt",
    atomsk_executable: Annotated[
        str | os.PathLike[str] | None,
        "Optional explicit Atomsk executable path; defaults to ATOMSK_EXECUTABLE discovery.",
    ] = None,
) -> Path:
    """
    Create a LAMMPS data polycrystal using explicit Atomsk nodes.

    This writes the Atomsk node parameter file, runs ``atomsk --polycrystal``
    with wrapping and optional duplication, and returns the generated ``.lmp``
    data-file path.
    """
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    param_path = write_polycrystal_node_file(output_path / param_filename, box, nodes)

    args: list[str | os.PathLike[str]] = [
        "--polycrystal",
        Path(seed_file),
        param_path.name,
        "-wrap",
        "-duplicate",
        str(duplicate[0]),
        str(duplicate[1]),
        str(duplicate[2]),
        output_prefix,
        "lmp",
        "-ow",
    ]
    run_atomsk_command(args, output_path, atomsk_executable=atomsk_executable)
    return output_path / f"{output_prefix}.lmp"


def create_hex_grains(
    output_parent: Annotated[
        str | os.PathLike[str],
        "Parent directory where the hex-grain output folder should be created.",
    ],
    element: Annotated[
        str,
        "Chemical element symbol for the generated grains.",
    ] = "Cu",
    lattice: Annotated[
        str,
        "Atomsk lattice keyword used for the seed unit cell.",
    ] = "fcc",
    lattice_parameter: Annotated[
        float,
        "Seed lattice parameter in Angstrom.",
    ] = 3.615,
    hex_size: Annotated[
        int,
        "Nominal hexagonal grain size in Angstrom used to derive box dimensions.",
    ] = 400,
    dupe_z: Annotated[
        int,
        "Number of repetitions to duplicate along the z direction.",
    ] = 5,
    hex_grid: Annotated[
        int,
        "Number of grains along each in-plane grid direction; currently only 2 is supported.",
    ] = 2,
    orientations: Annotated[
        Sequence[str],
        "Crystallographic orientation strings, one for each grain node.",
    ] = DEFAULT_HEX_GRAIN_ORIENTATIONS,
    z_positions: Annotated[
        Sequence[float] | None,
        "Optional normalized z positions for grain nodes; defaults to deterministic 0.5 values.",
    ] = None,
    output_dir_name: Annotated[
        str | None,
        "Optional output directory name; defaults to gs<size>_nhex<count>_zrep<dupe_z>.",
    ] = None,
    atomsk_executable: Annotated[
        str | os.PathLike[str] | None,
        "Optional explicit Atomsk executable path; defaults to ATOMSK_EXECUTABLE discovery.",
    ] = None,
) -> Path:
    """
    Reproduce the high-level Griffith hex-grain Atomsk workflow.

    By default this uses deterministic z positions to make tests and
    regeneration stable. Pass explicit z_positions to match a chosen reference.
    """
    if hex_grid != 2:
        raise NotImplementedError("create_hex_grains currently supports the 2x2 reference grid")
    if len(orientations) != hex_grid * hex_grid:
        raise ValueError("orientations must contain one orientation per grain")

    grain_count = hex_grid * hex_grid
    output_name = output_dir_name or f"gs{hex_size}_nhex{grain_count}_zrep{dupe_z}"
    output_dir = Path(output_parent).resolve() / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_file = create_unit_cell(
        output_dir,
        lattice=lattice,
        lattice_parameter=lattice_parameter,
        species=element,
        output_filename=f"{element}.xsf",
        atomsk_executable=atomsk_executable,
    )

    nodes = make_hex_grain_nodes(hex_size, hex_grid, orientations, z_positions=z_positions)
    box_size = grain_count / 2 * hex_size
    box_z = lattice_parameter * 2**0.5

    data_file = create_polycrystal_from_nodes(
        output_dir=output_dir,
        seed_file=seed_file,
        box=(box_size, box_size, box_z),
        nodes=nodes,
        output_prefix="data.hexslab",
        duplicate=(1, 1, dupe_z),
        atomsk_executable=atomsk_executable,
    )

    write_atomsk_command_log(
        output_dir / "atomsk_lattice_commands.txt",
        [
            f"atomsk --create {lattice} {lattice_parameter} {element} {element}.xsf -ow",
            (
                "atomsk --polycrystal "
                f"{element}.xsf atomsk_box_commands.txt -wrap "
                f"-duplicate 1 1 {dupe_z} data.hexslab lmp -ow"
            ),
        ],
    )
    write_grain_lammps_vars(data_file.with_suffix(""), dupe_z=dupe_z)
    return data_file


def make_hex_grain_nodes(
    hex_size: Annotated[
        int,
        "Nominal hexagonal grain size in Angstrom; retained for workflow parity.",
    ],
    hex_grid: Annotated[
        int,
        "Number of grains along each in-plane grid direction.",
    ],
    orientations: Annotated[
        Sequence[str],
        "Crystallographic orientation strings, one per grain.",
    ],
    z_positions: Annotated[
        Sequence[float] | None,
        "Optional normalized z positions, one per grain.",
    ] = None,
) -> tuple[GrainNode, ...]:
    """
    Build the hex-grain node layout used by the reference script.

    The returned nodes are reversed to match the original
    ``generate_hex_v0.py`` output order. Coordinates are normalized with
    ``*box`` expressions so Atomsk scales them to the final box.
    """
    del hex_size  # The node positions are normalized against box dimensions.
    grain_count = hex_grid * hex_grid
    if z_positions is None:
        z_positions = tuple(0.5 for _ in range(grain_count))
    if len(z_positions) != grain_count:
        raise ValueError("z_positions must contain one value per grain")

    incr = 1 / hex_grid
    nodes: list[GrainNode] = []
    orientation_index = 0
    for n in range(hex_grid):
        x = n / hex_grid
        for m in range(hex_grid):
            if n % 2 == 0:
                y = (m + 1) * incr - incr / 2
            else:
                y = m * incr
            nodes.append(
                GrainNode(
                    x=f"{_trim_float(x)}*box",
                    y=f"{_trim_float(y)}*box",
                    z=f"{_trim_float(z_positions[orientation_index])}*box",
                    orientation=orientations[orientation_index],
                )
            )
            orientation_index += 1

    return tuple(reversed(nodes))


def write_atomsk_command_log(
    path: Annotated[
        str | os.PathLike[str],
        "Destination path for the Atomsk command provenance log.",
    ],
    commands: Annotated[
        Iterable[str],
        "Atomsk command lines sufficient to reproduce the generated structure.",
    ],
) -> Path:
    """
    Write reproducible Atomsk command provenance.

    The command log is informational and should mirror the high-level wrapper
    calls closely enough for a user to inspect or rerun the Atomsk workflow.
    """
    command_path = Path(path)
    command_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    return command_path


def write_grain_lammps_vars(
    output_prefix: Annotated[
        str | os.PathLike[str],
        "Atomsk output prefix without .lmp, used to locate auxiliary grain files.",
    ],
    dupe_z: Annotated[
        int,
        "Number of z-direction duplicates used during Atomsk generation.",
    ],
    min_gb_dist: Annotated[
        float,
        "Minimum grain-boundary distance in Angstrom used in cylinder-radius estimation.",
    ] = 15.0,
    output_filename: Annotated[
        str,
        "LAMMPS variables filename to write next to the Atomsk outputs.",
    ] = "in.vars_grains",
) -> Path:
    """
    Write a LAMMPS grain variable file from Atomsk polycrystal metadata.

    This consumes Atomsk's ``*_nodes.xsf`` and ``*_id-size.txt`` files and
    reproduces the grain-variable output generated by the reference workflow.
    """
    prefix = Path(output_prefix)
    nodes_file = prefix.parent / f"{prefix.name}_nodes.xsf"
    size_file = prefix.parent / f"{prefix.name}_id-size.txt"
    output_path = prefix.parent / output_filename

    if not nodes_file.exists():
        raise AtomskError(f"Missing Atomsk nodes file: {nodes_file}")
    if not size_file.exists():
        raise AtomskError(f"Missing Atomsk grain size file: {size_file}")

    node_lines = nodes_file.read_text(encoding="utf-8").splitlines()
    scaled_z = float(node_lines[5].strip().split()[-1]) / dupe_z

    grain_count, grain_rows = _read_grain_size_rows(size_file)
    com_grains = _read_xsf_node_coordinates(nodes_file, grain_count)

    clus_radius_sum = 0.0
    for row in grain_rows:
        grain_vol = row[2]
        hex_area = grain_vol / scaled_z
        hex_side = (2 * hex_area / (3 * 3**0.5)) ** 0.5
        hex_h = 1 / (3 * hex_side) * hex_area
        clus_radius_sum += hex_h - min_gb_dist
    cyl_radius = round(clus_radius_sum / grain_count, 3)

    lines = [
        f"variable num_grains equal {grain_count}",
        f"variable cyl_radius equal {cyl_radius}",
    ]
    for index, coords in enumerate(com_grains, start=1):
        x, y, _z = coords
        lines.extend(
            [
                f"variable g{index}_reg string g{index}",
                f"variable g{index}_x equal {x}",
                f"variable g{index}_y equal {y}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def validate_atomsk_structure(
    data_file: Annotated[
        str | os.PathLike[str],
        "LAMMPS data file generated by Atomsk.",
    ],
    expected_atom_types: Annotated[
        int | None,
        "Expected number of LAMMPS atom types, or None to skip this check.",
    ] = None,
    expected_grains: Annotated[
        int | None,
        "Expected Atomsk grain count, or None to skip grain-count checks.",
    ] = None,
    expected_box: Annotated[
        tuple[float, float, float] | None,
        "Expected box lengths in Angstrom along x, y, and z.",
    ] = None,
    box_tolerance: Annotated[
        float,
        "Absolute tolerance in Angstrom for comparing box lengths.",
    ] = 1.0e-6,
) -> AtomskStructureValidation:
    """
    Validate a generated LAMMPS data file and related Atomsk outputs.

    The validator performs lightweight structural checks without running MD:
    data-file existence, atom/type counts, box dimensions, Masses section, and
    optional Atomsk grain metadata consistency.
    """
    path = Path(data_file)
    issues: list[str] = []
    if not path.exists():
        return AtomskStructureValidation(path, 0, 0, {}, {}, None, (), (f"Missing data file: {path}",))

    parsed = _parse_lammps_data_header(path)
    atom_count = parsed["atom_count"]
    atom_type_count = parsed["atom_type_count"]
    box_bounds = parsed["box_bounds"]
    masses = parsed["masses"]

    if atom_count <= 0:
        issues.append("Atom count must be positive")
    if atom_type_count <= 0:
        issues.append("Atom type count must be positive")
    if expected_atom_types is not None and atom_type_count != expected_atom_types:
        issues.append(f"Expected {expected_atom_types} atom types, found {atom_type_count}")
    if not masses:
        issues.append("Masses section is missing or empty")

    if expected_box is not None:
        for axis, expected_length in zip(("x", "y", "z"), expected_box):
            bounds = box_bounds.get(axis)
            if bounds is None:
                issues.append(f"Missing {axis} box bounds")
                continue
            observed_length = bounds[1] - bounds[0]
            if abs(observed_length - expected_length) > box_tolerance:
                issues.append(
                    f"{axis} box length mismatch: expected {expected_length}, found {observed_length}"
                )

    auxiliary_files = tuple(
        candidate
        for candidate in (
            _atomsk_auxiliary_path(path, "nodes.xsf"),
            _atomsk_auxiliary_path(path, "id-size.txt"),
            _atomsk_auxiliary_path(path, "grains-com.xsf"),
            _atomsk_auxiliary_path(path, "size-dist.dat"),
        )
        if candidate.exists()
    )

    grain_count = None
    size_file = _atomsk_auxiliary_path(path, "id-size.txt")
    if size_file.exists():
        grain_count, _rows = _read_grain_size_rows(size_file)
    elif expected_grains is not None:
        issues.append(f"Missing Atomsk grain size file: {size_file}")

    if expected_grains is not None and grain_count is not None and grain_count != expected_grains:
        issues.append(f"Expected {expected_grains} grains, found {grain_count}")

    return AtomskStructureValidation(
        data_file=path,
        atom_count=atom_count,
        atom_type_count=atom_type_count,
        box_bounds=box_bounds,
        masses=masses,
        grain_count=grain_count,
        auxiliary_files=auxiliary_files,
        issues=tuple(issues),
    )


def make_kim_read_minimize_script(
    data_file: Annotated[
        str,
        "LAMMPS data filename or relative path to read.",
    ],
    kim_model: Annotated[
        str,
        "OpenKIM model ID to initialize in LAMMPS.",
    ],
    species: Annotated[
        str,
        "Species string passed to kim interactions.",
    ],
    mass: Annotated[
        float,
        "LAMMPS mass for atom type 1 in metal units.",
    ],
) -> str:
    """
    Create a short KIM-compatible LAMMPS smoke-test script for a data file.

    The returned script only verifies that LAMMPS can read and minimize the
    generated structure with a KIM model. It intentionally avoids ``pair_style``
    and ``pair_coeff`` commands.
    """
    return f"""units metal
boundary p p p
atom_style atomic

kim init {kim_model} metal
read_data {data_file}
kim interactions {species}
mass 1 {mass}

neighbor 0.2 bin
neigh_modify delay 5
thermo 10
thermo_style custom step pe temp lx ly lz vol
min_style cg
minimize 1.0e-8 1.0e-10 100 1000
"""


def _is_usable_atomsk(
    path: Annotated[
        Path,
        "Candidate Atomsk executable path to probe with --version.",
    ],
) -> bool:
    """Return whether a candidate path exists, is executable, and runs as Atomsk."""
    if not path.exists() or not os.access(path, os.X_OK):
        return False
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _format_atomsk_value(
    value: Annotated[
        str | float,
        "Atomsk parameter value that may already be an expression string.",
    ],
) -> str:
    """Format a numeric or preformatted Atomsk parameter value."""
    if isinstance(value, str):
        return value
    return _trim_float(value)


def _trim_float(
    value: Annotated[
        float,
        "Floating-point value to format compactly for Atomsk text files.",
    ],
) -> str:
    """Format a float without unnecessary trailing precision noise."""
    return f"{value:.12g}"


def _atomsk_auxiliary_path(
    data_file: Annotated[
        Path,
        "Atomsk-generated LAMMPS data file used to derive auxiliary filenames.",
    ],
    suffix: Annotated[
        str,
        "Atomsk auxiliary filename suffix such as nodes.xsf or id-size.txt.",
    ],
) -> Path:
    """Derive the path to an Atomsk auxiliary file next to a data file."""
    name = data_file.name
    if name.endswith(".lmp"):
        prefix = name[:-4]
    else:
        prefix = data_file.stem
    return data_file.parent / f"{prefix}_{suffix}"


def _parse_lammps_data_header(
    path: Annotated[
        Path,
        "LAMMPS data file path whose header and Masses section should be parsed.",
    ],
) -> dict:
    """Parse lightweight metadata from a LAMMPS data file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count = 0
    atom_type_count = 0
    box_bounds: dict[str, tuple[float, float]] = {}
    masses: dict[int, float] = {}

    in_masses = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if in_masses:
                continue
            continue
        if line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) >= 2 and parts[1] == "atoms":
            atom_count = int(parts[0])
        elif len(parts) >= 3 and parts[1] == "atom" and parts[2] == "types":
            atom_type_count = int(parts[0])
        elif len(parts) >= 4 and parts[2] in {"xlo", "ylo", "zlo"}:
            axis = parts[2][0]
            box_bounds[axis] = (float(parts[0]), float(parts[1]))
        elif line == "Masses":
            in_masses = True
        elif line.startswith("Atoms"):
            in_masses = False
        elif in_masses:
            mass_parts = line.split("#", 1)[0].split()
            if len(mass_parts) >= 2:
                masses[int(mass_parts[0])] = float(mass_parts[1])

    return {
        "atom_count": atom_count,
        "atom_type_count": atom_type_count,
        "box_bounds": box_bounds,
        "masses": masses,
    }


def _read_grain_size_rows(
    size_file: Annotated[
        Path,
        "Atomsk *_id-size.txt file containing grain atom counts and volumes.",
    ],
) -> tuple[int, list[tuple[int, int, float]]]:
    """Read Atomsk grain-size rows and return the final grain count plus rows."""
    rows: list[tuple[int, int, float]] = []
    for line in size_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 3:
            rows.append((int(parts[0]), int(parts[1]), float(parts[2])))
    if not rows:
        raise AtomskError(f"No grain rows found in {size_file}")
    return rows[-1][0], rows


def _read_xsf_node_coordinates(
    nodes_file: Annotated[
        Path,
        "Atomsk *_nodes.xsf file containing grain node coordinates.",
    ],
    count: Annotated[
        int,
        "Number of trailing node coordinate records expected in the XSF file.",
    ],
) -> list[tuple[float, float, float]]:
    """Read trailing node coordinates from an Atomsk XSF nodes file."""
    coords: list[tuple[float, float, float]] = []
    for line in nodes_file.read_text(encoding="utf-8").splitlines()[-count:]:
        parts = line.strip().split()
        if len(parts) >= 4:
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if len(coords) != count:
        raise AtomskError(f"Expected {count} node coordinates in {nodes_file}, found {len(coords)}")
    return coords
