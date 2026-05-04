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
from typing import Iterable, Optional, Sequence


class AtomskError(RuntimeError):
    """Raised when Atomsk is unavailable or an Atomsk workflow fails."""


@dataclass(frozen=True)
class AtomskCommandResult:
    """Captured result from an Atomsk subprocess invocation."""

    command: tuple[str, ...]
    cwd: Path
    stdout: str
    stderr: str


@dataclass(frozen=True)
class GrainNode:
    """One Atomsk polycrystal grain node."""

    x: str | float
    y: str | float
    z: str | float
    orientation: str

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

    data_file: Path
    atom_count: int
    atom_type_count: int
    box_bounds: dict[str, tuple[float, float]]
    masses: dict[int, float]
    grain_count: Optional[int]
    auxiliary_files: tuple[Path, ...]
    issues: tuple[str, ...]

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
    args: Sequence[str | os.PathLike[str]],
    cwd: str | os.PathLike[str],
    atomsk_executable: str | os.PathLike[str] | None = None,
) -> AtomskCommandResult:
    """Run Atomsk with a safe argument list and captured output."""
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
    output_dir: str | os.PathLike[str],
    lattice: str,
    lattice_parameter: float,
    species: str | Sequence[str],
    output_filename: str | None = None,
    orientation: Sequence[str] | None = None,
    atomsk_executable: str | os.PathLike[str] | None = None,
) -> Path:
    """Create an Atomsk seed unit cell, typically an XSF file."""
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
    path: str | os.PathLike[str],
    box: tuple[float, float, float],
    nodes: Sequence[GrainNode],
) -> Path:
    """Write an Atomsk polycrystal parameter file with explicit grain nodes."""
    if not nodes:
        raise ValueError("At least one grain node is required")

    param_path = Path(path)
    param_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"box {box[0]} {box[1]} {box[2]}"]
    lines.extend(node.to_atomsk_line() for node in nodes)
    param_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return param_path


def create_polycrystal_from_nodes(
    output_dir: str | os.PathLike[str],
    seed_file: str | os.PathLike[str],
    box: tuple[float, float, float],
    nodes: Sequence[GrainNode],
    output_prefix: str = "data.hexslab",
    duplicate: tuple[int, int, int] = (1, 1, 1),
    param_filename: str = "atomsk_box_commands.txt",
    atomsk_executable: str | os.PathLike[str] | None = None,
) -> Path:
    """Create a LAMMPS data polycrystal using explicit Atomsk nodes."""
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
    output_parent: str | os.PathLike[str],
    element: str = "Cu",
    lattice: str = "fcc",
    lattice_parameter: float = 3.615,
    hex_size: int = 400,
    dupe_z: int = 5,
    hex_grid: int = 2,
    orientations: Sequence[str] = DEFAULT_HEX_GRAIN_ORIENTATIONS,
    z_positions: Sequence[float] | None = None,
    output_dir_name: str | None = None,
    atomsk_executable: str | os.PathLike[str] | None = None,
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
    hex_size: int,
    hex_grid: int,
    orientations: Sequence[str],
    z_positions: Sequence[float] | None = None,
) -> tuple[GrainNode, ...]:
    """Build the 2x2 hex-grain node layout used by the reference script."""
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


def write_atomsk_command_log(path: str | os.PathLike[str], commands: Iterable[str]) -> Path:
    """Write reproducible Atomsk command provenance."""
    command_path = Path(path)
    command_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    return command_path


def write_grain_lammps_vars(
    output_prefix: str | os.PathLike[str],
    dupe_z: int,
    min_gb_dist: float = 15.0,
    output_filename: str = "in.vars_grains",
) -> Path:
    """Write LAMMPS grain variable file from Atomsk polycrystal metadata."""
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
    data_file: str | os.PathLike[str],
    expected_atom_types: int | None = None,
    expected_grains: int | None = None,
    expected_box: tuple[float, float, float] | None = None,
    box_tolerance: float = 1.0e-6,
) -> AtomskStructureValidation:
    """Validate a generated LAMMPS data file and related Atomsk outputs."""
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
    data_file: str,
    kim_model: str,
    species: str,
    mass: float,
) -> str:
    """Create a short KIM-compatible LAMMPS smoke-test script for a data file."""
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


def _is_usable_atomsk(path: Path) -> bool:
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


def _format_atomsk_value(value: str | float) -> str:
    if isinstance(value, str):
        return value
    return _trim_float(value)


def _trim_float(value: float) -> str:
    return f"{value:.12g}"


def _atomsk_auxiliary_path(data_file: Path, suffix: str) -> Path:
    name = data_file.name
    if name.endswith(".lmp"):
        prefix = name[:-4]
    else:
        prefix = data_file.stem
    return data_file.parent / f"{prefix}_{suffix}"


def _parse_lammps_data_header(path: Path) -> dict:
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


def _read_grain_size_rows(size_file: Path) -> tuple[int, list[tuple[int, int, float]]]:
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


def _read_xsf_node_coordinates(nodes_file: Path, count: int) -> list[tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    for line in nodes_file.read_text(encoding="utf-8").splitlines()[-count:]:
        parts = line.strip().split()
        if len(parts) >= 4:
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if len(coords) != count:
        raise AtomskError(f"Expected {count} node coordinates in {nodes_file}, found {len(coords)}")
    return coords
