"""
Tests for standalone Atomsk structure-generation helpers.
"""
from pathlib import Path

import pytest

from atomsk_tools import (
    AtomskError,
    DEFAULT_HEX_GRAIN_ORIENTATIONS,
    GrainNode,
    create_hex_grains,
    create_polycrystal_from_nodes,
    create_unit_cell,
    find_atomsk_executable,
    make_hex_grain_nodes,
    make_kim_read_minimize_script,
    validate_atomsk_structure,
    write_grain_lammps_vars,
    write_polycrystal_node_file,
)


def test_grain_node_line_format():
    node = GrainNode("0.25*box", "0.75*box", "0.5*box", "[100] [010] [001]")

    assert node.to_atomsk_line() == "node 0.25*box 0.75*box 0.5*box [100] [010] [001]"


def test_make_hex_grain_nodes_matches_reference_layout():
    nodes = make_hex_grain_nodes(
        hex_size=400,
        hex_grid=2,
        orientations=DEFAULT_HEX_GRAIN_ORIENTATIONS,
        z_positions=(0.1, 0.2, 0.3, 0.4),
    )

    assert len(nodes) == 4
    assert nodes[0].to_atomsk_line() == (
        "node 0.5*box 0.5*box 0.4*box [-1_-1_1] [1_1_2] [1_-1_0]"
    )
    assert nodes[-1].to_atomsk_line() == (
        "node 0*box 0.25*box 0.1*box [2_2_-5] [5_5_4] [1_-1_0]"
    )


def test_write_polycrystal_node_file(tmp_path):
    node_file = write_polycrystal_node_file(
        tmp_path / "atomsk_box_commands.txt",
        box=(20.0, 20.0, 3.615),
        nodes=(GrainNode("0.5*box", "0.25*box", "0.5*box", "[100] [010] [001]"),),
    )

    assert node_file.read_text(encoding="utf-8") == (
        "box 20.0 20.0 3.615\n"
        "node 0.5*box 0.25*box 0.5*box [100] [010] [001]\n"
    )


def test_validate_lammps_data_fixture(tmp_path):
    data_file = tmp_path / "tiny.lmp"
    data_file.write_text(
        """ # Voronoi polycrystal with 2 grains

         4  atoms
           1  atom types

      0.000000000000      20.000000000000  xlo xhi
      0.000000000000      20.000000000000  ylo yhi
      0.000000000000       7.230000000000  zlo zhi

Masses

            1   63.54600000             # Cu

Atoms # atomic

         1    1        0.0      0.0       0.0
         2    1        1.0      1.0       1.0
         3    1        2.0      2.0       2.0
         4    1        3.0      3.0       3.0
""",
        encoding="utf-8",
    )
    (tmp_path / "tiny_id-size.txt").write_text(
        "# grainID atoms volume\n1 2 10.0\n2 2 10.0\n",
        encoding="utf-8",
    )

    result = validate_atomsk_structure(
        data_file,
        expected_atom_types=1,
        expected_grains=2,
        expected_box=(20.0, 20.0, 7.23),
    )

    assert result.is_valid
    assert result.atom_count == 4
    assert result.atom_type_count == 1
    assert result.grain_count == 2
    assert result.masses == {1: 63.546}


def test_validate_lammps_data_reports_mismatches(tmp_path):
    data_file = tmp_path / "bad.lmp"
    data_file.write_text(
        """1 atoms
1 atom types
0 1 xlo xhi
0 1 ylo yhi
0 1 zlo zhi
Atoms # atomic
1 1 0 0 0
""",
        encoding="utf-8",
    )

    result = validate_atomsk_structure(data_file, expected_atom_types=2, expected_grains=1)

    assert not result.is_valid
    assert "Expected 2 atom types, found 1" in result.issues
    assert "Masses section is missing or empty" in result.issues
    assert any(issue.startswith("Missing Atomsk grain size file") for issue in result.issues)


def test_make_kim_read_minimize_script_uses_read_data_and_kim():
    script = make_kim_read_minimize_script(
        data_file="data.hexslab.lmp",
        kim_model="EAM_Dynamo_Test_Cu__MO_000000000000_000",
        species="Cu",
        mass=63.546,
    )

    assert "kim init EAM_Dynamo_Test_Cu__MO_000000000000_000 metal" in script
    assert "read_data data.hexslab.lmp" in script
    assert "kim interactions Cu" in script
    assert "pair_style" not in script
    assert "pair_coeff" not in script


def test_find_atomsk_executable_requires_env_var(monkeypatch):
    monkeypatch.delenv("ATOMSK_EXECUTABLE", raising=False)

    with pytest.raises(AtomskError, match="ATOMSK_EXECUTABLE is not set"):
        find_atomsk_executable()


def test_find_atomsk_executable_rejects_invalid_env_path(monkeypatch, tmp_path):
    bad_path = tmp_path / "missing-atomsk"
    monkeypatch.setenv("ATOMSK_EXECUTABLE", str(bad_path))

    with pytest.raises(AtomskError, match="not a runnable Atomsk executable"):
        find_atomsk_executable()


def _atomsk_or_skip() -> Path:
    try:
        return find_atomsk_executable()
    except AtomskError as exc:
        pytest.skip(str(exc))


@pytest.mark.integration
def test_atomsk_create_unit_cell_smoke(tmp_path):
    atomsk = _atomsk_or_skip()

    seed_file = create_unit_cell(
        tmp_path,
        lattice="fcc",
        lattice_parameter=3.615,
        species="Cu",
        output_filename="Cu.xsf",
        atomsk_executable=atomsk,
    )

    assert seed_file.exists()
    assert "Cu" in seed_file.read_text(encoding="utf-8", errors="ignore")


@pytest.mark.integration
def test_atomsk_polycrystal_smoke(tmp_path):
    atomsk = _atomsk_or_skip()
    seed_file = create_unit_cell(
        tmp_path,
        lattice="fcc",
        lattice_parameter=3.615,
        species="Cu",
        output_filename="Cu.xsf",
        atomsk_executable=atomsk,
    )
    data_file = create_polycrystal_from_nodes(
        output_dir=tmp_path,
        seed_file=seed_file,
        box=(20.0, 20.0, 3.615),
        nodes=(
            GrainNode("0.25*box", "0.25*box", "0.5*box", "[100] [010] [001]"),
            GrainNode("0.75*box", "0.75*box", "0.5*box", "[110] [1_-1_0] [001]"),
        ),
        output_prefix="data.hex",
        duplicate=(1, 1, 2),
        atomsk_executable=atomsk,
    )

    result = validate_atomsk_structure(
        data_file,
        expected_atom_types=1,
        expected_grains=2,
        expected_box=(20.0, 20.0, 7.23),
    )

    assert result.is_valid, result.issues
    assert result.atom_count > 0
    assert result.grain_count == 2


@pytest.mark.integration
def test_create_hex_grains_smoke(tmp_path):
    atomsk = _atomsk_or_skip()

    data_file = create_hex_grains(
        tmp_path,
        hex_size=20,
        dupe_z=2,
        z_positions=(0.1, 0.2, 0.3, 0.4),
        atomsk_executable=atomsk,
    )

    assert data_file.name == "data.hexslab.lmp"
    assert data_file.parent.name == "gs20_nhex4_zrep2"
    assert (data_file.parent / "atomsk_box_commands.txt").exists()
    assert (data_file.parent / "atomsk_lattice_commands.txt").exists()
    assert (data_file.parent / "in.vars_grains").exists()

    result = validate_atomsk_structure(
        data_file,
        expected_atom_types=1,
        expected_grains=4,
        expected_box=(40.0, 40.0, 3.615 * 2**0.5 * 2),
        box_tolerance=1.0e-9,
    )
    assert result.is_valid, result.issues


@pytest.mark.integration
def test_write_grain_lammps_vars_from_atomsk_outputs(tmp_path):
    atomsk = _atomsk_or_skip()
    data_file = create_hex_grains(
        tmp_path,
        hex_size=20,
        dupe_z=2,
        z_positions=(0.1, 0.2, 0.3, 0.4),
        atomsk_executable=atomsk,
    )

    vars_file = write_grain_lammps_vars(data_file.with_suffix(""), dupe_z=2)
    vars_text = vars_file.read_text(encoding="utf-8")

    assert "variable num_grains equal 4" in vars_text
    assert "variable cyl_radius equal" in vars_text
    assert "variable g1_x equal" in vars_text
