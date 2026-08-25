"""The campaign template contract — parsing and filling the variable block.

A campaign template is an ordinary, runnable LAMMPS script whose first
executable content is a sentinel-delimited block of scalar declarations:

    # ==== CAMPAIGN VARIABLES (auto-filled per iteration — do not move) ====
    variable a0                equal 4.05
    variable temperature       equal 300
    variable strain_rate       equal 0.005
    # ==== END CAMPAIGN VARIABLES ====

Everything below references ${a0}, ${temperature}, ... Filling an iteration
rewrites ONLY the numeric right-hand sides inside the sentinels, so any two
trials of a campaign differ by exactly those lines and nothing else. That is
what makes cross-trial metric comparisons fair by construction, rather than by
asking an LLM to regenerate a stable script each time.

This module is deliberately dependency-free (stdlib only): it holds the logic
that decides whether a script can be swept, so it stays importable and testable
without pulling in langchain, langgraph, or the rest of the tool stack.
"""
import re
from typing import Dict, Tuple


VAR_BLOCK_BEGIN = "# ==== CAMPAIGN VARIABLES"
VAR_BLOCK_END = "# ==== END CAMPAIGN VARIABLES"

# `variable <name> equal <number>` split into prefix / value / trailing comment
# so a rewrite can preserve the author's column alignment and comments.
_VAR_LINE_RE = re.compile(
    r"^(?P<prefix>\s*variable\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+equal\s+)"
    r"(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?)"
    r"(?P<suffix>\s*(?:#.*)?)$"
)


def format_number(v: float) -> str:
    """Render a knob value the way a human would write it in a LAMMPS script.

    Integral floats lose the trailing '.0' (300.0 -> '300') because that is how
    temperatures and cell counts are conventionally written; everything else
    uses repr, which round-trips exactly and keeps 0.005 as '0.005' rather than
    '5e-03'.
    """
    f = float(v)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return repr(f)


def parse_variable_block(script: str) -> Tuple[Dict[str, float], Tuple[int, int]]:
    """Locate and read the campaign variable block.

    Returns (values, (begin_line_idx, end_line_idx)) with 0-based indices of the
    two sentinel lines themselves.

    Raises ValueError with an actionable message when the block is missing,
    unterminated, empty, or contains a non-scalar declaration. Callers turn that
    message into a {"error": ...} the orchestrator can act on — computed
    quantities (`variable ymid equal ly/2`) are legal LAMMPS but belong BELOW
    the block, since only literal scalars can be swept.
    """
    lines = script.split("\n")

    begin = end = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if begin == -1 and stripped.startswith(VAR_BLOCK_BEGIN):
            begin = i
        elif begin != -1 and stripped.startswith(VAR_BLOCK_END):
            end = i
            break

    if begin == -1:
        raise ValueError(
            f"no campaign variable block found — the script must open with a "
            f"'{VAR_BLOCK_BEGIN} ... ====' marker line"
        )
    if end == -1:
        raise ValueError(
            f"campaign variable block opened at line {begin + 1} but never "
            f"closed — add a '{VAR_BLOCK_END} ====' marker line"
        )

    values: Dict[str, float] = {}
    for i in range(begin + 1, end):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _VAR_LINE_RE.match(lines[i])
        if not m:
            raise ValueError(
                f"line {i + 1} inside the campaign variable block is not a "
                f"literal scalar declaration: {stripped!r}. Only "
                f"'variable <name> equal <number>' lines belong in the block; "
                f"move computed variables below the END marker."
            )
        name = m.group("name")
        if name in values:
            raise ValueError(
                f"variable {name!r} declared twice inside the campaign "
                f"variable block (line {i + 1})"
            )
        values[name] = float(m.group("value").replace("d", "e").replace("D", "e"))

    if not values:
        raise ValueError("campaign variable block is empty — it must declare at "
                         "least the swept knobs")

    return values, (begin, end)


def extract_variable_block(script: str) -> str:
    """Return the campaign variable block, sentinels included, as text.

    Used to show the user what an iteration actually substituted: the block is
    the only part of the script that differs between trials, so it is the whole
    diff in about ten lines.
    """
    _, (begin, end) = parse_variable_block(script)
    return "\n".join(script.split("\n")[begin:end + 1])


def fill_variable_block(script: str, values: Dict[str, float]) -> str:
    """Return `script` with the given variables reassigned inside the block.

    Only the numeric right-hand sides of matching lines change; prefixes,
    trailing comments, and every byte outside the sentinels are preserved. When
    a line carries a trailing comment the new value is padded to the original
    value's width so comment columns stay aligned.

    Raises ValueError if a requested name is not declared in the block.
    """
    current, (begin, end) = parse_variable_block(script)

    unknown = [k for k in values if k not in current]
    if unknown:
        raise ValueError(
            f"variable(s) {unknown} are not declared in the campaign variable "
            f"block. Declared: {sorted(current)}"
        )

    lines = script.split("\n")
    for i in range(begin + 1, end):
        m = _VAR_LINE_RE.match(lines[i])
        if not m or m.group("name") not in values:
            continue
        new_value = format_number(values[m.group("name")])
        suffix = m.group("suffix")
        if suffix.strip():
            # Keep trailing comments in their original column where possible.
            new_value = new_value.ljust(len(m.group("value")))
        lines[i] = f"{m.group('prefix')}{new_value}{suffix}"

    return "\n".join(lines)


_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve_placeholders(
    script: str, variables: Dict[str, float], swept=()
) -> Tuple[str, list]:
    """Replace ${name} references with literal numbers.

    Used for the OVITO script, which is plain Python and is stored once and
    reused verbatim — nothing ever fills it the way fill_variable_block fills
    the LAMMPS template, so a surviving ${...} is a syntax error that silently
    costs you the whole custom visualization.

    Returns (resolved_script, problems). A swept knob is NOT substituted: its
    value changes per iteration, so freezing iteration 1's number into a
    reused script would be quietly wrong. Unknown names are left alone too.
    Both cases are reported as problems for the caller to reject on.
    """
    swept = set(swept)
    unknown, swept_used = [], []

    def _sub(m):
        name = m.group(1)
        if name in swept:
            swept_used.append(name)
            return m.group(0)
        if name not in variables:
            unknown.append(name)
            return m.group(0)
        return format_number(variables[name])

    resolved = _PLACEHOLDER_RE.sub(_sub, script)

    problems = []
    if swept_used:
        problems.append(
            f"references swept knob(s) {sorted(set(swept_used))}, whose values "
            f"change every iteration — the visualization must not depend on them"
        )
    if unknown:
        problems.append(
            f"references undeclared variable(s) {sorted(set(unknown))}"
        )
    return resolved, problems


def check_knob_bounds(plan: dict, knobs: Dict[str, float]) -> list:
    """Return a list of human-readable bound violations (empty when all ok).

    Values are rejected rather than clamped: a knob outside the planned range
    means the proposal was wrong, and silently running a different simulation
    than the one reasoned about would poison the trial history.
    """
    problems = []
    bounds = (plan or {}).get("knobs") or {}
    for name, value in knobs.items():
        rng = bounds.get(name)
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            continue
        low, high = float(rng[0]), float(rng[1])
        try:
            v = float(value)
        except (TypeError, ValueError):
            problems.append(f"{name}={value!r} is not a number")
            continue
        if not (low <= v <= high):
            problems.append(f"{name}={v} is outside the planned range [{low}, {high}]")
    return problems
