
import pytest
from agents import IterativeLammpsGeneration, configure_dspy_lm

def test_mvp1_end_to_end():
    """
    Tests the full MVP 1 workflow: Generator -> Reviewer via DSPy module.
    Makes a real (but small) LLM call.
    """
    # Configure DSPy LM
    configure_dspy_lm()

    # Instantiate the iterative generation module
    generator = IterativeLammpsGeneration(max_attempts=1, use_memory=False)

    # Run generation + review
    result = generator.forward(
        material="Si",
        kim_model="SW_StillingerWeber_1985_Si__MO_405512056662_005",
        temperature=300.0,
        ensemble="NVT",
        duration_ps=1.0,
        system_size=3,
    )

    # Assert that the generated script contains the crucial 'kim' command
    assert "kim" in result.script.lower(), "The generated script should use the 'kim' command."

    # Assert that review passed
    assert result.review_passed, f"Reviewer failed the script. Issues: {result.issues}"
