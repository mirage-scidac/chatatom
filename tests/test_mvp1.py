
import pytest
from agents import LammpsGeneratorAgent, LammpsReviewerAgent

def test_mvp1_end_to_end():
    """
    Tests the full MVP 1 workflow: Generator -> Reviewer
    Makes a real (but small) LLM call.
    """
    # The user prompt for the test
    test_prompt = "Create a LAMMPS script for a simple simulation of silicon using a KIM potential."

    # 1. Instantiate the agents
    generator = LammpsGeneratorAgent()
    reviewer = LammpsReviewerAgent()

    # 2. Generate the script
    # We use a low temperature for more predictable output
    generated_script = generator.execute(test_prompt, temperature=0.0)

    # 3. Review the script
    review_result = reviewer.execute(generated_script, temperature=0.0)

    # 4. Assertions
    # Assert that the generated script contains the crucial 'kim' command
    assert "kim" in generated_script.lower(), "The generated script should use the 'kim' command."

    # Assert that the reviewer passed the script
    assert "REVIEW: PASS" in review_result, f"Reviewer failed the script with comments: {review_result}"
