"""
OVITO Script Generator - LLM-powered dynamic OVITO Python script generation.

This module provides an agent that generates complete OVITO Python scripts
from natural language visualization requests, similar to how LAMMPS scripts
are generated dynamically.
"""

from agents import Agent
import config
from prompts import OVITO_SCRIPT_GENERATOR_SYSTEM_PROMPT


class OVITOScriptGenerator(Agent):
    """
    LLM-powered generator for OVITO Python scripts.
    
    This agent generates complete, executable OVITO Python code based on 
    natural language visualization requests. It has deep knowledge of OVITO's 
    API, modifiers, and best practices for molecular dynamics visualization.
    """
    
    def __init__(self):
        super().__init__(
            OVITO_SCRIPT_GENERATOR_SYSTEM_PROMPT,
            model=config.OVITO_SCRIPT_MODEL,
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
