from openai import OpenAI
import config
import kim_utils
from typing import Dict, Any, Optional
from prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    LAMMPS_GENERATOR_SYSTEM_PROMPT,
    LAMMPS_REVIEWER_SYSTEM_PROMPT,
)

class Agent:
    def __init__(self, system_prompt, model=None, default_temperature=0.1):
        self.system_prompt = system_prompt
        self.model = model or config.DEFAULT_MODEL
        self.default_temperature = default_temperature
        self.client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)

    def execute(self, user_prompt, temperature=None):
        # Use provided temperature, or agent's default, or fallback to 0.1
        temp = temperature if temperature is not None else self.default_temperature
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
        )
        return completion.choices[0].message.content

    def execute_stream(self, user_prompt, temperature=None):
        # Use provided temperature, or agent's default, or fallback to 0.1
        temp = temperature if temperature is not None else self.default_temperature
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Try streaming first
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                stream=True,
            )
            
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            # Fallback to non-streaming if streaming fails
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
            )
            
            # Simulate streaming by yielding the content in chunks
            content = completion.choices[0].message.content
            chunk_size = 50  # Characters per chunk
            for i in range(0, len(content), chunk_size):
                yield content[i:i + chunk_size]

class OrchestratorAgent(Agent):
    def __init__(self):
        # Import here to avoid circular dependency
        from ovito_script_generator import OVITOScriptGenerator
        from ovito_executor import DynamicOVITOExecutor

        self.ovito_generator = OVITOScriptGenerator()
        self.ovito_executor = DynamicOVITOExecutor()

        super().__init__(
            ORCHESTRATOR_SYSTEM_PROMPT,
            model=config.ORCHESTRATOR_MODEL,
            default_temperature=config.ORCHESTRATOR_TEMPERATURE
        )
    
    def parse_visualization_request(
        self, 
        user_request: str, 
        simulation_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate OVITO script if custom visualization is requested.
        
        This method is the single source of truth for understanding visualization intent.
        The LLM determines whether custom visualization is needed and generates the script.
        
        Args:
            user_request: Natural language visualization request
            simulation_context: Optional simulation context (material, structure, etc.)
            
        Returns:
            OVITO Python script as a string, or None for default visualization
        """
        # Use LLM to determine if this is a custom visualization request
        detection_prompt = f"""Analyze this user request and determine if it contains a CUSTOM visualization request.

USER REQUEST: {user_request}

A CUSTOM visualization request is one that asks for specific visual analysis like:
- "show only crystalline atoms"
- "color by structure type"
- "highlight defects"
- "visualize surface atoms"
- "filter out amorphous regions"

A request is NOT custom if it's just a general simulation request without specific visualization instructions.

Respond with ONLY "YES" if this is a custom visualization request, or "NO" if it's not."""
        
        detection_response = self.execute(detection_prompt, temperature=0.1).strip().upper()
        has_custom_request = "YES" in detection_response
        
        if not has_custom_request:
            # No custom visualization - use default
            print("[VIZ AGENT] No custom visualization requested, using default")
            return None
        
        # Prepare simulation context
        if simulation_context is None:
            simulation_context = {}
        
        # Check if CNA is needed and mark for pre-computation in LAMMPS
        cna_keywords = ['structure', 'fcc', 'bcc', 'hcp', 'crystalline', 'crystal', 'amorphous']
        if any(keyword in user_request.lower() for keyword in cna_keywords):
            simulation_context['cna_precomputed'] = True
            print("[VIZ AGENT] CNA-based visualization detected, will use pre-computed values")
        
        # Generate OVITO script using the specialized generator
        script = self.ovito_generator.generate_ovito_script(
            user_request, 
            simulation_context
        )
        
        print(f"[VIZ AGENT] Generated custom OVITO script ({len(script)} chars)")
        
        return script

class LammpsGeneratorAgent(Agent):
    def __init__(self):
        super().__init__(
            LAMMPS_GENERATOR_SYSTEM_PROMPT,
            model=config.LAMMPS_GENERATOR_MODEL,
            default_temperature=config.LAMMPS_GENERATOR_TEMPERATURE
        )

class LammpsReviewerAgent(Agent):
    def __init__(self):
        super().__init__(
            LAMMPS_REVIEWER_SYSTEM_PROMPT,
            model=config.LAMMPS_REVIEWER_MODEL,
            default_temperature=config.LAMMPS_REVIEWER_TEMPERATURE
        )
