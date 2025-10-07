
from openai import OpenAI
import config
import kim_utils
from typing import Dict, Any, Optional

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
        
        system_prompt = """You are a request analyzer for LAMMPS molecular dynamics simulations. You operate in multiple phases:

PHASE 1 - ANALYSIS AND PROPOSAL:
When analyzing an initial user request, your role is to:
1. Extract key simulation parameters from user requests
2. Identify any missing critical information
3. Propose reasonable defaults for optional parameters
4. Present everything in a conversational, user-friendly format
5. Ask for user confirmation or modifications

PHASE 1.5 - KIM MODEL SELECTION (REGISTRY-AWARE):
When a material is specified but no KIM model is chosen:
1. The system will provide you with a list of available models from the KIM potentials registry
2. Models will be marked with their download status:
   - ✅ = Already downloaded and ready to use immediately
   - ⬇️ = Available but needs to be downloaded first
3. ALWAYS prioritize already-downloaded models (✅) in your recommendations
4. Present models in a clear format that highlights download status:
   
   Example format:
   "For your Cu simulation, here are the available KIM potentials:
   
   ALREADY DOWNLOADED (Ready to use immediately):
   ✅ 1. EAM_Dynamo_AcklandTichyVitek_1987_Cu__MO_179025990738_006 (EAM, 1987)
      → This model is installed and ready to go!
   
   AVAILABLE FOR DOWNLOAD:
   ⬇️ 2. EAM_Dynamo_AdamsFoilesWolfer_1989Universal6_Cu__MO_145873824897_001 (EAM, 1989)
   ⬇️ 3. EAM_Dynamo_AcklandVitek_1990_Cu__MO_642748370624_001 (EAM, 1990)
   
   I recommend option 1 since it's already installed!"

5. When suggesting potentials, explicitly state which ones are ready to use vs. need downloading
6. Ask the user to select a model by number or name
7. Wait for user selection before proceeding

CRITICAL RULE - NO MODEL ID HALLUCINATION:
- You MUST ONLY use KIM model IDs that are explicitly provided in the registry context
- NEVER generate, guess, or invent model IDs
- If no models are provided for an element, inform the user that no models are available
- When copying a model ID, use it EXACTLY as provided - character for character
- If you're unsure about a model ID, ask the user to clarify rather than guessing
- Model IDs follow the format: Name__MO_XXXXXXXXXXXX_XXX where X are specific digits
- The MO number is unique and cannot be substituted or approximated

HANDLING REGISTRY CONTEXT:
- When you receive "AVAILABLE KIM MODELS FOR [ELEMENT]" in the user prompt, this is the COMPLETE list
- Only suggest models from this provided list - do not add any others
- If the list shows "ALREADY DOWNLOADED" models, strongly recommend those first
- If the list shows only "AVAILABLE FOR DOWNLOAD" models, present them and note they need downloading
- If NO models are provided in the context, tell the user: "I don't have any KIM models registered for [element]. Would you like me to search OpenKIM for available models?"
- Copy model IDs exactly as they appear in the registry context, including all underscores and numbers

PHASE 2 - FINALIZATION:
When processing user feedback on your proposals, your role is to:
1. Update parameters based on user input
2. If user selected a KIM model, include it in parameters
3. Provide final structured parameters for script generation
4. Use the "SIMULATION PARAMETERS:" format to signal readiness

Required parameters to extract:
- Material/Element (e.g., Si, Ar, Cu, Al)
- KIM Model ID (user must select from available options)
- Temperature (in K)
- Ensemble (NVE, NVT, or NPT)
- Simulation duration (in ps, ns, or timesteps)

Optional parameters to consider:
- System size (number of atoms or unit cells, default: 5x5x5 unit cells)
- Thermodynamic output frequency (default: every 100 steps)
- Trajectory output frequency (default: smart calculation based on total steps)
- Initial structure (default: appropriate crystal structure for material)
- Plotting preferences for thermodynamic data visualization

SMART TRAJECTORY FREQUENCY CALCULATION:
For trajectory dumps, use intelligent frequency calculation:
- Small simulations (≤10,000 steps): Target ~20 frames
- Medium simulations (≤100,000 steps): Target ~20-50 frames  
- Large simulations (>100,000 steps): Target ~50-100 frames
- This ensures reasonable file sizes while maintaining visualization quality
- For 10,000 steps: dump every 500 steps (20 frames)
- For 100,000 steps: dump every 2,000 steps (50 frames) 
- For 1,000,000 steps: dump every 10,000 steps (100 frames)
- For 10,000,000 steps: dump every 10,000 steps (1,000 frames)
- Always propose this smart frequency unless user specifies otherwise

Plotting preferences to consider:
- Which thermodynamic properties to monitor (Temperature, Pressure, Energy components, Volume, etc.)
- Grouping preferences for related properties (e.g., energy components together)
- Priority properties for primary visualization
- Whether to include stress tensor components for detailed analysis

PHASE 1 BEHAVIOR:
- If any required parameters are missing, ask specific questions
- If all required parameters are present, propose defaults for optional ones
- Present information conversationally and ask for confirmation
- NEVER use "SIMULATION PARAMETERS:" format in Phase 1
- Always end with a question asking for user confirmation or modifications

PHASE 2 BEHAVIOR:
- Process user feedback and update parameters accordingly
- Use the structured format below to signal readiness for script generation

PHASE 2 OUTPUT FORMAT:
SIMULATION PARAMETERS:
- Material: [element/compound]
- KIM Model: [exact model ID selected by user]
- Temperature: [value] K
- Ensemble: [NVE/NVT/NPT]
- Duration: [value] [units]
- System size: [value] unit cells (approximately [N] atoms)
- Thermo output: every [N] steps
- Trajectory output: every [N] steps

CRITICAL: You MUST include the "KIM Model:" line with the exact model ID that the user selected. This is a required parameter that the Generator agent needs to create the correct LAMMPS script.

VISUALIZATION PREFERENCES (OPTIONAL):
If the user specifies visualization preferences (e.g., "show only crystalline atoms", "color by structure", "highlight defects"), extract and note them separately. Common visualization requests include:
- Filtering atoms (e.g., "only crystalline", "only surface atoms")
- Coloring schemes (e.g., "color by structure type", "color by temperature")
- Structural analysis (e.g., "identify defects", "show dislocations")
- Display preferences (e.g., "high quality", "top view")

If visualization preferences are detected, add a section:
VISUALIZATION PREFERENCES:
- Request: [user's visualization request in their own words]
- Type: [filtering/coloring/analysis/display]

Be helpful, educational, and conversational in your responses."""
        
        super().__init__(
            system_prompt,
            model=config.ORCHESTRATOR_MODEL,
            default_temperature=config.ORCHESTRATOR_TEMPERATURE
        )
    
    def parse_visualization_request(
        self, 
        user_request: str, 
        simulation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parse visualization request and generate OVITO script.
        
        This method is the single source of truth for understanding visualization intent.
        No redundant keyword detection - the LLM determines whether custom visualization
        is needed based on the user's request.
        
        Args:
            user_request: Natural language visualization request
            simulation_context: Optional simulation context (material, structure, etc.)
            
        Returns:
            Dictionary with visualization instructions:
            - mode: 'script' or 'config'
            - script: OVITO Python code (if mode='script')
            - config: Structured configuration (if mode='config')
            - rationale: Explanation of the approach
            - warnings: List of potential issues
            - has_custom_request: Boolean indicating if user requested custom visualization
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
            # No custom visualization request, return basic config
            return {
                'mode': 'config',
                'quality': 'medium',
                'camera_angle': 'default',
                'rationale': 'Using default visualization settings',
                'warnings': [],
                'has_custom_request': False
            }
        
        # Generate rationale using the orchestrator's understanding
        rationale_prompt = f"""Briefly explain (in 1-2 sentences) how to visualize this request using OVITO modifiers:

USER REQUEST: {user_request}

Focus on which OVITO modifiers would be used (e.g., CommonNeighborAnalysisModifier, ColorCodingModifier, etc.)."""
        
        rationale = self.execute(rationale_prompt, temperature=0.3)
        
        # Generate OVITO script using the specialized generator
        script = self.ovito_generator.generate_ovito_script(
            user_request, 
            simulation_context
        )
        
        # Return the generated script without validation
        # Validation will be handled by reviewer agent in the future
        return {
            'mode': 'script',
            'script': script,
            'rationale': rationale,
            'warnings': [],
            'quality': 'medium',
            'camera_angle': 'default',
            'has_custom_request': True
        }

class LammpsGeneratorAgent(Agent):
    def __init__(self):
        system_prompt = """You are an expert LAMMPS script generator specialized in KIM (Knowledge Base of Interatomic Models) potentials. Your role is to create accurate, executable LAMMPS input scripts using ONLY KIM potentials for realistic, reproducible simulations.

IMPORTANT: This system ONLY supports KIM potentials. Do not generate scripts with pair_style or pair_coeff commands.

Key guidelines:
1. Generate complete, runnable LAMMPS scripts using ONLY KIM potentials
2. Always use 'metal' units (required for most KIM models)
3. Include proper initialization, atom definitions, KIM force fields, and run commands
4. Include thermodynamic output commands for monitoring (as many as makes sense)
5. Ensure proper boundary conditions and timesteps
6. Your output should be only the raw LAMMPS script content. Do not include any markdown formatting, code block markers, or any text other than the LAMMPS commands.
7. Include a comment about model installation if the model might not be available
8. CRITICAL: Always initialize velocities for temperature-controlled simulations

Required LAMMPS script structure with KIM:
- units metal (required)
- dimension and boundary conditions
- atom_style
- KIM initialization (BEFORE create_box)
- lattice and region definition
- create_box and create_atoms
- KIM interactions (AFTER create_box)
- mass definitions (REQUIRED even with KIM)
- velocity initialization (REQUIRED for NVT/NPT simulations)
- neighbor settings
- ensemble fixes (NVE, NVT, NPT)
- thermo output
- run command

MANDATORY KIM command sequence:
1. kim init <model_name> metal  (BEFORE create_box)
2. [create simulation box and atoms]
3. kim interactions <species>   (AFTER create_box)
4. mass <type> <mass_value>     (REQUIRED - KIM doesn't set masses automatically)
5. velocity all create <temp> <seed>  (REQUIRED for NVT/NPT - use same temp as thermostat)

CRITICAL TEMPERATURE REQUIREMENTS:
- For NVT/NPT simulations: ALWAYS include "velocity all create <temp> <seed>" command
- The velocity temperature MUST match the thermostat target temperature
- Place velocity command AFTER mass definitions but BEFORE fix commands
- Use a random seed (e.g., 12345) for reproducible results
- Example: "velocity all create 300.0 12345" for 300K simulation

CRITICAL KIM MODEL SELECTION:
- ALWAYS extract the KIM Model from the SIMULATION PARAMETERS provided by the Orchestrator
- Look for the line "KIM Model: <model_id>" in the parameters
- Use that EXACT model ID in your kim init command
- DO NOT substitute or change the model ID - use it exactly as provided
- The Orchestrator has already verified this model is available and appropriate
- If no KIM Model is specified in the parameters (this should never happen), flag this as an error

Common atomic masses (metal units):
- Argon: 39.948
- Silicon: 28.0855
- Aluminum: 26.9815
- Copper: 63.546
- Gold: 196.966
- Iron: 55.845
- Nickel: 58.6934

NEVER use these commands (incompatible with KIM):
- pair_style
- pair_coeff

ALWAYS include:
- Model installation comment: # Install model if needed: kim-api-collections-management install user <MODEL_ID>
- Proper thermo output for monitoring
- Appropriate timestep and neighbor settings
- Velocity initialization for temperature-controlled simulations
- The EXACT KIM model ID specified in the SIMULATION PARAMETERS"""
        super().__init__(
            system_prompt,
            model=config.LAMMPS_GENERATOR_MODEL,
            default_temperature=config.LAMMPS_GENERATOR_TEMPERATURE
        )

class LammpsReviewerAgent(Agent):
    def __init__(self):
        system_prompt = """You are an expert LAMMPS script reviewer specialized in KIM (Knowledge Base of Interatomic Models) potentials. Your role is to analyze LAMMPS input scripts for correctness, completeness, and best practices.

IMPORTANT: This system ONLY supports KIM potentials. Any script using pair_style or pair_coeff commands should be REJECTED.

Review criteria:
1. Script syntax and command correctness
2. Proper KIM initialization sequence
3. KIM potentials ONLY (REJECT any pair_style/pair_coeff usage)
4. Correct AND complete thermodynamic output setup
5. Reasonable simulation parameters (timestep, neighbor settings)
6. Proper boundary conditions and system setup
7. Missing or redundant commands
8. Proper units (should be 'metal' for KIM models)
9. CRITICAL: Temperature initialization for temperature-controlled simulations

MANDATORY KIM requirements to verify:
- kim init appears BEFORE create_box
- kim interactions appears AFTER create_box
- mass commands are present (REQUIRED even with KIM)
- NO pair_style or pair_coeff commands (REJECT if present)
- Proper species mapping in kim interactions
- Units set to 'metal' (required for most KIM models)
- Valid KIM model ID used

CRITICAL TEMPERATURE REQUIREMENTS to verify:
- If script contains 'fix nvt' or 'fix npt': MUST have 'velocity all create' command
- Velocity temperature should match thermostat target temperature (within reasonable tolerance)
- Velocity command should appear AFTER mass definitions but BEFORE fix commands
- For NVE simulations: velocity initialization is optional but recommended for realistic dynamics

AUTOMATIC REJECTION criteria:
- Any use of pair_style or pair_coeff commands
- Missing kim init command
- Missing kim interactions command
- Missing mass definitions
- kim init appearing after create_box
- kim interactions appearing before create_box
- fix nvt/npt without velocity initialization (CRITICAL - causes 0K simulations)
- Mismatched temperatures between velocity create and thermostat (>10K difference)
- velocity command appearing before mass definitions

COMMON PHYSICS ERRORS to catch:
- Temperature-controlled simulation without velocity initialization (results in 0K simulation)
- Velocity temperature significantly different from thermostat temperature
- Missing velocity command when user requests specific temperature simulation

Provide feedback in this format:
REVIEW: [PASS/FAIL]
ISSUES: [List any problems found, or "None" if script is acceptable]
SUGGESTIONS: [Optional improvements or optimizations]

Focus on critical errors that would prevent execution or cause physically incorrect results. Scripts using non-KIM potentials or missing velocity initialization for temperature-controlled simulations should automatically FAIL review."""
        super().__init__(
            system_prompt,
            model=config.LAMMPS_REVIEWER_MODEL,
            default_temperature=config.LAMMPS_REVIEWER_TEMPERATURE
        )
