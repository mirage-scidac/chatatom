
from openai import OpenAI
import config

class Agent:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
        self.client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)

    def execute(self, user_prompt, temperature=0.1):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        completion = self.client.chat.completions.create(
            model="gpt4o",  # Or whatever model is appropriate
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content

class OrchestratorAgent(Agent):
    def __init__(self):
        super().__init__(
            "You are an orchestrator agent. Your job is to coordinate other agents to fulfill the user's request. "
            "You will receive a request from the user, then decide which agent to call first. "
            "For now, you will always follow this sequence: LammpsGeneratorAgent -> LammpsReviewerAgent."
        )

class LammpsGeneratorAgent(Agent):
    def __init__(self):
        self.system_prompt = """You are an expert LAMMPS script generator specialized in KIM (Knowledge Base of Interatomic Models) potentials. Your role is to create accurate, executable LAMMPS input scripts using ONLY KIM potentials for realistic, reproducible simulations.

IMPORTANT: This system ONLY supports KIM potentials. Do not generate scripts with pair_style or pair_coeff commands.

Key guidelines:
1. Generate complete, runnable LAMMPS scripts using ONLY KIM potentials
2. Always use 'metal' units (required for most KIM models)
3. Include proper initialization, atom definitions, KIM force fields, and run commands
4. Include thermodynamic output commands for monitoring
5. Ensure proper boundary conditions and timesteps
6. Your output should be only the raw LAMMPS script content. Do not include any markdown formatting, code block markers, or any text other than the LAMMPS commands.
7. Include a comment about model installation if the model might not be available

Required LAMMPS script structure with KIM:
- units metal (required)
- dimension and boundary conditions
- atom_style
- KIM initialization (BEFORE create_box)
- lattice and region definition
- create_box and create_atoms
- KIM interactions (AFTER create_box)
- mass definitions (REQUIRED even with KIM)
- neighbor settings
- thermo output
- run command

MANDATORY KIM command sequence:
1. kim init <model_name> metal  (BEFORE create_box)
2. [create simulation box and atoms]
3. kim interactions <species>   (AFTER create_box)
4. mass <type> <mass_value>     (REQUIRED - KIM doesn't set masses automatically)

Verified KIM model IDs (use these):
- LJ_Shifted_Bernardes_1958MedCutoff_Ar__MO_126566794224_004 (Argon LJ)
- SW_StillingerWeber_1985_Si__MO_405512056662_005 (Silicon SW)
- EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005 (Aluminum EAM)
- EAM_Dynamo_FoilesBasteseDaw_1986_Cu__MO_831902330215_000 (Copper EAM)

Common atomic masses (metal units):
- Argon: 39.948
- Silicon: 28.0855
- Aluminum: 26.9815
- Copper: 63.546
- Gold: 196.966
- Iron: 55.845

NEVER use these commands (incompatible with KIM):
- pair_style
- pair_coeff

ALWAYS include:
- Model installation comment: # Install model if needed: kim-api-collections-management install user <MODEL_ID>
- Proper thermo output for monitoring
- Appropriate timestep and neighbor settings

If a user requests a material/element, select the most appropriate KIM model from the verified list above."""
        super().__init__(self.system_prompt)

class LammpsReviewerAgent(Agent):
    def __init__(self):
        self.system_prompt = """You are an expert LAMMPS script reviewer specialized in KIM (Knowledge Base of Interatomic Models) potentials. Your role is to analyze LAMMPS input scripts for correctness, completeness, and best practices.

IMPORTANT: This system ONLY supports KIM potentials. Any script using pair_style or pair_coeff commands should be REJECTED.

Review criteria:
1. Script syntax and command correctness
2. Proper KIM initialization sequence
3. KIM potentials ONLY (REJECT any pair_style/pair_coeff usage)
4. Correct thermodynamic output setup
5. Reasonable simulation parameters (timestep, neighbor settings)
6. Proper boundary conditions and system setup
7. Missing or redundant commands
8. Proper units (should be 'metal' for KIM models)

MANDATORY KIM requirements to verify:
- kim init appears BEFORE create_box
- kim interactions appears AFTER create_box
- mass commands are present (REQUIRED even with KIM)
- NO pair_style or pair_coeff commands (REJECT if present)
- Proper species mapping in kim interactions
- Units set to 'metal' (required for most KIM models)
- Valid KIM model ID used

AUTOMATIC REJECTION criteria:
- Any use of pair_style or pair_coeff commands
- Missing kim init command
- Missing kim interactions command
- Missing mass definitions
- kim init appearing after create_box
- kim interactions appearing before create_box

Provide feedback in this format:
REVIEW: [PASS/FAIL]
ISSUES: [List any problems found, or "None" if script is acceptable]
SUGGESTIONS: [Optional improvements or optimizations]

Focus on critical errors that would prevent execution. Scripts using non-KIM potentials should automatically FAIL review."""
        super().__init__(self.system_prompt)
