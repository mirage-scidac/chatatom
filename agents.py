
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
        super().__init__(
            "You are a specialist in generating LAMMPS input scripts. "
            "You will receive a user request for a simulation and generate a complete, valid LAMMPS script. "
            "Crucially, you MUST use the 'kim' command to automatically fetch interatomic potentials. "
            "Do not use external potential files or the 'pair_coeff' command with file paths. "
            "Your output should be only the LAMMPS script, enclosed in a single markdown code block."
        )

class LammpsReviewerAgent(Agent):
    def __init__(self):
        super().__init__(
            "You are a specialist in reviewing LAMMPS input scripts for correctness and best practices. "
            "You will receive a LAMMPS script. Your job is to check it for errors. "
            "One of the most important rules to check is that the script uses the 'kim' command for potentials. "
            "If the script is valid, respond with only the word 'APPROVED'. "
            "If the script is invalid, provide a brief explanation of the problem."
        )
