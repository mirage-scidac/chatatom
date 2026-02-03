from openai import OpenAI
import config
import kim_utils
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import dspy
from prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    LAMMPS_GENERATOR_SYSTEM_PROMPT,
    LAMMPS_REVIEWER_SYSTEM_PROMPT,
    AnalyzeRequest,
    VerifySimulationPlan,
    GenerateLAMMPSScript,
    ReviewLAMMPSScript,
    SelectTools,
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


# =============================================================================
# DSPy-Based Agentic Modules
# =============================================================================

def configure_dspy_lm():
    """Configure DSPy to use the Argonne API endpoint."""
    lm = dspy.LM(
        model=f'openai/{config.DEFAULT_MODEL}',
        api_base=config.API_BASE_URL,
        api_key=config.API_KEY,
        temperature=0.1
    )
    dspy.configure(lm=lm)
    return lm


@dataclass
class GenerationResult:
    """Result from the iterative LAMMPS generation process."""
    script: str
    review_passed: bool
    attempts: int
    issues: List[str]
    suggestions: List[str]
    conversation: List[Dict[str, str]]
    confidence: float


class IterativeLammpsGeneration(dspy.Module):
    """
    DSPy module that implements iterative LAMMPS script generation with review.

    This module generates a LAMMPS script, reviews it, and if issues are found,
    iteratively improves it up to max_attempts times.
    """

    def __init__(self, max_attempts: int = 3):
        super().__init__()
        self.max_attempts = max_attempts
        self.generator = dspy.ChainOfThought(GenerateLAMMPSScript)
        self.reviewer = dspy.ChainOfThought(ReviewLAMMPSScript)

    def forward(
        self,
        material: str,
        kim_model: str,
        temperature: float,
        ensemble: str,
        duration_ps: float,
        system_size: int = 5,
        thermo_freq: int = 100,
        dump_freq: int = 1000,
        visualization_notes: str = ""
    ) -> GenerationResult:
        """
        Generate and iteratively refine a LAMMPS script.

        Returns:
            GenerationResult with the final script and metadata
        """
        conversation = []
        current_script = ""
        all_issues = []
        all_suggestions = []

        for attempt in range(1, self.max_attempts + 1):
            # Generate or regenerate script
            if attempt == 1:
                gen_result = self.generator(
                    material=material,
                    kim_model=kim_model,
                    temperature=temperature,
                    ensemble=ensemble,
                    duration_ps=duration_ps,
                    system_size=system_size,
                    thermo_freq=thermo_freq,
                    dump_freq=dump_freq,
                    visualization_notes=visualization_notes
                )
            else:
                # Include previous issues in visualization_notes for context
                issues_context = f"Previous issues to fix: {'; '.join(all_issues[-3:])}"
                gen_result = self.generator(
                    material=material,
                    kim_model=kim_model,
                    temperature=temperature,
                    ensemble=ensemble,
                    duration_ps=duration_ps,
                    system_size=system_size,
                    thermo_freq=thermo_freq,
                    dump_freq=dump_freq,
                    visualization_notes=f"{visualization_notes}\n{issues_context}"
                )

            current_script = gen_result.lammps_script
            conversation.append({
                "role": "generator",
                "attempt": attempt,
                "content": current_script,
                "explanation": gen_result.explanation
            })

            # Review the script
            review_result = self.reviewer(
                lammps_script=current_script,
                intended_parameters=f"{material} at {temperature}K, {ensemble}, {duration_ps}ps"
            )

            conversation.append({
                "role": "reviewer",
                "attempt": attempt,
                "is_valid": review_result.is_valid,
                "issues": review_result.issues,
                "suggestions": review_result.suggestions,
                "severity": review_result.severity
            })

            all_issues.extend(review_result.issues)
            all_suggestions.extend(review_result.suggestions)

            # If review passes, we're done
            if review_result.is_valid or review_result.severity == "none":
                return GenerationResult(
                    script=current_script,
                    review_passed=True,
                    attempts=attempt,
                    issues=[],
                    suggestions=review_result.suggestions,
                    conversation=conversation,
                    confidence=0.9 if attempt == 1 else 0.8 - (attempt - 1) * 0.1
                )

        # Max attempts reached without passing review
        return GenerationResult(
            script=current_script,
            review_passed=False,
            attempts=self.max_attempts,
            issues=all_issues[-5:],  # Last 5 issues
            suggestions=all_suggestions[-3:],
            conversation=conversation,
            confidence=0.3
        )


@dataclass
class VerificationResult:
    """Result from plan verification."""
    is_feasible: bool
    issues: List[str]
    suggestions: List[str]
    confidence: float
    kim_model_available: bool


class PlanVerifier(dspy.Module):
    """
    DSPy module that verifies simulation feasibility before generation.

    Checks KIM model availability, physical parameter reasonableness,
    and other prerequisites.
    """

    def __init__(self):
        super().__init__()
        self.verifier = dspy.ChainOfThought(VerifySimulationPlan)

    def forward(
        self,
        material: str,
        kim_model: str,
        temperature: float,
        ensemble: str,
        duration_ps: float,
        system_size: int = 5
    ) -> VerificationResult:
        """
        Verify that a simulation plan is feasible.

        Performs both LLM-based verification and programmatic checks.
        """
        issues = []
        suggestions = []

        # Programmatic check: Is KIM model installed?
        kim_installed = kim_utils.is_model_installed(kim_model)
        if not kim_installed:
            issues.append(f"KIM model '{kim_model}' is not installed")
            suggestions.append(f"Install with: kim-api-collections-management install user {kim_model}")

        # Programmatic check: Do we have element properties?
        element_props = kim_utils.get_element_properties(material)
        if not element_props:
            issues.append(f"No element properties found for '{material}'")
            suggestions.append("Check if the element symbol is correct")

        # Programmatic check: Temperature range
        if temperature < 0:
            issues.append("Temperature cannot be negative")
        elif temperature > 10000:
            issues.append("Temperature unusually high (>10000K)")
            suggestions.append("Most simulations use temperatures below 5000K")

        # Programmatic check: Duration
        if duration_ps <= 0:
            issues.append("Duration must be positive")
        elif duration_ps > 1000:
            suggestions.append("Long simulation (>1ns) - consider using fewer trajectory frames")

        # LLM-based verification for more nuanced checks
        try:
            llm_result = self.verifier(
                material=material,
                kim_model=kim_model,
                temperature=temperature,
                ensemble=ensemble,
                duration_ps=duration_ps,
                system_size=system_size
            )
            issues.extend(llm_result.issues)
            suggestions.extend(llm_result.suggestions)
            llm_confidence = llm_result.confidence
        except Exception as e:
            # LLM verification failed, use only programmatic checks
            llm_confidence = 0.5

        is_feasible = len([i for i in issues if "not installed" not in i.lower()]) == 0
        # Confidence based on number of issues and LLM confidence
        confidence = max(0.1, llm_confidence - len(issues) * 0.1)

        return VerificationResult(
            is_feasible=is_feasible,
            issues=issues,
            suggestions=suggestions,
            confidence=confidence,
            kim_model_available=kim_installed
        )


@dataclass
class AnalysisResult:
    """Result from request analysis."""
    material: str
    temperature: float
    ensemble: str
    duration_ps: float
    kim_model: str
    system_size: int
    needs_clarification: bool
    clarification_question: str
    confidence: float


class SmartOrchestrator(dspy.Module):
    """
    DSPy module for intelligent request analysis and tool selection.

    Analyzes user requests, extracts parameters, and determines
    what tools/actions are needed.
    """

    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(AnalyzeRequest)
        self.tool_selector = dspy.ChainOfThought(SelectTools)

    def analyze_request(
        self,
        user_request: str,
        registry_context: str = ""
    ) -> AnalysisResult:
        """
        Analyze a user request and extract simulation parameters.
        """
        result = self.analyzer(
            user_request=user_request,
            registry_context=registry_context
        )

        return AnalysisResult(
            material=result.material,
            temperature=result.temperature,
            ensemble=result.ensemble,
            duration_ps=result.duration_ps,
            kim_model=result.kim_model,
            system_size=result.system_size,
            needs_clarification=result.needs_clarification,
            clarification_question=result.clarification_question,
            confidence=result.confidence
        )

    def select_tools(
        self,
        user_request: str,
        available_tools: List[str],
        conversation_context: str = ""
    ) -> Tuple[List[str], List[str], str]:
        """
        Select appropriate tools for handling a request.

        Returns:
            Tuple of (selected_tools, tool_arguments, reasoning)
        """
        result = self.tool_selector(
            user_request=user_request,
            available_tools=available_tools,
            conversation_context=conversation_context
        )

        return result.selected_tools, result.tool_arguments, result.reasoning
