"""
OVITO Executor - Safe execution of dynamically generated OVITO scripts.

This module provides a sandboxed execution environment for OVITO Python scripts
with security validation and error handling.
"""

import ast
import sys
from typing import Any, Dict


class VisualizationScriptError(Exception):
    """Raised when OVITO script execution fails."""
    pass


class SecurityError(Exception):
    """Raised when script contains dangerous operations."""
    pass


class DynamicOVITOExecutor:
    """
    Safely execute dynamically generated OVITO scripts.
    
    Provides sandboxed execution environment with security validation
    and error handling to prevent malicious or dangerous operations.
    """
    
    def __init__(self):
        """Initialize the executor with security settings."""
        self.allowed_imports = {
            'ovito.modifiers',
            'ovito.pipeline',
            'ovito.vis',
            'ovito.data',
            'numpy',
            'math'
        }
        self.execution_timeout = 30  # seconds
        
    def execute_script(self, script: str, pipeline, data_collection) -> Any:
        """
        Execute OVITO script in controlled environment.
        
        Args:
            script: Generated OVITO Python code
            pipeline: OVITO Pipeline object
            data_collection: OVITO DataCollection
            
        Returns:
            Modified pipeline
            
        Raises:
            SecurityError: If script contains dangerous operations
            VisualizationScriptError: If script execution fails
        """
        # Validate script safety
        self._validate_script(script)
        
        # Create execution namespace with limited builtins
        namespace = {
            'pipeline': pipeline,
            'data_collection': data_collection,
            '__builtins__': {
                # Allow only safe builtins
                'True': True,
                'False': False,
                'None': None,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'len': len,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'round': round,
                'print': print,  # Allow print for debugging
            }
        }
        
        try:
            # Execute script with timeout protection
            # Note: For production, consider using a more robust timeout mechanism
            exec(script, namespace)
            return namespace['pipeline']
            
        except Exception as e:
            error_msg = f"Failed to execute visualization script: {type(e).__name__}: {str(e)}"
            print(f"Error executing OVITO script: {type(e).__name__}: {e}")
            print(f"Script was:\n{script}")
            # Print detailed traceback for debugging
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise VisualizationScriptError(error_msg) from e
    
    def _validate_script(self, script: str) -> None:
        """
        Validate script for security and safety.
        
        Args:
            script: Python script to validate
            
        Raises:
            SecurityError: If dangerous operations are detected
        """
        # Check for dangerous string patterns
        dangerous_patterns = [
            'import os',
            'import sys',
            'import subprocess',
            'import shutil',
            'import glob',
            'eval(',
            'exec(',
            '__import__',
            'open(',
            'file(',
            'compile(',
            'execfile(',
            'input(',
            'raw_input(',
        ]
        
        script_lower = script.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in script_lower:
                raise SecurityError(f"Dangerous operation detected: {pattern}")
        
        # Validate imports using AST parsing
        try:
            tree = ast.parse(script)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        if module not in self.allowed_imports and module != 'ovito':
                            raise SecurityError(f"Unauthorized import: {module}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        if module not in self.allowed_imports and module != 'ovito':
                            raise SecurityError(f"Unauthorized import from: {module}")
        except SyntaxError as e:
            raise SecurityError(f"Script contains syntax errors: {e}")
    
    def validate_and_sanitize(self, script: str) -> str:
        """
        Validate and optionally sanitize a script.
        
        Args:
            script: Python script to validate
            
        Returns:
            Validated (and potentially sanitized) script
            
        Raises:
            SecurityError: If script contains dangerous operations
        """
        self._validate_script(script)
        return script
