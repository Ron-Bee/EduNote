#!/usr/bin/env python3
"""
Simple wrapper for llama.cpp 'llama-cli' binary.
Uses subprocess to run prompts against a GGUF model.
"""
import subprocess
import shlex
import os
from pathlib import Path
from typing import Optional

class LlamaWrapper:
    def __init__(self, model_path: str, binary_path: Optional[str] = None):
        self.model_path = Path(os.path.expanduser(model_path)).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # Default binary path if not provided
        default_bin = os.path.expanduser("~/EduNote/llama.cpp/build/bin/llama-cli")
        self.binary_path = Path(os.path.expanduser(binary_path or default_bin)).resolve()
        if not self.binary_path.exists():
            raise FileNotFoundError(f"llama-cli binary not found at {self.binary_path}")

    def generate(self, prompt: str, n_predict: int = 128, temperature: float = 0.8, top_k: int = 40, top_p: float = 0.95) -> str:
        """
        Run llama-cli with a prompt and return the output text.
        """
        # Build command list (avoid shell interpolation issues)
        cmd = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(n_predict),
            "--temp", str(temperature),
            "--top-k", str(top_k),
            "--top-p", str(top_p)
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"llama-cli failed (rc={proc.returncode}):\n{proc.stderr}")

        return self._parse_output(proc.stdout)

    @staticmethod
    def _parse_output(stdout: str) -> str:
        """
        Return the generator output text from llama-cli stdout.
        This is intentionally permissive: returns stdout after the first blank line
        following the prompt-related messages, falling back to all stdout.
        """
        lines = stdout.splitlines()
        # Find first empty line after info header, then join remaining lines
        for i, line in enumerate(lines):
            if line.strip() == "":
                return "\n".join(lines[i+1:]).strip()
        return stdout.strip()
