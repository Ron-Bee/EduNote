# llama_runner.py
# Simple wrapper that calls llama.cpp CLI (./main) with a prompt and returns stdout.
# Update MODEL_PATH to your downloaded/quantized .gguf model path.

import subprocess
import shlex
from pathlib import Path
from typing import Optional

LLAMA_MAIN_BIN = "./main"  # path to llama.cpp built binary
MODEL_PATH = "models/tinyllama.gguf"  # update to your model path

def run_llama_once(prompt: str, model_path: Optional[str] = None, n_predict: int = 512) -> str:
    """
    Calls llama.cpp `main` binary with a single prompt (-p).
    Returns the generated text (stdout).
    NOTE: CLI flags vary by llama.cpp versions; this uses common flags.
    """
    model = model_path or MODEL_PATH
    model_file = Path(model)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model}. Place a quantized model there.")

    # Basic invocation. Tweak flags as needed (temperature, n_predict, tokens, etc.)
    cmd = f"{LLAMA_MAIN_BIN} -m {shlex.quote(str(model))} -p {shlex.quote(prompt)} -n {n_predict}"
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        # forward stderr for debugging
        raise RuntimeError(f"llama.cpp failed: {result.stderr.strip()}")
    return result.stdout.strip()
