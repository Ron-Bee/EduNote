#!/usr/bin/env bash
set -euo pipefail

BASE="$HOME/EduNote"
EDU_LLMPKG="$BASE/edunote_llm"
MODELS_ALPACA="$BASE/models/alpaca"
MODELS_TINY="$BASE/models/tinyllama"
LLAMA_BIN="$BASE/llama.cpp/build/bin/llama-cli"
MODEL_FILE_ALPACA_REL="models/alpaca/claude2-alpaca-7b.Q3_K_M.gguf"
VENV_DIR="$BASE/venv"

echo
echo "=== EduNote automated setup starting ==="
echo "Base dir: $BASE"
echo

# 1) Create directory layout
mkdir -p "$BASE"
mkdir -p "$BASE/app"
mkdir -p "$EDU_LLMPKG"
mkdir -p "$MODELS_ALPACA"
mkdir -p "$MODELS_TINY"
mkdir -p "$(dirname "$LLAMA_BIN")"

# 2) Create Python package init
if [ ! -f "$EDU_LLMPKG/__init__.py" ]; then
  cat > "$EDU_LLMPKG/__init__.py" <<'PY_INIT'
# edunote_llm package
__version__ = "0.1"
PY_INIT
  echo "Created $EDU_LLMPKG/__init__.py"
fi

# 3) Write llama_wrapper.py (overwrite safe)
cat > "$EDU_LLMPKG/llama_wrapper.py" <<'PY_WRAPPER'
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
PY_WRAPPER

chmod +x "$EDU_LLMPKG/llama_wrapper.py"
echo "Wrote $EDU_LLMPKG/llama_wrapper.py"

# 4) Write run_llama.py (runner)
cat > "$EDU_LLMPKG/run_llama.py" <<'PY_RUN'
#!/usr/bin/env python3
import os
import sys

# Ensure the parent directory (EduNote) is on sys.path so imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edunote_llm.llama_wrapper import LlamaWrapper

def main():
    # default model path (edit if needed)
    model_path = os.path.expanduser("~/EduNote/models/alpaca/claude2-alpaca-7b.Q3_K_M.gguf")
    binary_path = os.path.expanduser("~/EduNote/llama.cpp/build/bin/llama-cli")

    try:
        wrapper = LlamaWrapper(model_path=model_path, binary_path=binary_path)
    except Exception as e:
        print("ERROR initializing LlamaWrapper:", e)
        sys.exit(2)

    prompt = "Summarize: Photosynthesis converts sunlight into energy."
    print("Running model with prompt:\n", prompt, "\n")
    try:
        out = wrapper.generate(prompt, n_predict=64)
        print("\n=== MODEL OUTPUT ===\n")
        print(out)
    except Exception as e:
        print("ERROR during generation:", e)
        sys.exit(3)

if __name__ == '__main__':
    main()
PY_RUN

chmod +x "$EDU_LLMPKG/run_llama.py"
echo "Wrote $EDU_LLMPKG/run_llama.py"

# 5) Create top-level convenience runner
RUN_TOP="$BASE/run_edunote.py"
cat > "$RUN_TOP" <<'RUN_TOP_PY'
#!/usr/bin/env python3
from edunote_llm.run_llama import main

if __name__ == "__main__":
    main()
RUN_TOP_PY
chmod +x "$RUN_TOP"
echo "Created top-level runner $RUN_TOP"

# 6) Move stray model files from base into models if present (non-destructive)
shopt -s nullglob
for f in "$BASE"/*.gguf "$BASE"/*.bin "$BASE"/*.gz; do
  if [ -f "$f" ]; then
    echo "Found stray model file: $f -> moving to $MODELS_ALPACA"
    mv "$f" "$MODELS_ALPACA"/
  fi
done
shopt -u nullglob

# 7) Create Python virtual environment and install dependencies
echo
echo "=== Python virtualenv setup ==="
if [ -d "$VENV_DIR" ]; then
  echo "Virtualenv already exists at $VENV_DIR"
else
  python3 -m venv "$VENV_DIR"
  echo "Created venv at $VENV_DIR"
fi

# Activate for the scope of this script to install packages
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# If a requirements.txt exists at project root, use it; otherwise install defaults
if [ -f "$BASE/requirements.txt" ]; then
  echo "Installing Python packages from $BASE/requirements.txt"
  pip install --upgrade pip
  pip install -r "$BASE/requirements.txt"
else
  echo "No requirements.txt found; installing minimal default packages"
  pip install --upgrade pip
  pip install huggingface-hub fastapi uvicorn python-multipart pdfminer.six python-docx
fi

# 8) Verify llama-cli binary and model presence and print status
deactivate

echo
if [ -f "$LLAMA_BIN" ]; then
  echo "OK: found llama-cli -> $LLAMA_BIN"
else
  echo "WARNING: llama-cli not found at $LLAMA_BIN"
  echo "If you built llama.cpp in a different location, edit $EDU_LLMPKG/llama_wrapper.py or specify binary path."
fi

if [ -f "$BASE/$MODEL_FILE_ALPACA_REL" ]; then
  echo "OK: model file present -> $BASE/$MODEL_FILE_ALPACA_REL"
else
  echo "WARNING: Alpaca model not found at $BASE/$MODEL_FILE_ALPACA_REL"
  echo "Place your .gguf file there or edit the path in edunote_llm/run_llama.py"
fi

# 9) Print final layout (small)
echo
echo "=== EduNote (top-level) layout (2 levels) ==="
if command -v tree >/dev/null 2>&1; then
  tree -L 2 "$BASE"
else
  ls -la "$BASE"
  echo "(install 'tree' for nicer output)"
fi

echo
echo "Setup finished. To test run:"
echo "  source \"$VENV_DIR/bin/activate\""
echo "  python3 \"$RUN_TOP\""
echo
echo "To leave the venv use: deactivate"
echo
echo "If you need to change paths, edit:"
echo "  $EDU_LLMPKG/llama_wrapper.py  (binary/model defaults)"
echo "  $EDU_LLMPKG/run_llama.py     (example runner)"
echo
echo "=== Done ==="

