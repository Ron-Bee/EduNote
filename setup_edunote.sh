#!/usr/bin/env bash
# EduNote Automated Setup Script
# Author: Ron Baute
# Purpose: Setup Python venv, directory structure, llama.cpp build, and wrapper scripts

BASE="$HOME/EduNote"
EDU_LLMPKG="$BASE/edunote_llm"
MODELS_ALPACA="$BASE/models/alpaca"
MODELS_TINY="$BASE/models/tinyllama"
LLAMA_BIN="$BASE/llama.cpp/build/bin/llama-cli"
MODEL_FILE_ALPACA_REL="models/alpaca/claude2-alpaca-7b.Q3_K_M.gguf"
VENV_DIR="$BASE/venv"

echo "=== EduNote Automated Setup ==="
echo "Base Directory: $BASE"

# 1) Create directories
mkdir -p "$BASE" "$BASE/app" "$EDU_LLMPKG" "$MODELS_ALPACA" "$MODELS_TINY" "$(dirname "$LLAMA_BIN")"

# 2) Create Python package init
[[ ! -f "$EDU_LLMPKG/__init__.py" ]] && cat > "$EDU_LLMPKG/__init__.py" <<'EOF'
# edunote_llm package
__version__ = "0.1"
EOF

# 3) Write llama_wrapper.py
cat > "$EDU_LLMPKG/llama_wrapper.py" <<'EOF'
#!/usr/bin/env python3
import subprocess, os
from pathlib import Path
from typing import Optional

class LlamaWrapper:
    def __init__(self, model_path: str, binary_path: Optional[str] = None):
        self.model_path = Path(os.path.expanduser(model_path)).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        default_bin = os.path.expanduser("~/EduNote/llama.cpp/build/bin/llama-cli")
        self.binary_path = Path(os.path.expanduser(binary_path or default_bin)).resolve()
        if not self.binary_path.exists():
            raise FileNotFoundError(f"Binary not found at {self.binary_path}")

    def generate(self, prompt: str, n_predict: int=128, temperature: float=0.8, top_k: int=40, top_p: float=0.95) -> str:
        cmd = [str(self.binary_path), "-m", str(self.model_path), "-p", prompt,
               "-n", str(n_predict), "--temp", str(temperature), "--top-k", str(top_k), "--top-p", str(top_p)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"llama-cli failed (rc={proc.returncode}):\n{proc.stderr}")
        lines = proc.stdout.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "":
                return "\n".join(lines[i+1:]).strip()
        return proc.stdout.strip()
EOF
chmod +x "$EDU_LLMPKG/llama_wrapper.py"

# 4) Write run_llama.py
cat > "$EDU_LLMPKG/run_llama.py" <<'EOF'
#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edunote_llm.llama_wrapper import LlamaWrapper

def main():
    model_path = os.path.expanduser("~/EduNote/models/alpaca/claude2-alpaca-7b.Q3_K_M.gguf")
    binary_path = os.path.expanduser("~/EduNote/llama.cpp/build/bin/llama-cli")
    wrapper = LlamaWrapper(model_path, binary_path)
    prompt = "Summarize: Photosynthesis converts sunlight into energy."
    out = wrapper.generate(prompt, n_predict=64)
    print("\n=== MODEL OUTPUT ===\n", out)

if __name__ == "__main__":
    main()
EOF
chmod +x "$EDU_LLMPKG/run_llama.py"

# 5) Create top-level runner
RUN_TOP="$BASE/run_edunote.py"
cat > "$RUN_TOP" <<'EOF'
#!/usr/bin/env python3
from edunote_llm.run_llama import main
if __name__ == "__main__":
    main()
EOF
chmod +x "$RUN_TOP"

# 6) Move stray model files to models
shopt -s nullglob
for f in "$BASE"/*.gguf "$BASE"/*.bin "$BASE"/*.gz; do
  mv "$f" "$MODELS_ALPACA"/
done
shopt -u nullglob

# 7) Setup virtual environment & install dependencies
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
if [ -f "$BASE/requirements.txt" ]; then
  pip install --upgrade pip
  pip install -r "$BASE/requirements.txt"
else
  pip install --upgrade pip
  pip install huggingface-hub fastapi uvicorn python-multipart pdfminer.six python-docx
fi
deactivate

echo "=== EduNote Setup Complete ==="
echo "Activate venv: source $VENV_DIR/bin/activate"
echo "Run model: python3 $RUN_TOP"
