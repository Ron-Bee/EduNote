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
