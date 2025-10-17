
# EduNote

EduNote is an open-source project for AI-assisted document handling, summarization, and automation.  
It provides a Python backend, a llama.cpp wrapper for local LLM execution, and setup scripts to streamline deployment.

## Features

- Run local LLM models using `llama.cpp`
- Automatic project setup with `setup_edunote.sh`
- Python virtual environment creation and dependency installation
- Example scripts for summarization and document processing
- Supports Alpaca and TinyLlama GGUF models

## Prerequisites

- Python 3.10 or later
- Git
- bash shell (Linux/macOS) or WSL for Windows

## Quick Setup

1. **Clone the repository (with submodules):**

```bash
git clone --recurse-submodules git@github.com:Ron-Bee/EduNote.git
cd EduNote
