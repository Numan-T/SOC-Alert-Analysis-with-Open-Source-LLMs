### Automated SOC Alert Analysis and Evaluation Pipeline

This repository contains the source code and evaluation framework for my Master's Thesis "SOC Alert Analysis with Open-Source Large Language Models: A Feasibility Stude"

The project evaluates the capabilities of various Large Language Models (LLMs) in automating Security Operations Center (SOC) alert analysis, utilizing an LLM-as-a-judge framework and synthetic Ground Truth References created with GPT 5.2.

# Directory Tree

├── data/ # Test alerts and master reference Excel sheets
├── docker/ # Docker setup for the evaluation pipelineS
├── models/ # Huggingface Modelfiles for Candidate Model setup via 'scripts/setup_models.py'
├── prompts/ # Analysis, LLM-judge and Reference Generator prompt templates
├── reports/ # PoerBI reports for statistical analysis of evaluation results and graphics creation
├── results/
│ ├── analysis/ # Raw inference JSON logs from candidate models
│ └── evaluation/ # LLM Judge evaluation results
├── scripts/ # Scripts for repeating and merging single analysis, model setup and reference generation
├── src/
│ ├── config/ # Central pipeline configurations
│ ├── core/ # Ollama client and prompt assemblers
│ └── evaluation/ # Evaluator class and Judge logic
│ └── utils/ # Helper functions
│ └── run_analysis.py # CLI entry point for alert analysis by candidate models
└──── run_evaluations.py # CLI entry point for evaluation via the LLM Judge

## Project Setup:

# Prerequisites:

- Environment Variables: Create a .env file in the root directory:
  GOOGLE_API_KEY="..." (Gemini Judge - needed for Evaluation)
  OPENAI_API_KEY="..." (GPT Ground Truth Generator - needed only for synthetic ground truth generation)

# Local Environment Setup:

1. Create conda environmant (e.g. named "llm_soc_alert_analysis_venv")
2. Activate environment (run "conda activate llm_soc_alert_analysis_venv") Install requirements via "pip install -r requirements.txt"
3. Setup LLM models by running "python scripts/setup_models.py" (only first time)
4. Activate ollama by entering this command in the terminal: "ollama serve"

# Run Analysis Locally:

1. See step "Local Environment Setup"
2. Run 'python -m src.run_analysis "<model_id>"'
   E.g. python -m src.run_analysis "foundationsec"
   or python -m src.run_analysis "foundationsec" --repititions 5

Model IDs | Model URL:

---

granite | hf.co/ibm-granite/granite-3.3-8b-instruct-GGUF:Q4_K_M
qwen | hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
gpt-oss | hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_M
mistral | hf.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M
foundationsec | hf.co/fdtn-ai/Foundation-Sec-1.
| 1-8B-Instruct-Q4_K_M-GGUF:Q4_K_M

# Run Analysis on Linux VM via Docker:

1. Setup VM and docker:
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   newgrp docker
2. Clone this repository:
   git clone https://github.com/Numan-T/SOC-Alert-Analysis-with-Open-Source-LLMs.git
3. Adjust evaluation configuration if necessary:
   Adjust 'models' list or 'repititions' value in 'docker_script.py'
4. Start code
   cd SOC-Alert-Analysis-with-Open-Source-LLMs
   docker compose up --build -d
5. In case it is the first usage, repeat these stept to clear memory:
   docker compose down
   docker compose up --build -d
6. Results will be safed in 'results/' directory

# Run Evaluation Locally (no docker setup):

1. See step "Local Environment Setup"
2. Start Evaluation for a specific analysis file:
   Run 'python -m src.run_evaluations "<analysis_filepath>"'
   or 'python -m src.run_evaluations "<analysis_filepath>" --first_id=<int> --last_id=<int> --rep_num=<int>'

# How to install new models from huggingface (additionally to the evaluated candidate models):

1. Get the model name from Huggingface.co and look for preferred quantization. It must be available in GGUF-format.
2. Enter "ollama pull hf.co/<model name>:<quantization>".
3. Enter "ollama serve" to activate ollama and use the model.
4. Change the "model_name" variable in run_analysis.py to the exact same link from step 2 to "hf.co/<model name>:<quantization>".
