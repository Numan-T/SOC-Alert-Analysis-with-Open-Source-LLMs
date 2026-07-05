"""Coordinates the automated process of setting up and evaluating the models.

This script manages the automated process of setting up the models via external
scripts and then executing benchmark tasks on all the predefined models within the runtime environment.
"""

import subprocess
import sys


def run_command(command):
    """Runs a command in the CMD and stopps at Errors."""
    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"Error while trying command: {command}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    # Download models
    run_command("python -m scripts.setup_models")

    # Run analysis for these models
    models = ["gpt-oss", "qwen", "granite", "mistral", "foundationsec"]
    for model in models:
        run_command(f'python -m src.run_analysis "{model}" --repititions 3')

    print("-"*39)
    print("--- All docker tasks are completed! ---")
    print("-"*39)
