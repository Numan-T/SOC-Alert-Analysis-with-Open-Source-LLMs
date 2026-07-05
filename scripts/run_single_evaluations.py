"""
import subprocess

# Path to the analysis file that should be judged
analysis_filepath = r"C:\Users\<user_name>\llm_soc_alert_analysis\results\analysis\round_4\mistral_2026_04_30-22_19_34.json"

# Enter strings of <ID.repetition_num> e.g. ["11.1", "89.2"]
single_evals = ["55.3"]

for eval in single_evals:
    id, rep = eval.split('.')
    command = [
        "python", "-m", "src.run_evaluations",
        analysis_filepath,
        "--skip_bleurt",
        "--first_id", id,
        "--last_id", id,
        "--rep_num", rep
    ]

    subprocess.run(
        command,
    )


# Run command: 'python scripts/run_single_evaluations.py'
"""
import subprocess
from typing import List


def run_single_evaluations(analysis_filepath: str, evaluation_targets: List[str]) -> None:
    """Executes targeted evaluations for specific analysis IDs via a subprocess.

    Iterates through a list of evaluation targets, parses the target ID and 
    repetition number, constructs the command-line arguments, and triggers 
    the evaluation module "src.run_evaluations".

    Args:
        analysis_filepath (str): The absolute or relative path to the analysis 
            JSON file to be judged.
        evaluation_targets (List[str]): A list of strings representing the 
            target IDs and their repetition numbers, formatted as '<ID>.
            <repetition_num>' (e.g., '55.3', '89.2').


    Example:
        >>> run_single_evaluations(
        ...     analysis_filepath="results/analysis.json",
        ...     evaluation_targets=["11.1", "55.3"]
        ... )
    """
    for eval_target in evaluation_targets:
        eval_id, rep_num = eval_target.split('.')

        command = [
            "python", "-m", "src.run_evaluations",
            analysis_filepath,
            "--skip_bleurt",
            "--first_id", eval_id,
            "--last_id", eval_id,
            "--rep_num", rep_num
        ]

        # check=True ensures an exception is raised if the command fails
        subprocess.run(command, check=True)


if __name__ == "__main__":
    # Path to the analysis file that should be judged
    ANALYSIS_FILEPATH = r"C:\Users\<user_name>\llm_soc_alert_analysis\results\analysis\round_4\gpt-oss_2026_04_30-10_23_25.json"

    # Enter strings of <ID.repetition_num> e.g. ["11.1", "89.2"]
    SINGLE_EVALS = ["70.1"]

    run_single_evaluations(
        analysis_filepath=ANALYSIS_FILEPATH,
        evaluation_targets=SINGLE_EVALS
    )
