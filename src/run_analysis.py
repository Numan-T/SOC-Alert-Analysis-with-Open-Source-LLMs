"""Coordinates the LLM analysis pipeline for SOC alerts.

The script acts as an entry point for running loading the analysis references and alerts load, setting up the central SOCAlertAnalyst orchestrator and controlling its process priorities, in addition to saving its evaluation log to timestamped JSON files.
"""

import os
import time
import argparse
from pathlib import Path
from src.config import main_config as conf
from src.utils import data_handler
from src.utils.process_priority_handler import set_high_process_prio
from src.core.analyst import SOCAlertAnalyst


def main(model_id: str, num_repetitions: int):
    """Performs a batch analysis of SOC alerts for the specified LLM.

    Load the static prompt for the system and the alert window that has been configured, call the batch processing module of the analyst to obtain the model inferences, and serialize the enriched results to the project's results directory.

    Args:
        model_id: The string identifier of the targeted LLM deployment.
        num_repetitions: Number of evaluations per alert to achieve statistical significance.
    """

    print("-"*50 + f"\n--- Starting analysis for model: {model_id} ---\n")
    # Define project paths
    project_dir = Path(conf.PROJECT_DIR)
    sys_prompt_path = project_dir / "prompts" / conf.SYS_PROMPT_FILENAME
    test_alerts_path = project_dir / "data" / conf.TEST_ALERTS_FILE

    # Load and format alerts
    try:
        alerts = data_handler.load_alerts(test_alerts_path)
        alerts = alerts[conf.FIRST_ALERT_ID-1:conf.LAST_ALERT_ID]
        print(
            f"Running analysis for alert IDs {conf.FIRST_ALERT_ID} to {conf.LAST_ALERT_ID}. Can be adjusted under: 'src/config/main_config'.")
    except Exception as e:
        print(f"ERROR: Loading alerts failed. Error: {e}")
        exit(1)

    # Load system prompt
    sys_prompt = data_handler.read_txt(sys_prompt_path)

    # Analyze all alerts
    analyst = SOCAlertAnalyst(model_id, sys_prompt, num_repetitions)
    processing_info = analyst.analyze_alerts_batch(alerts)

    # Save processing results
    processing_info = {
        "model_id": model_id,
    } | processing_info

    model_id_safe = model_id.replace(':', '-').replace('/', '-')
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    filename = f"{model_id_safe}_{timestamp}.json"
    output_path = project_dir / "results" / "analysis" / filename

    data_handler.save_json(processing_info, output_path)
    print("-"*50 + f"\n - Saved analysis results in: {output_path}")
    print("--- Analysis completed ---\n" + "-"*50)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    set_high_process_prio()

    parser = argparse.ArgumentParser(
        description="Performs SOC alert analysis using an LLM.")
    parser.add_argument(
        "model_id",
        type=str,
        help="The ID of the model to be used (e.g., ‘foundationsec’)."
    )
    parser.add_argument(
        "--repititions",
        type=int,
        default=3,
        help="Number of analysis repetitions for each alert."
    )
    args = parser.parse_args()

    main(model_id=args.model_id, num_repetitions=args.repititions)


#########################################################################
# Run script with 'python -m src.run_analysis "<model_id>"'
# E.g. python -m src.run_analysis "foundationsec"
#  or  python -m src.run_analysis "foundationsec" --repititions 5
#
# Model IDs     | Model URL:
# ------------------------------------------------------------------------
# granite       | hf.co/ibm-granite/granite-3.3-8b-instruct-GGUF:Q4_K_M
# qwen          | hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
# gpt-oss       | hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_M
# mistral       | hf.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M
# foundationsec | hf.co/fdtn-ai/Foundation-Sec-1.
#               | 1-8B-Instruct-Q4_K_M-GGUF:Q4_K_M
#
##########################################################################
