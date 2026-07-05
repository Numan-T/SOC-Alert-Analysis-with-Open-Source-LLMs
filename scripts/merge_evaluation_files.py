import json
import os
import glob
from datetime import datetime
from src.config import main_config as conf


def merge_evaluation_folder(
    source_folder: str,
    file_prefix: str,
    max_alert_id: int,
    max_repetition_num: int,
    output_dir: str = None
):
    """
    Scans a directory for JSON files matching a prefix, loads them all, 
    and merges them into a single file. 
    Iterates through the expected Alert IDs and Repetitions and picks 
    the first available data point found across the loaded files.

    Args:
        source_folder (str): The directory path containing the JSON files to 
            merge.
        file_prefix (str): The prefix used to identify relevant files 
            (e.g., 'eval_mistral_2026_04_30').
        max_alert_id (int): The maximum alert ID number to iterate through.
        max_repetition_num (int): The maximum repetition number per alert ID.
        output_dir (str, optional): The directory path to save the merged file. 
            Defaults to None, which saves the output directly in the 
            `source_folder`.

    Returns:
        None: The function writes the merged data to a new JSON file and prints 
        the status and missing data logs to the console.
    """

    # --- Load Files ---
    search_pattern = os.path.join(source_folder, f"{file_prefix}*.json")
    file_paths = glob.glob(search_pattern)

    file_paths.sort()

    if not file_paths:
        print(
            f"ERROR: No files found matching prefix '{file_prefix}' in folder:\n{source_folder}")
        return

    print(f"Found {len(file_paths)} files to merge:")
    for p in file_paths:
        print(f" - {os.path.basename(p)}")

    loaded_datasets = []

    for fp in file_paths:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
                loaded_datasets.append(data)
        except Exception as e:
            print(
                f"WARNING: Could not load {os.path.basename(fp)}. Skipping. Error: {e}")

    if not loaded_datasets:
        print("ERROR: No valid data loaded.")
        return

    # --- Helper functions ---
    def normalize_id(val):
        """Helper to ensure IDs match (e.g., '1.0' becomes '1')"""
        return str(val).replace('.0', '').strip()

    def index_bleurt(data):
        indexed = {}
        scores = data.get('bleurt_scores')
        if not isinstance(scores, dict) or 'per_alert_statistics' not in scores:
            return indexed

        for item in scores['per_alert_statistics']:
            a_id = normalize_id(item['alert_id'])
            indexed[a_id] = {}
            for i, score in enumerate(item.get('single_scores', [])):
                indexed[a_id][i + 1] = score
        return indexed

    def index_judge(data):
        indexed = {}
        scores = data.get('llm_judge_scores')
        if not isinstance(scores, dict) or 'per_alert_statistics' not in scores:
            return indexed

        for item in scores['per_alert_statistics']:
            a_id = normalize_id(item['alert_id'])
            indexed[a_id] = {}
            for judgement in item.get('single_judgements', []):
                rep_num = judgement.get('repetition_num')
                if rep_num:
                    indexed[a_id][rep_num] = judgement
        return indexed

    # Create indices for easy data access
    all_bleurt_maps = [index_bleurt(d) for d in loaded_datasets]
    all_judge_maps = [index_judge(d) for d in loaded_datasets]

    # --- Merge ---
    merged_bleurt_list = []
    merged_judge_list = []
    missing_data_bleurt = []
    missing_data_judge = []

    # Iterate through the alerts and repititions
    for alert_id_int in range(1, max_alert_id + 1):
        alert_id_str = str(alert_id_int)

        alert_bleurt_entry = {
            "alert_id": alert_id_str, "total_alert_mean": 0, "total_alert_stdev": 0,
            "single_scores": []
        }

        alert_judge_entry = {
            "alert_id": alert_id_str, "total_alert_mean": 0, "total_alert_stdev": 0,
            "single_judgements": []
        }

        # Repitition Loop
        for rep in range(1, max_repetition_num + 1):
            # --- BLEURT merge ---
            val_b = None
            for b_map in all_bleurt_maps:
                temp_val = b_map.get(alert_id_str, {}).get(rep)
                if temp_val is not None:
                    val_b = temp_val
                    break

            if val_b is not None:
                alert_bleurt_entry['single_scores'].append(val_b)
            else:
                missing_data_bleurt.append(f"{alert_id_str}.{rep}")

            # --- Judge merge ---
            val_j = None
            for j_map in all_judge_maps:
                temp_val = j_map.get(alert_id_str, {}).get(rep)
                if temp_val is not None:
                    val_j = temp_val
                    break

            if val_j is not None:
                alert_judge_entry['single_judgements'].append(val_j)
            else:
                missing_data_judge.append(f"{alert_id_str}.{rep}")

        if alert_bleurt_entry['single_scores']:
            merged_bleurt_list.append(alert_bleurt_entry)
        if alert_judge_entry['single_judgements']:
            merged_judge_list.append(alert_judge_entry)

    merged_data = loaded_datasets[0].copy()

    def find_valid_structure(datasets, key):
        """
        Searches for a valid JSON structure as basis for the final merged file.
        """
        for d in datasets:
            if isinstance(d.get(key), dict):
                return d[key].copy()
        return {
            "evaluation_metadata": {},
            "between_alert_statistics": {},
            "per_alert_statistics": []
        }

    if merged_bleurt_list:
        if not isinstance(merged_data.get('bleurt_scores'), dict):
            # Template is broken/skipped, find a better one
            merged_data['bleurt_scores'] = find_valid_structure(
                loaded_datasets, 'bleurt_scores')

        merged_data['bleurt_scores']['per_alert_statistics'] = merged_bleurt_list

    if merged_judge_list:
        if not isinstance(merged_data.get('llm_judge_scores'), dict):
            # Template is broken/skipped, find a better one
            merged_data['llm_judge_scores'] = find_valid_structure(
                loaded_datasets, 'llm_judge_scores')

        merged_data['llm_judge_scores']['per_alert_statistics'] = merged_judge_list

    # Replace statistics with zeros as placesholders
    def reset_stats(stats_dict):
        for key in stats_dict:
            stats_dict[key] = 0

    if isinstance(merged_data.get('bleurt_scores'), dict) and \
       'between_alert_statistics' in merged_data['bleurt_scores']:
        reset_stats(merged_data['bleurt_scores']['between_alert_statistics'])

    if isinstance(merged_data.get('llm_judge_scores'), dict) and \
       'between_alert_statistics' in merged_data['llm_judge_scores']:
        reset_stats(merged_data['llm_judge_scores']
                    ['between_alert_statistics'])

     # Save with new timestamp: replaces evaluation timestamp by merge timestamp
    new_eval_timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    merged_data['evaluation_timestamp'] = new_eval_timestamp

    new_filename = f"{file_prefix}_MERGED_{new_eval_timestamp}.json"

    if output_dir:
        output_path = os.path.join(output_dir, new_filename)
    else:
        output_path = os.path.join(source_folder, new_filename)

    try:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(merged_data, f_out, indent=4)
        print(f"Merge completed! Saved file: {output_path}")
    except Exception as e:
        print(f"ERROR saving file: {e}")

    # Print information about still missing values and output path
    print(f"\nMissing data for LLM judge after merge: ")
    if missing_data_bleurt:
        print(f"Total Missing: {len(missing_data_bleurt)}")
        print(missing_data_bleurt)
    else:
        print("No data missing.")

    print(f"\nMissing data for LLM judge after merge: ")
    if missing_data_judge:
        print(f"Total Missing: {len(missing_data_judge)}")
        print(missing_data_judge)
    else:
        print("No data missing.")


if __name__ == "__main__":
    # Run 'python -m scripts.merge_evaluation_files'

    # Folder with all files to merge
    FOLDER = r"C:\Users\Numan - Uni & Privat\llm_soc_alert_analysis\results\evaluation\round_1_judge_repetition\merged_files\merged_qwen"
    folder_path = conf.PROJECT_DIR / FOLDER

    # Prefix for identification of file belonging to the same analysis
    # (e.g. "eval_gpt-oss_2026_01_29-19_07_57")
    PREFIX = "eval_qwen_2026_01_29-22_12_49"

    MAX_ID = 91
    MAX_REP = 3

    merge_evaluation_folder(folder_path, PREFIX, MAX_ID, MAX_REP)
