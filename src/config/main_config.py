from pathlib import Path

# Files
PROJECT_DIR = Path(__file__).parent.parent.parent
# SYS_PROMPT_FILENAME = "1-2_analysis_prompt_(baseline).txt"
SYS_PROMPT_FILENAME = "3_analysis_prompt_(few-shot).txt"
# SYS_PROMPT_FILENAME = "4_analysis_prompt_(prompt_engineering).txt"
TEST_ALERTS_FILE = "soc_alerts_and_ground_truth_data.xlsx"
TEST_ALERTS_SHEET_NAME = "Alerts"
# SILVER_PROMPT_FILE = "silver_standard_generation_prompt.txt"
SILVER_PROMPT_FILE = "silver_standard_generation_prompt_IPI_adjusted.txt"

# Silver Gen
SILVER_GEN_MODEL_NAME = 'gpt-5.2-2025-12-11'
SILVER_GEN_TEMPERATURE = 0.2

# Judge
JUDGE_MODEL_NAME = 'gemini-2.5-pro'
JUDGE_TEMP = 0
JUDGE_THOUGHTS = True

# Analysis Parameters
ANALYSIS_TEMPERATURE = 0.2
ANALYSIS_REPEAT_PENALTY = None  # 1.2

# Set Range of alerts to analyze or evaluate (run_analysis/run_evaluation)
FIRST_ALERT_ID = 92  # 1
LAST_ALERT_ID = 98  # 91
