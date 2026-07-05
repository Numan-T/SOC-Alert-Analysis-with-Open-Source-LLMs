import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from src.config import main_config as conf


project_dir = Path(conf.PROJECT_DIR)
silver_prompt_path = project_dir / "prompts" / \
    conf.SILVER_PROMPT_FILE
test_alerts_path = project_dir / "data" / conf.TEST_ALERTS_FILE


class SilverSetGenerator:
    def __init__(self):
        self.client = self._openai_client_connection()

    def _openai_client_connection(self):
        """
        Connects to an open ai model used as ground truth generation model via 
        the official API.
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please check your .env file.")
        client = OpenAI(api_key=api_key)
        return client

    def _send_request_openai_api(self, prompt):
        """
        Sends a request/prompt to the open ai client initialized by 
        self._openai_client_connection().
        """
        response = self.client.chat.completions.create(
            model=conf.SILVER_GEN_MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            reasoning_effort="none",  # Enables temperature param
            temperature=conf.SILVER_GEN_TEMPERATURE
        )
        response = response.choices[0].message.content.strip()
        return response

    def _get_gold_examples(self, df):
        """
        Filters rows that are marked with 'Gold' in the 'Ground Truth Category' 
        column and creates a few-shot prompt from that.
        """
        # Filter by 'Gold' (case insensitive)
        seed_rows = df[df['Ground Truth Category'].astype(
            str).str.lower() == 'gold']

        formatted_examples = ""
        for _, row in seed_rows.iterrows():
            alert_id = row['Alert_id']
            alert = row['Alert']
            gold_example = row['Ground Truth (Handcrafted)']
            formatted_examples += f"\n--- EXAMPLE (ID: {alert_id}) ---\n"
            formatted_examples += f"INPUT ALERT:\n{alert}\n\n"
            formatted_examples += f"IDEAL ANALYSIS:\n{gold_example}\n"

        print(
            f"Created prompt for synthetic ground truth data generation with {len(seed_rows)} examples.\n")

        return formatted_examples

    def gen_silver_set(self):
        """
        Generates silver standard (synthetic) ground truth alert analysis 
        samples for the alerts in the test data file defined in main_config.py. 
        Uses the gold standard (handcrafted) examples marked with 'gold' in 
        this file as few-shot exampels. 
        """

        # Load test alerts and create gold standard examples string
        df = pd.read_excel(
            test_alerts_path,
            sheet_name=conf.TEST_ALERTS_SHEET_NAME,
            dtype={'Alert_id': str})
        try:
            gold_examples_text = self._get_gold_examples(df)
        except ValueError as e:
            print(f"ERROR: {e} - Stopped data generation.")
            return

        # Load prompt template
        with open(silver_prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        # Write results into column for model name
        target_column = f"Ground Truth ({conf.SILVER_GEN_MODEL_NAME})"
        if target_column not in df.columns:
            df[target_column] = None
            df[target_column] = df[target_column].astype('object')
        print(f"Results will be written into column: '{target_column}'")

        # Start generating
        print(
            f"Starting synthetic ground truth data generation with {conf.SILVER_GEN_MODEL_NAME}...\n")
        for index, row in df.iterrows():
            alert_id = str(row['Alert_id']).strip().replace('.0', '')
            alert_role = row['Ground Truth Category'].lower()
            if alert_role == 'gold' or int(alert_id) < conf.FIRST_ALERT_ID or int(alert_id) > conf.LAST_ALERT_ID:
                continue
            print(
                f"[{index+1}/{len(df)}] Generating for Alert ID {alert_id}...")

            try:
                silver_prompt = prompt_template.format(
                    gold_examples_text=gold_examples_text,
                    alert=row['Alert']
                )

                response = self._send_request_openai_api(
                    silver_prompt
                )
                df.at[index, target_column] = response
                # time.sleep(2)  # In case of API rate limits.
                # See https://platform.openai.com/settings/organization/limits
            except Exception as e:
                print(f"SKIPPED due to ERROR: {e}")
                df.at[index, target_column] = f"ERROR: {e}"

        df.to_excel(test_alerts_path, index=False,
                    sheet_name=conf.TEST_ALERTS_SHEET_NAME)
        print(f"Finished! Updated file: {test_alerts_path}")


if __name__ == "__main__":
    # Run with 'python -m scripts.generate_silver_set'
    # Adjust relevant Alert IDs in src/config/main_config
    gen = SilverSetGenerator()
    gen.gen_silver_set()
