import os
import json
import evaluate
import statistics
import re
import time
import pandas as pd
import re
from dotenv import load_dotenv
from pathlib import Path
from src.utils import data_handler
from src.config import main_config as conf
# Google SDK Docs: https://googleapis.github.io/python-genai/
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig


class Evaluator:
    """Evaluates LLM-generated SOC alert analyses using ground truth references.

    This class is responsible for evaluation batch processing by reading 
    model-generated outputs and validation spreadsheets, checking data 
    integrity and linking execution logs to human/synthetic references.

    Attributes:
        judge_prompt_template (str): Template used for generating the instructionfor the LLM judge.
        first_alert_id (int): First alert id to consider in evaluation range.
        last_alert_id (int): Last alert id to consider in evaluation range.
        grouped_analyses (dict[str, list[str]]): Mapping between an alert ID and
            a list of its repetition analysis.
        reference_data (dict[str, dict[str, str]]): Mapping between an alert ID and a sub-dictionary of the alert and its ground truth.
    """

    def __init__(self,
                 analysis_filepath: Path,
                 ground_truth_filepath: Path,
                 judge_prompt_filepath: Path,
                 first_alert_id: int,
                 last_alert_id: int
                 ):
        """Initializes the Evaluator through loading, validation, and preparation of batch data.

        Args:
            analysis_filepath: Location of the JSON log file containing the reults of the candidate models.
            ground_truth_filepath: Location of the Excel file containing reference sets.
            judge_prompt_filepath: Location of the file judge system prompt.
            first_alert_id: First alert id to consider in evaluation range.
            last_alert_id: Last alert id to consider in evaluation range.

        Raises:
            ValueError: When structural mismatch is identified in the number of unique analyzed alerts and their corresponding references.  
        """
        self.judge_prompt_template = data_handler.read_txt(
            judge_prompt_filepath)
        self.first_alert_id = first_alert_id
        self.last_alert_id = last_alert_id

        # Get analysis results grouped in lists per alert_id
        self.grouped_analyses = self._get_grouped_analyses(analysis_filepath)

        # Get ground truth and alert texts per alert_id
        self.reference_data = self._get_reference_data(ground_truth_filepath)

        # Compare the number of analysis and ground truth texts
        num_analysis = len(self.grouped_analyses)
        num_references = len(self.reference_data)
        if num_analysis != num_references:
            raise ValueError(
                f"Mismatch in number of items: "
                f"{num_analysis} analysed alerts vs "
                f"{num_references} ground truth references and alerts."
            )

        print(
            f"Successfully loaded {num_analysis} analysis/reference data pairs",
            f"for alert IDs {self.first_alert_id} to {self.last_alert_id}.")

    def _get_grouped_analyses(self, analysis_filepath):
        """Performs ingestion and restructuring of the execution logs of candidate models.

        Breaks down individual iterations of the model into separate clusters 
        with respect to their corresponding alert IDs.

        Args:
            analysis_filepath: Inference logging JSON file path.

        Returns:
            dict[str, list[str]]: Mapping of alert IDs to their corresponding list of stripped repetition texts.

        Raises:
            ValueError: When there is no analyzed data that satisfies the current ID requirements.

        Example:
        {
            '1': ["...", "...", ...],
            '2': ["...", "...", ...],
            ...
        }
        """
        # Get analysis texts per alert_id
        analysis_data = data_handler.load_json(analysis_filepath)
        process_results_list = analysis_data.get('process_results', [])
        grouped_analyses = {}

        for alert_analyses in process_results_list:
            alert_id = alert_analyses["alert_id"]
            if int(alert_id) < self.first_alert_id:
                continue
            if int(alert_id) > self.last_alert_id:
                continue
            grouped_analyses[alert_id] = [
                rep.get('analysis', '').strip() for rep in
                alert_analyses.get('repetitions', [])
            ]

        if not grouped_analyses:
            raise ValueError("No analysis results found to evaluate.")

        return grouped_analyses

    def _get_reference_data(self, ground_truth_filepath):
        """Parses the ground truth reference sheet and matches the alert 
        content with its baseline.

        Gets the necessary reference data for the evaluation by returning a
        dictionary containing the alert IDs as keys and dictionaries with
        the corresponding alert and ground truth reference as values.
        Filters by self.first_alert_id and self.last_alert_id.

        Args:
            ground_truth_filepath: Path to reference Excel workbook.

        Returns:
            dict[str, dict[str, str]]: A dictionary where the keys are normalized alert ids
                and the values are dictionaries with 'alert' and 'ground_truth'.

        Example:
        {
            '1': {
                'alert': "..."
                'ground_truth': "..."
            },
            '2': {
                'alert': "..."
                'ground_truth': "..."
            },
            ...
        }
        """

        df = pd.read_excel(
            ground_truth_filepath,
            sheet_name=conf.TEST_ALERTS_SHEET_NAME,
            dtype={'Alert_id': str})

        reference_data = {}
        for _, row in df.iterrows():
            if row['Ground Truth Category'].lower() == 'gold':
                ground_truth_text = row['Ground Truth (Handcrafted)']
            else:
                ground_truth_text = row['Ground Truth (gpt-5.2-2025-12-11)']
            alert_id = alert_id = str(
                row['Alert_id']).strip().replace('.0', '')
            if int(alert_id) < self.first_alert_id:
                continue
            if int(alert_id) > self.last_alert_id:
                continue
            reference_data[alert_id] = {
                'alert': row['Alert'],
                'ground_truth': ground_truth_text
            }

        return reference_data

    def calculate_bleurt_scores(self) -> dict:
        """Calculates BLEURT scores for all pairs of predictions and references.

        Computes semantic similarity scores for all the repeated instances of model output candidates and their respective ground truth baselines.

        Calculates BLEURT scores for all prediction/reference pairs. Scores
        range is not fixed but typically ranges from about -1.5 to +1.5.
        Meanings:
            - score >= 1: Almost perfect semantic similarity
                       (nearly identical meaning)
            - score = 0: Random / Neutral / No similarity.
            - score < 0: The texts contradict each other.

        Returns:
            dict: A summary of statistical metrics that include mean and std dev.

        Raises:
            ConnectionError: The HuggingFace evaluation service could not load the embedding checkpoint.
        """
        # Create a flat predictions/analysis list
        predictions = [
            analysis
            for analysis_list in self.grouped_analyses.values()
            for analysis in analysis_list
        ]

        # Create a predictions and a reference list matching the repititions
        references = []
        predictions = []
        for key, content in self.reference_data.items():
            if 'ground_truth' in content:
                ground_truth_value = content['ground_truth']
                alert_predictions = self.grouped_analyses.get(key, [])
                num_predictions = len(alert_predictions)
                references.extend(
                    [ground_truth_value] * num_predictions)
                predictions.extend(alert_predictions)

        assert len(references) == len(
            predictions), "Mismatch between reference and prediction counts!"

        # Initialize BLEURT model
        try:
            # Smaller embedding model
            embedding_model = "BLEURT-20"
            # Bigger embedding model
            # embedding_model = "bleurt-large-512"
            bleurt = evaluate.load(
                "bleurt", module_type="metric", checkpoint=embedding_model)
            print(f"Initialized BLEURT embedding model {embedding_model}")
        except Exception as e:
            raise ConnectionError(
                f"Could not load BLEURT model. Original error: {e}")

        # Calculate BLEURT scores
        print("--- BLEURT Evaluation Started ---")
        bleurt_results = bleurt.compute(
            predictions=predictions, references=references)

        # Create a statistical report with mean and stdev values
        bleurt_report = self._create_overall_bleurt_report(
            bleurt_results, embedding_model)
        total_mean = bleurt_report['between_alert_statistics']['total_mean']
        print(f"BLEURT Mean Score: {total_mean}\n\n")
        print("--- BLEURT Evaluation Completed ---")
        return bleurt_report

    def _clean_text_for_bleurt(self, text):
        """Cleans texts to enable a fair semantic comparison.

        Removes escape characters of the platform, markdown formatting, 
        numbering, and targeted headers. Normalization removes everything 
        except the actual text content to prevent style differences from 
        negatively impacting semantic similarity scores.

        Args:
            text: Input string payload that is being cleaned.

        Returns:
            str: Single spaced and cleaned up text.
        """
        if not isinstance(text, str):
            return ""

        # Replace by escapes:  \\n, \n, \r
        text = text.replace("\\n", " ").replace("\n", " ").replace("\r", "")

        # Remove markdown formating:  **text**, __text__, `text`
        text = text.replace("**", "").replace("__", "").replace("`", "")

        # Replaca section headers by spaces
        headers = [
            r'Description\s*[:\-]*',
            r'Assessment\s*[:\-]*',
            r'Recommendation\s*[:\-]*',
            r'Risk level\s*[:\-]*',
            r'Potential implications\s*[:\-]*'
        ]
        for header in headers:
            text = re.sub(header, ' ', text, flags=re.IGNORECASE)

        # Remove numberings and bullet points
        text = re.sub(r'\d+\.\s+', '', text)
        text = re.sub(r'-\s+', '', text)  # Bulletpoints weg

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Remove leading and tailing whitespaces
        return text.strip()

    def _clean_text_for_judge(text):
        """Cleans text to avoid formatting bias in LLM judge assessment.

        Removes markdown emphasis symbols and normalizes excessive 
        whitespace spacing for a fair text-only assessment.

        Args:
            text: The initial raw string to be sanitized.

        Returns:
            str: The cleaned string text without the markdown symbols.
        """
        if not isinstance(text, str):
            return ""

        # Remove markdown formating: **text**, __text__, `text`
        text = text.replace("**", "").replace("__", "").replace("`", "")

        # Max. allow 2 newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _calculate_single_alert_bleurt_statistics(self, alert_id, alert_scores):
        """Calculates statistical summary metrics of a particular alert.

        Calculates the mean and standard deviation of all repetition scores for 
        a particular alert identifier.

        Args:
            alert_id (string): Unique id of the alert that is being evaluated.
            alert_scores (list of float): List of BLEURT scores corresponding
                to each repetition of the alert.

        Returns:
            dict: Dictionary containing rounded metrics of the alert:
                - "alert_id": Key to the evaluation.
                - "total_alert_mean" (float): Mean score rounded to two decimal places.
                - "total_alert_stdev" (float): Standard deviation rounded to two
                    decimal places or 0 if there is only one repetition.
                - "single_scores" (list): Unrounded input scores.
        """
        mean_score = statistics.mean(alert_scores)
        if len(alert_scores) > 1:
            stdev_score = statistics.stdev(alert_scores)
        else:
            stdev_score = 0.0

        return {
            "alert_id": alert_id,
            "total_alert_mean": round(mean_score, 2),
            "total_alert_stdev": round(stdev_score, 2),
            "single_scores": alert_scores
        }

    def _create_overall_bleurt_report(self, bleurt_results, embedding_model):
        """Flattened BLEURT results are summarized into an hierarchically 
        organized statistical report.

        Flattened sequence of scores is mapped to the groups corresponding to 
        their alert repetitions. Calculates within-group summary values (per-alert statistics) as well as global system performance baseline indicators (between-alert statistics).

        Args:
            bleurt_results: Raw dictionary output of the Hugging Face 
                evaluate library containing a "scores" key with a list of floats.
            embedding_model: String specifying the ID of the BLEURT checkpoint.

        Returns:
            dict: Hierarchical report consisting of metadata and aggregated sets:
                - "evaluation_metadata": Dictionary with the name of the model.
                - "between_alert_statistics": Mean and standard deviation of 
                the alerts' average scores.
                - "per_alert_statistics": List of dictionaries with the output of '_calculate_single_alert_bleurt_statistics'.
        """

        # Extract scores
        scores = bleurt_results.get("scores", [])
        all_alerts_judgements = []
        alert_means = []
        current_score_idx = 0

        # Calculate per-alert statistics
        for alert_id, analyses in self.grouped_analyses.items():
            num_predictions = len(analyses)
            alert_scores = scores[current_score_idx:
                                  current_score_idx + num_predictions]
            current_score_idx += num_predictions
            stats = self._calculate_single_alert_bleurt_statistics(
                alert_id, alert_scores)

            all_alerts_judgements.append(stats)
            if alert_scores:
                alert_means.append(statistics.mean(alert_scores))

        # Calculate between alert statistics
        total_mean = statistics.mean(alert_means) if alert_means else 0.0
        total_stdev = statistics.stdev(
            alert_means) if len(alert_means) > 1 else 0.0

        return {
            "evaluation_metadata": {
                "bleurt_embedding_model": embedding_model,
            },
            "between_alert_statistics": {
                "total_mean": round(total_mean, 2),
                "total_stdev": round(total_stdev, 2)
            },
            "per_alert_statistics": all_alerts_judgements
        }

    def run_llm_as_judge(self, rep_num) -> dict:
        """Judges repeated alert analyses using Googles Gemini LLM and provides
        overall and per-alert statistics.

        Coordinates the entire LLM-as-a-judge evaluation process by creating a
        Gemini client, looping through all group repetitions, triggering 
        inference, and computing both per-alert and global statistics.

        Args:
            rep_num: An integer indicating the index of a specific repetition to be evaluated, or None for evaluating all possible repetitions.

        Returns:
            dict: A full consolidated report including metadata for evaluation, global alert statistics, and detailed per-alert logs.

        Raises:
            ValueError: When the GOOGLE_API_KEY environment variable is undefined.
        """
        print(f"--- Starting LLM-as-a-judge evaluation ---")

        # Initialize Gemini judge model using the Google API:
        # https://ai.google.dev/gemini-api/docs/models?hl=de#gemini-2.5-pro
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please check your .env file.")
        judge_client = genai.Client(api_key=api_key)
        judge_model_name = conf.JUDGE_MODEL_NAME
        generation_config = GenerateContentConfig(
            temperature=conf.JUDGE_TEMP,
            thinking_config=ThinkingConfig(
                include_thoughts=conf.JUDGE_THOUGHTS)
        )

        # Iterate through analysis repitition groups
        all_alerts_judge_reports = []
        for alert_id, analysis_texts in self.grouped_analyses.items():
            print("-"*50)
            print(f" - Judging alert analysis (ID: {alert_id}):")

            reference = self.reference_data.get(alert_id)
            alert = reference['alert']
            ground_truth = reference['ground_truth']
            if not ground_truth or not alert:
                print(f"WARNING: No ground truth or alert found for id",
                      " '{alert_id}'. Skipped.")
                continue

            alert_analysis_judgements = []
            for repitition_i, candidate_analysis in enumerate(analysis_texts):
                repetition_num = repitition_i + 1
                if rep_num and repetition_num != rep_num:
                    continue
                print(f" -> For analysis repetition ",
                      f"{repetition_num}/{len(analysis_texts)}...")
                if not candidate_analysis:
                    judgement = {
                        "description_quality": {
                            "score": 0,
                            "error_tags": ["OTHER"]
                        },
                        "assessment_quality": {
                            "score": 0,
                            "error_tags": ["OTHER"]
                        },
                        "recommendation_quality": {
                            "score": 0,
                            "error_tags": ["OTHER"]
                        },
                        "thinking_text": "[Skipped LLM Judgement]: Empty analysis string."
                    }
                else:
                    judgement = self._llm_judge_single_alert(
                        judge_client,
                        judge_model_name,
                        generation_config,
                        alert,
                        candidate_analysis,
                        ground_truth
                    )
                if judgement:
                    alert_analysis_judgements.append(
                        {"repetition_num": repetition_num} | judgement)
                else:
                    print("No judgement received.")

            if alert_analysis_judgements:
                single_judge_report = self._calculate_single_alert_judge_statistics(
                    alert_analysis_judgements, alert_id)
                all_alerts_judge_reports.append(single_judge_report)

        llm_judge_report = self._create_overall_judge_report(
            all_alerts_judge_reports)
        print("--- LLM judgement completed ---")
        return llm_judge_report

    def _llm_judge_single_alert(
        self,
        judge_client: genai.Client,
        judge_model_name: str,
        generation_config: dict,
        alert: str,
        candidate_analysis: str,
        ground_truth: str,
    ) -> dict | None:
        """Performs a single API call to the judge LLM for one alert. 

        Wraps the text attributes of the input into the evaluation prompt 
        template, sends the generation request to the Gemini API, and uses 
        exponential backoff strategy for recovery from temporary connectivity 
        errors or format issues.

        Args:
            judge_client: The initialized client object of Google GenAI API.
            judge_model_name: The name of the Gemini judge model.
            generation_config: The constraints for generation parameters.
            alert: The security alert text to be analyzed.
            candidate_analysis: The analysis text produced by the candidate model.
            ground_truth: The text to be used as the evaluation reference.

        Returns:
            dict | None: A dictionary containing the scores and error tags per
                criteria, the dictionary with default values on failure, or None
                if all retry attempts have been exhausted.
        """
        max_retries = 8
        backoff_factor = 2

        prompt = self.judge_prompt_template.format(
            alert=alert,
            ground_truth=ground_truth,
            candidate_analysis=candidate_analysis
        )

        for num_api_call in range(max_retries):
            response = None
            try:
                # Extract response and parse to dict
                response = judge_client.models.generate_content(
                    model=judge_model_name,
                    contents=prompt,
                    config=generation_config
                )
                judgement_json = self._clean_and_parse_judge_response(
                    response)
                if judgement_json == {}:
                    raise ValueError("Judge produced invalid/no JSON.")
                return judgement_json
            except Exception as e:
                print(
                    f"WARNING: API call attempt {num_api_call + 1}",
                    f"/{max_retries} failed. Error: {e}")
                if response:
                    print(
                        f"DEBUG - Raw API response was: ",
                        f"'{repr(response.text)}'")
                    print(
                        "Skipping due to failed judgement. Expecting repetition loop in analysis.")
                    return {
                        "description_quality": {
                            "score": 0,
                            "error_tags": ["OTHER"]
                        },
                        "assessment_quality": {
                            "score": 0,
                            "error_tags": ["OTHER"]
                        },
                        "recommendation_quality": {
                            "score": 0,
                            "error_tags": ["OTHER"]
                        },
                        "thinking_text": "[LLM Judgement FAILED]: Invalid response format.",
                        "error": f"{e}",
                        "judge_response": f"'{repr(response.text)}'"
                    }
                if num_api_call < max_retries - 1:
                    wait_time = backoff_factor ** (num_api_call + 1)
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"-> All {max_retries} attempts failed. Skipping ",
                          "this repetition.")
                    return None

    def _clean_and_parse_judge_response(self, judge_response: str) -> dict:
        """Extracts and parses JSON sub-object targets from the Gemini response.

        Separates criteria quality scores using regular expressions, converts
        successful regex matches to Python dictionaries, and retrieves the 
        native thinking/model logs if enabled.

        Args:
            judge_response: The raw response object returned from the GenAI SDK
                execution layer.

        Returns:
            dict: A cleaned dictionary that tracks criteria scores, error codes,
                and thinking logs.

        Raises:
            ValueError: In case of a missing judge response, missing structure 
            schema keys, or JSON parse failure of sub-blocks.
        """
        if not judge_response:
            raise ValueError("Mising Judge reponse.")

        keys = [
            "description_quality",
            "assessment_quality",
            "recommendation_quality"
        ]
        cleaned_reponse = {}

        # Extract judgements by categories
        for key in keys:
            pattern = f'"{key}"\s*:\s*({{.*?}})'
            match = re.search(pattern, judge_response.text, re.DOTALL)

            if match:
                judgement_for_key = match.group(1)
                try:
                    cleaned_reponse[key] = json.loads(judgement_for_key)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON for key '{key}'.",
                                     f" Found reponse: {judgement_for_key}.",
                                     f" Error: {e}")
            else:
                raise ValueError(
                    f"Couldn't match key '{key}' in the judge response.")

        # Extract reasoning/thinking text and add to dict
        if conf.JUDGE_THOUGHTS:
            thoughts = []
            if (judge_response.candidates and
                    judge_response.candidates[0].content.parts):
                for part in judge_response.candidates[0].content.parts:
                    if part.thought:
                        thoughts.append(part.text)
            thinking_text = "\n".join(thoughts)
            cleaned_reponse["thinking_text"] = thinking_text
        else:
            cleaned_reponse["thinking_text"] = ""

        return cleaned_reponse

    def _calculate_single_alert_judge_statistics(
        self,
        judgements: list,
        alert_id: str
    ) -> dict:
        """
        Calculates statistical mean and variance measurements for a single alert ID.

        Evaluates repetitions and computes averages and standard deviation samples for all individual evaluation categories (description, evaluation, recommendation).

         Parameters:
            judgements: List of dictionaries of individual repetition verification results.
            alert_id: Unique ID key identifying target alert.

        Returns:
            dict: Dictionary with individual standard measurements, average 
            values, and successful single execution lists.
        """
        # Filter out failed judgements (None)
        valid_judgements = [j for j in judgements if j is not None]
        if not valid_judgements:
            return {
                "alert_id": alert_id,
                "error": "No valid judgements for this alert."
            }

        # Get scores for the different judgement criteria
        scores_desciption = [j.get('description_quality', {}).get(
            'score') for j in valid_judgements if j.get('description_quality', {}).get('score') is not None]

        scores_assessment = [j.get('assessment_quality', {}).get(
            'score') for j in valid_judgements if j.get('assessment_quality', {}).get('score') is not None]

        scores_recommendation = [j.get('recommendation_quality', {}).get(
            'score') for j in valid_judgements if j.get('recommendation_quality', {}).get('score') is not None]

        # Calculate mean and stdev over the different judgement criteria
        mean_description_score = round(statistics.mean(
            scores_desciption), 2) if scores_desciption else 0
        stdev_description_score = round(statistics.stdev(
            scores_desciption), 2) if len(scores_desciption) > 1 else 0

        mean_assessment_score = round(statistics.mean(
            scores_assessment), 2) if scores_assessment else 0
        stdev_assessment_score = round(statistics.stdev(
            scores_assessment), 2) if len(scores_assessment) > 1 else 0

        mean_recommendation_score = round(statistics.mean(
            scores_recommendation), 2) if scores_recommendation else 0
        stdev_recommendation_score = round(statistics.stdev(
            scores_recommendation), 2) if len(scores_recommendation) > 1 else 0

        # Calculate total mean and stdev
        all_scores = scores_desciption + scores_assessment + \
            scores_recommendation
        total_mean_score = round(statistics.mean(
            all_scores), 2) if all_scores else 0
        total_stdev_score = round(statistics.stdev(
            all_scores), 2) if len(all_scores) > 1 else 0

        return {
            "alert_id": alert_id,
            "total_alert_mean": total_mean_score,
            "total_alert_stdev": total_stdev_score,
            "mean_description": mean_description_score,
            "stdev_description": stdev_description_score,
            "mean_assessment": mean_assessment_score,
            "stdev_assessment": stdev_assessment_score,
            "mean_recommendation": mean_recommendation_score,
            "stdev_recommendation": stdev_recommendation_score,
            "single_judgements": valid_judgements
        }

    def _create_overall_judge_report(
            self,
            all_alerts_judgements: list,
    ) -> dict:
        """Generates the global performance benchmarks based on local logs.

        Aggregates grouped evaluations statistics to calculate the final 
        macro-means and sample standard deviations for all valid target and 
        target analysis features.

        Args:
            all_alerts_judgements: A list of dictionaries representing alert 
            statistics and variances of criteria.

        Returns:
            dict: The final hierarchical dictionary that contains the 
            configuration metadata, global variance indices, and individual 
            evaluation logs.
        """
        valid_alerts_judgements = [
            j for j in all_alerts_judgements if 'error' not in j]
        if not valid_alerts_judgements:
            return {"error": "There are no judgements."}
        elif len(valid_alerts_judgements) == 1:
            return {
                "evaluation_metadata": {
                    "judge_model": conf.JUDGE_MODEL_NAME,
                    "judge_temperature": conf.JUDGE_TEMP,
                    "judge_include_thoughts": conf.JUDGE_THOUGHTS
                },
                "between_alert_statistics": {
                    "total_mean": 0,
                    "total_stdev": 0,
                    "mean_description": 0,
                    "stdev_description": 0,
                    "mean_assessment": 0,
                    "stdev_assessment": 0,
                    "mean_recommendation": 0,
                    "stdev_recommendation": 0
                },
                "per_alert_statistics": all_alerts_judgements
            }

        means_across_alerts = {
            "total": statistics.mean([j['total_alert_mean'] for j in valid_alerts_judgements]),
            "description": statistics.mean([j['mean_description'] for j in valid_alerts_judgements]),
            "assessment": statistics.mean([j['mean_assessment'] for j in valid_alerts_judgements]),
            "recommendation": statistics.mean([j['mean_recommendation'] for j in valid_alerts_judgements])
        }

        stdevs_between_alerts = {
            "total": statistics.stdev([j['total_alert_mean'] for j in valid_alerts_judgements]),
            "description": statistics.stdev([j['mean_description'] for j in valid_alerts_judgements]),
            "assessment": statistics.stdev([j['mean_assessment'] for j in valid_alerts_judgements]),
            "recommendation": statistics.stdev([j['mean_recommendation'] for j in valid_alerts_judgements])
        }

        return {
            "evaluation_metadata": {
                "judge_model": conf.JUDGE_MODEL_NAME,
                "judge_temperature": conf.JUDGE_TEMP,
                "judge_include_thoughts": conf.JUDGE_THOUGHTS
            },
            "between_alert_statistics": {
                "total_mean":
                round(means_across_alerts["total"], 2),
                "total_stdev":
                round(stdevs_between_alerts["total"], 2),
                "mean_description":
                round(means_across_alerts["description"], 2),
                "stdev_description":
                round(stdevs_between_alerts["description"], 2),
                "mean_assessment":
                round(means_across_alerts["assessment"], 2),
                "stdev_assessment":
                round(stdevs_between_alerts["assessment"], 2),
                "mean_recommendation":
                round(means_across_alerts["recommendation"], 2),
                "stdev_recommendation":
                round(stdevs_between_alerts["recommendation"], 2)
            },
            "per_alert_statistics": all_alerts_judgements
        }
