from src.utils.timer import Timer
from src.core.prompt_assembler import PromptAssembler
from src.core.llm_client import OllamaClient
import json


class SOCAlertAnalyst:
    """Handles the automated analysis of SOC alerts using an LLM.

    This class oversees the process of the alert evaluation, from prompt
    creation to LLM client interaction, repetition handling for statistical
    significance, and metrics aggregation.
    """

    def __init__(self, model_id, sys_prompt, num_repetitions: int = 1):
        """Initializes the analyst with target model and prompt settings.

        Args:
            model_id: ID of the LLM to be used as the target by Ollama.
            sys_prompt: System prompt that describes the persona/rule set of the analyst.
            num_repetitions: Number of times to repeat the evaluations for each alert.
        """
        self.model_id = model_id
        self.prompt_assembler = PromptAssembler(sys_prompt)
        self.llm_client = OllamaClient(self.model_id)
        self.num_repetitions = num_repetitions

    def analyze_alerts_batch(self, alerts: list) -> dict:
        """Analyzes a batch of alerts sequentially and aggregates metrics.

        Args:
            alerts: A list of dictionaries in which each entry is a
                security alert and consists of at least "alert_id" and "content".
        Returns:
            A dictionary with accumulated statistics about processing:
                - "process_status" (str): Status of the entire process (e.g., "Successful").
                - "process_duration" (float): Time spent in seconds.
                - "process_results" (list): A list of results for each particular alert.
        """
        if not alerts:
            return {
                "process_status": "No alerts provided.",
                "process_results": []
            }

        # Analyze the alerts one by one, after removing metadata
        results = []
        with Timer() as timer:
            for alert in alerts:
                # Remove alert metadata
                alert_id = alert.get("alert_id", "unknown_id")
                alert_payload = alert.get("content", "")
                alert_payload_str = json.dumps(alert_payload, indent=2)

                # Generate message list for LLM
                messages = self.prompt_assembler.generate(
                    user_prompt=alert_payload_str,
                )

                # Start analysis - optionally repeated for significance
                print("-"*50)
                print(f"Analyzing Alert (ID: {alert_id}):\n{alert_payload}\n")
                repetitions = []
                prompt_token_count = 0
                for rep_num in range(1, self.num_repetitions + 1):
                    request_info, prompt_tokens = self.analyze_alert(
                        rep_num,
                        messages
                    )
                    repetitions.append(request_info)
                    prompt_token_count = prompt_tokens
                    print(
                        f"Analysis ({prompt_tokens} tokens):\n{request_info}")

                # Create entry for this alert
                result = {
                    "alert_id": alert_id,
                    "messages": messages,
                    "prompt_token_count": prompt_token_count,
                    "repetitions": repetitions
                }
                results.append(result)

        # Return Process information of batch analysis
        processing_info = {
            "process_status": "Successful",
            "process_duration": timer.elapsed_seconds(),
            "process_results": results
        }
        return processing_info

    def analyze_alert(self, repetition_num: int, messages: str, ) -> dict:
        """Performs a single inference step against the LLM for a specific alert.

        Args:
            repetition_num: The iteration number of the experiment.
            messages: Conversation/prompt history of the LLM.

        Returns:
            A tuple with the following elements:
                - request_info (dict): Inference output including status metadata, analysis text, and reasoning tokens.
                - prompt_tokens (int): Number of prompt tokens used.
        """
        print(
            f" - Analysis repetition: {repetition_num}/{self.num_repetitions}...")
        try:
            # Send request to LLM
            response = self.llm_client.send_request(messages)

            # Parse ollama reponse information
            duration_ns = response.get("total_duration", 0)
            duration_s = f"{(duration_ns / 1_000_000_000):.2f}"
            prompt_tokens = response.get("prompt_eval_count", 0)
            reponse_tokens = response.get("eval_count", 0)
            analysis = response.get("message", {}).get("content", "")
            reasoning = response.get("message", {}).get("thinking") or ""

            request_info = {
                "repetition_num": repetition_num,
                "status": "Successful",
                "analysis": analysis,
                "reasoning": reasoning,
                "duration_seconds": duration_s,
                "response_token_count": reponse_tokens
            }

            print(f" -> Analyzed alert successfully in {duration_s}s")
            return request_info, prompt_tokens

        except Exception as e:
            error_message = f"Alert processing failed with Error: {e}"
            print("ERROR - " + error_message)
            request_info = {
                "repetition_num": repetition_num,
                "status": error_message,
                "analysis": "",
                "reasoning": "",
                "duration_seconds": "0.00",
                "response_token_count": 0
            }
            return request_info, 0
