class PromptAssembler:
    """Assembles and structures raw prompt strings into proper instruction hierarchy payloads.

    This helper class encapsulates the process of formatting system and user 
    messages to the required list dictionary format for modern chat completion 
    APIs (e.g., OpenAI, Ollama).
    """

    def __init__(self, sys_prompt):
        """ Initializes the Assembler with a static system prompt.

        Args:
            sys_prompt: The initial system instruction or persona description 
                specifying the task and constraints for the LLM.
        """
        self.sys_prompt = sys_prompt

    def generate(self, user_prompt):
        """Constructs messages from the system and user prompts.

        Args:
            user_prompt: The input or payload (alert) to be processed by the model.

        Returns:
            list[dict[str, str]]: A list of message dictionaries, where each 
                dictionary includes 'role' and 'content' attributes, ready to be
                used directly by the LLM client component.
        """
        messages = [
            {
                'role': 'system',
                'content': self.sys_prompt
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ]
        return messages
