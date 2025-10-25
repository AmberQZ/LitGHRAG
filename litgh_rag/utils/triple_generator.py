from openai import OpenAI, AzureOpenAI
from transformers import Pipeline
from litgh_rag.utils.json_repair import fix_and_validate_response
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_fixed
max_retries = 10
retry_decorator = retry(
    stop=stop_after_attempt(max_retries),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((Exception)),
    reraise=True
)
stage_to_prompt_type = {
    1: "entity_relation",
    2: "event_entity",
    3: "event_relation",
}
@retry_decorator
def client_generation(model:OpenAI, model_name, messages, max_new_tokens=4096, temperature=0.1):
    response = model.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            max_tokens=max_new_tokens,
                            temperature=temperature,
                        )
                        # validate the response
    # try:
    #     validate_response(response.choices[0].message.content, prompt_type)
    # else:
    return response.choices[0].message.content, None
    # except Exception as e:
    #     raise e
    
class TripleGenerator():
    def __init__(self, client: Pipeline|OpenAI|AzureOpenAI, model_name, max_new_tokens=4096, temperature=0.1, frequency_penalty=1.1):
        self.client = client
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.model_name = model_name
        if isinstance(client, OpenAI) or isinstance(client, AzureOpenAI):
            self.inference_type = "openai"
        elif isinstance(client, Pipeline):
            self.inference_type = "pipeline"
        else:
            raise ValueError("Unsupported client type. Please provide either an OpenAI client or a Huggingface Pipeline Object.")

    def generate(self, messages, max_tokens=4096, stage=None):
        """
        Generate triples from the input messages using the specified model.
        Args:
            messages: List of messages in OpenAI chat format.
            prompt_type: Type of prompt to use for generation (e.g., "entity_relation", "event_entity", "event_relation").
        Returns:
            A list of valid, non-empty corrected JSON strings.
        """
        responses = []
        prompt_type = stage_to_prompt_type.get(stage, None)  # Default to "entity_relation" if stage is not provided
        if isinstance(messages[0], dict):
            messages = [messages]  # Ensure messages is a list for single message input
        try:
            if self.inference_type == "openai":
                for message in messages:
                    response, _ = client_generation(self.client, self.model_name, message, max_new_tokens=max_tokens)
                    if response and response.strip():  # Check for None and empty strings
                        responses.append(response.strip())

            elif self.inference_type == "pipeline":
                generated = self.client(messages, max_new_tokens=max_tokens, temperature=self.temperature, return_full_text=False)
                responses = [response[0]['generated_text'] for response in generated if response and response[0]['generated_text'].strip()]  # Filter non-empty texts

        except Exception as e:
            raise e

        # Process responses and collect only valid ones
        if prompt_type is None:
            return responses
        validated_responses = []
        for response in responses:
            if not response.strip():  # Double-check for empty strings
                continue
            corrected, error = fix_and_validate_response(response, prompt_type)
            if corrected is not None:  # Only include successfully corrected responses
                validated_responses.append(corrected)

        return validated_responses
        
