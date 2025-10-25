import json
from openai import OpenAI, AzureOpenAI, NOT_GIVEN
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed, wait_exponential, wait_random
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import time

from atlas_rag.llm_generator.prompt.rag_prompt import cot_system_instruction, cot_system_instruction_kg, cot_system_instruction_no_doc, prompt_template
from atlas_rag.llm_generator.format.validate_json_output import validate_filter_output, messages as filter_messages
from atlas_rag.llm_generator.prompt.lkg_prompt import ner_prompt, validate_keyword_output, keyword_filtering_prompt
from atlas_rag.retriever.base import BaseEdgeRetriever, BasePassageRetriever
from atlas_rag.llm_generator.format.validate_json_output import fix_and_validate_response

from transformers.pipelines import Pipeline
import jsonschema
from typing import Union
from logging import Logger

stage_to_prompt_type = {
    1: "entity_relation",
    2: "event_entity",
    3: "event_relation",
}
retry_decorator = retry(
    stop=(stop_after_delay(120) | stop_after_attempt(5)),
    wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(min=0, max=2),
)

class LLMGenerator:
    def __init__(self, client, model_name):
        self.model_name = model_name
        self.client: OpenAI | Pipeline = client
        if isinstance(client, (OpenAI, AzureOpenAI)):
            self.inference_type = "openai"
        elif isinstance(client, Pipeline):
            self.inference_type = "pipeline"
        else:
            raise ValueError("Unsupported client type6Please provide either an OpenAI client or a Huggingface Pipeline Object.")

    @retry_decorator
    def _generate_response(self, messages, do_sample=True, max_new_tokens=8192, temperature=0.7,
                          frequency_penalty=None, response_format={"type": "text"}, return_text_only=True,
                          return_thinking=False, reasoning_effort=None):
        if temperature == 0.0:
            do_sample = False
        if self.inference_type == "openai":
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                frequency_penalty=NOT_GIVEN if frequency_penalty is None else frequency_penalty,
                response_format=response_format if response_format is not None else {"type": "text"},
                timeout=120,
                reasoning_effort=NOT_GIVEN if reasoning_effort is None else reasoning_effort,
            )
            time_cost = time.time() - start_time
            content = response.choices[0].message.content
            if content is None and hasattr(response.choices[0].message, 'reasoning_content'):
                content = response.choices[0].message.reasoning_content
            else:
                content = response.choices[0].message.content
            if '</think>' in content and not return_thinking:
                content = content.split('</think>')[-1].strip()
            else:
                if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content is not None:
                    content = '<think>' + response.choices[0].message.reasoning_content + '</think>' + content
            if return_text_only:
                return content
            else:
                completion_usage_dict = response.usage.model_dump()
                completion_usage_dict['time'] = time_cost
                return content, completion_usage_dict
        elif self.inference_type == "pipeline":
            start_time = time.time()
            if hasattr(self.client, 'tokenizer'):
                input_text = self.client.tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                input_text = "\n".join([msg["content"] for msg in messages])
            response = self.client(
                input_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            time_cost = time.time() - start_time
            content = response[0]['generated_text'].strip()
            if '</think>' in content and not return_thinking:
                content = content.split('</think>')[-1].strip()
            if return_text_only:
                return content
            else:
                token_count = len(content.split())
                completion_usage_dict = {
                    'completion_tokens': token_count,
                    'time': time_cost
                }
                return content, completion_usage_dict

    def _generate_batch_responses(self, batch_messages, do_sample=True, max_new_tokens=8192,
                                 temperature=0.7, frequency_penalty=None, response_format={"type": "text"},
                                 return_text_only=True, return_thinking=False, reasoning_effort=None):
        if self.inference_type == "openai":
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(
                        self._generate_response, messages, do_sample, max_new_tokens, temperature,
                        frequency_penalty, response_format, return_text_only, return_thinking, reasoning_effort
                    ) for messages in batch_messages
                ]
                results = [future.result() for future in futures]
            return results
        elif self.inference_type == "pipeline":
            if not hasattr(self.client, 'tokenizer'):
                raise ValueError("Pipeline must have a tokenizer for batch processing.")
            batch_inputs = [self.client.tokenizer.apply_chat_template(messages, tokenize=False) for messages in batch_messages]
            start_time = time.time()
            responses = self.client(
                batch_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            time_cost = time.time() - start_time
            contents = [resp['generated_text'].strip() for resp in responses]
            if not return_thinking:
                contents = [content.split('</think>')[-1].strip() if '</think>' in content else content for content in contents]
            if return_text_only:
                return contents
            else:
                usage_dicts = [{
                    'completion_tokens': len(content.split()),
                    'time': time_cost / len(batch_messages)
                } for content in contents]
                return list(zip(contents, usage_dicts))

    def generate_cot(self, questions, max_new_tokens=1024):
        if isinstance(questions, str):
            messages = [{"role": "system", "content": "".join(cot_system_instruction_no_doc)},
                        {"role": "user", "content": questions}]
            return self._generate_response(messages, max_new_tokens=max_new_tokens)
        elif isinstance(questions, list):
            batch_messages = [[{"role": "system", "content": "".join(cot_system_instruction_no_doc)},
                              {"role": "user", "content": q}] for q in questions]
            return self._generate_batch_responses(batch_messages, max_new_tokens=max_new_tokens)

    def generate_with_context(self, question, context, max_new_tokens=1024, temperature=0.7):
        if isinstance(question, str):
            messages = [{"role": "system", "content": "".join(cot_system_instruction)},
                        {"role": "user", "content": f"{context}\n\n{question}\nThought:"}]
            return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature=temperature)
        elif isinstance(question, list):
            batch_messages = [[{"role": "system", "content": "".join(cot_system_instruction)},
                              {"role": "user", "content": f"{context}\n\n{q}\nThought:"}] for q in question]
            return self._generate_batch_responses(batch_messages, max_new_tokens=max_new_tokens, temperature=temperature)

    def generate_with_context_one_shot(self, question, context, max_new_tokens=4096, temperature=0.7):
        if isinstance(question, str):
            messages = deepcopy(prompt_template)
            messages.append({"role": "user", "content": f"{context}\n\nQuestions:{question}\nThought:"})
            return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature=temperature)
        elif isinstance(question, list):
            batch_messages = [deepcopy(prompt_template) + [{"role": "user", "content": f"{context}\n\nQuestions:{q}\nThought:"}]
                             for q in question]
            return self._generate_batch_responses(batch_messages, max_new_tokens=max_new_tokens, temperature=temperature)

    def generate_with_context_kg(self, question, context, max_new_tokens=1024, temperature=0.7):
        if isinstance(question, str):
            messages = [{"role": "system", "content": "".join(cot_system_instruction_kg)},
                        {"role": "user", "content": f"{context}\n\n{question}"}]
            return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature=temperature)
        elif isinstance(question, list):
            batch_messages = [[{"role": "system", "content": "".join(cot_system_instruction_kg)},
                              {"role": "user", "content": f"{context}\n\n{q}"}] for q in question]
            return self._generate_batch_responses(batch_messages, max_new_tokens=max_new_tokens, temperature=temperature)

    @retry_decorator
    def filter_triples_with_entity(self, question, nodes, max_new_tokens=1024):
        if isinstance(question, str):
            messages = [{"role": "system", "content": """
            Your task is to filter text candidates based on their relevance to a given query...
            """}, {"role": "user", "content": f"{question} \n Output Before Filter: {nodes} \n Output After Filter:"}]
            try:
                response = json.loads(self._generate_response(messages, max_new_tokens=max_new_tokens))
                return response
            except Exception:
                return json.loads(nodes)
        elif isinstance(question, list):
            batch_messages = [[{"role": "system", "content": """
            Your task is to filter text candidates based on their relevance to a given query...
            """}, {"role": "user", "content": f"{q} \n Output Before Filter: {nodes} \n Output After Filter:"}]
                             for q in question]
            responses = self._generate_batch_responses(batch_messages, max_new_tokens=max_new_tokens)
            return [json.loads(resp) if json.loads(resp) else json.loads(nodes) for resp in responses]

    @retry_decorator
    def filter_triples_with_entity_event(self, question, triples):
        if isinstance(question, str):
            messages = deepcopy(filter_messages)
            messages.append({"role": "user", "content": f"[ ## question ## ]]\n{question}\n[[ ## fact_before_filter ## ]]\n{triples}"})
            try:
                response = self._generate_response(messages, max_new_tokens=4096, temperature=0.0, response_format={"type": "json_object"})
                cleaned_data = validate_filter_output(response)
                return cleaned_data['fact']
            except Exception:
                return []
        elif isinstance(question, list):
            batch_messages = [deepcopy(filter_messages) + [{"role": "user", "content": f"[ ## question ## ]]\n{q}\n[[ ## fact_before_filter ## ]]\n{triples}"}]
                             for q in question]
            responses = self._generate_batch_responses(batch_messages, max_new_tokens=4096, temperature=0.0, response_format={"type": "json_object"})
            return [validate_filter_output(resp)['fact'] if validate_filter_output(resp) else [] for resp in responses]

    def generate_with_custom_messages(self, custom_messages, do_sample=True, max_new_tokens=1024, temperature=0.8, frequency_penalty=None):
        if isinstance(custom_messages[0], dict):
            return self._generate_response(custom_messages, do_sample, max_new_tokens, temperature, frequency_penalty)
        elif isinstance(custom_messages[0], list):
            return self._generate_batch_responses(custom_messages, do_sample, max_new_tokens, temperature, frequency_penalty)

    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_filter_keywords_with_entity(self, question, keywords):
        if isinstance(question, str):
            messages = deepcopy(keyword_filtering_prompt)
            messages.append({"role": "user", "content": f"[[ ## question ## ]]\n{question}\n[[ ## keywords_before_filter ## ]]\n{keywords}"})
            try:
                response = self._generate_response(messages, response_format={"type": "json_object"}, temperature=0.0, max_new_tokens=2048)
                cleaned_data = validate_keyword_output(response)
                return cleaned_data['keywords']
            except Exception:
                return keywords
        elif isinstance(question, list):
            batch_messages = [deepcopy(keyword_filtering_prompt) + [{"role": "user", "content": f"[[ ## question ## ]]\n{q}\n[[ ## keywords_before_filter ## ]]\n{k}"}]
                             for q, k in zip(question, keywords)]
            responses = self._generate_batch_responses(batch_messages, response_format={"type": "json_object"}, temperature=0.0, max_new_tokens=2048)
            return [validate_keyword_output(resp)['keywords'] if validate_keyword_output(resp) else keywords for resp in responses]

    def ner(self, text):
        if isinstance(text, str):
            messages = [{"role": "system", "content": "Please extract the entities..."},
                        {"role": "user", "content": f"Extract the named entities from: {text}"}]
            return self._generate_response(messages)
        elif isinstance(text, list):
            batch_messages = [[{"role": "system", "content": "Please extract the entities..."},
                              {"role": "user", "content": f"Extract the named entities from: {t}"}] for t in text]
            return self._generate_batch_responses(batch_messages)

    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_ner(self, text):
        if isinstance(text, str):
            messages = deepcopy(ner_prompt)
            messages.append({"role": "user", "content": f"[[ ## question ## ]]\n{text}"})
            try:
                response = self._generate_response(messages, max_new_tokens=4096, temperature=0.7, frequency_penalty=1.1, response_format={"type": "json_object"})
                cleaned_data = validate_keyword_output(response)
                return cleaned_data['keywords']
            except Exception:
                return []
        elif isinstance(text, list):
            batch_messages = [deepcopy(ner_prompt) + [{"role": "user", "content": f"[[ ## question ## ]]\n{t}"}] for t in text]
            responses = self._generate_batch_responses(batch_messages, max_new_tokens=4096, temperature=0.7, frequency_penalty=1.1, response_format={"type": "json_object"})
            return [validate_keyword_output(resp)['keywords'] if validate_keyword_output(resp) else [] for resp in responses]

    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_tog_ner(self, text):
        if isinstance(text, str):
            messages = [{"role": "system", "content": "You are an advanced AI assistant..."},
                        {"role": "user", "content": f"Extract the named entities from: {text}"}]
            try:
                response = self._generate_response(messages, max_new_tokens=4096, temperature=0.7, frequency_penalty=1.1, response_format={"type": "json_object"})
                cleaned_data = validate_keyword_output(response)
                return cleaned_data['keywords']
            except Exception:
                return []
        elif isinstance(text, list):
            batch_messages = [[{"role": "system", "content": "You are an advanced AI assistant..."},
                              {"role": "user", "content": f"Extract the named entities from: {t}"}] for t in text]
            responses = self._generate_batch_responses(batch_messages, max_new_tokens=4096, temperature=0.7, frequency_penalty=1.1, response_format={"type": "json_object"})
            return [validate_keyword_output(resp)['keywords'] if validate_keyword_output(resp) else [] for resp in responses]

    def generate_with_react(self, question, context=None, max_new_tokens=1024, search_history=None, logger=None):
        # Implementation remains single-input focused as it’s iterative; batching not applicable here
        react_system_instruction = (
            'You are an advanced AI assistant that uses the ReAct framework...'
        )
        full_context = []
        if search_history:
            for i, (thought, action, observation) in enumerate(search_history):
                full_context.append(f"\nPrevious search attempt {i}:\n{action}\n  Result: {observation}\n")
        if context:
            full_context.append(f"Current Retrieved Context:\n{context}\n")
        messages = [{"role": "system", "content": react_system_instruction},
                    {"role": "user", "content": f"Search History:\n\n{''.join(full_context)}\n\nQuestion: {question}"
                    if full_context else f"Question: {question}"}]
        return self._generate_response(messages, max_new_tokens=max_new_tokens)

    def generate_with_rag_react(self, question: str, retriever: Union['BaseEdgeRetriever', 'BasePassageRetriever'],
                               max_iterations: int = 5, max_new_tokens: int = 1024, logger: Logger = None):
        # Single-input iterative process; batching not applicable
        search_history = []
        if isinstance(retriever, BaseEdgeRetriever):
            initial_context, _ = retriever.retrieve(question, topN=5)
            current_context = ". ".join(initial_context)
        elif isinstance(retriever, BasePassageRetriever):
            initial_context, _ = retriever.retrieve(question, topN=5)
            current_context = "\n".join(initial_context)
        for iteration in range(max_iterations):
            analysis_response = self.generate_with_react(
                question=question, context=current_context, max_new_tokens=max_new_tokens, search_history=search_history, logger=logger
            )
            try:
                thought = analysis_response.split("Thought:")[1].split("\n")[0]
                action = analysis_response.split("Action:")[1].split("\n")[0]
                answer = analysis_response.split("Answer:")[1].strip()
                if answer.lower() != "need more information":
                    search_history.append((thought, action, "Using current context"))
                    return answer, search_history
                if "search" in action.lower():
                    search_query = action.split("search for")[-1].strip()
                    if isinstance(retriever, BaseEdgeRetriever):
                        new_context, _ = retriever.retrieve(search_query, topN=3)
                        current_contexts = current_context.split(". ")
                        new_context = [ctx for ctx in new_context if ctx not in current_contexts]
                        new_context = ". ".join(new_context)
                    elif isinstance(retriever, BasePassageRetriever):
                        new_context, _ = retriever.retrieve(search_query, topN=3)
                        current_contexts = current_context.split("\n")
                        new_context = [ctx for ctx in new_context if ctx not in current_contexts]
                        new_context = "\n".join(new_context)
                    observation = f"Found information: {new_context}" if new_context else "No new information found..."
                    search_history.append((thought, action, observation))
                    if new_context:
                        current_context = f"{current_context}\n{new_context}"
                else:
                    search_history.append((thought, action, "No action taken but answer not found"))
                    return "Unable to find answer", search_history
            except Exception as e:
                return analysis_response, search_history
        return answer, search_history

    def triple_extraction(self, messages, max_tokens=4096, stage=None, record=False):
        if isinstance(messages[0], dict):
            messages = [messages]
        responses = self._generate_batch_responses(
            batch_messages=messages,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
            frequency_penalty=0.5,
            reasoning_effort="none",
            return_text_only=not record
        )
        processed_responses = []
        for response in responses:
            if record:
                content, usage_dict = response
            else:
                content = response
                usage_dict = None
            try:
                prompt_type = stage_to_prompt_type.get(stage, None)
                if prompt_type:
                    corrected, error = fix_and_validate_response(content, prompt_type)
                    if error:
                        raise ValueError(f"Validation failed for prompt_type '{prompt_type}'")
                else:
                    corrected = content
                if corrected and corrected.strip():
                    if record:
                        processed_responses.append((corrected, usage_dict))
                    else:
                        processed_responses.append(corrected)
                else:
                    raise ValueError("Invalid response")
            except Exception as e:
                print(f"Failed to process response: {str(e)}")
                if record:
                    usage_dict = {'completion_tokens': 0, 'total_tokens': 0, 'time': 0}
                    processed_responses.append(("[]", usage_dict))
                else:
                    processed_responses.append("[]")
        return processed_responses