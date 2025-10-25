import json
from openai import OpenAI, NOT_GIVEN
from tenacity import retry, wait_fixed, stop_after_delay, stop_after_attempt
from copy import deepcopy
from litgh_rag.retrieval.filter_template import messages as filter_messages, validate_filter_output
from litgh_rag.retrieval.prompt_template import prompt_template
from litgh_rag.billion.prompt_template import ner_prompt, validate_keyword_output, keyword_filtering_prompt
from transformers.pipelines import Pipeline
import jsonschema
from typing import Union
from logging import Logger
from litgh_rag.retrieval.retriever.base import BaseEdgeRetriever, BasePassageRetriever

# from https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/qa/qa_reader.py
# prompts from hipporag qa_reader
cot_system_instruction = ('As an advanced reading comprehension assistant, your task is to analyze passages and corresponding questions meticulously. If the information is not enough, you can use your own knowledge to answer the question.'
                          'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')
cot_system_instruction_no_doc = ('As an advanced reading comprehension assistant, your task is to analyze the questions and then answer them. '
                                 'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                 'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')
# This is the instruction for the KG-based QA task
cot_system_instruction_kg = ('As an advanced reading comprehension assistant, your task is to analyze extracted information and corresponding questions meticulously. If the knowledge graph information is not enough, you can use your own knowledge to answer the question. '
                                'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')

# 利用 reason path（知识图谱中的实体-关系-实体-关系路径）和 passages 的知识进行推理
# cot_system_instruction_reason_passage = (
#     'As an advanced reading comprehension assistant, your task is to answer questions by first reasoning with the provided reason paths (reasoning chains from the knowledge graph, formatted as entity-relation-entity-relation...). '
#     'If the reason paths alone are insufficient, refer to the provided passages for additional information. '
#     'If both are still not enough, you can use your own internal knowledge to answer the question. '
#     'Your response should start after "Thought: ", where you methodically break down the reasoning process, illustrating how you arrive at conclusions. '
#     'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.'
# )

cot_system_instruction_reason_passage = (
    'As an advanced reading comprehension assistant, your task is to answer questions by first reasoning with the provided reason paths (reasoning chains from the knowledge graph, formatted as entity-relation-entity-relation...). '
    'If the knowledge graph information is not enough, you can use your own knowledge to answer the question. '
    'Your response should start after "Thought: ", where you methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.'
)

system_instruction_select_t_e = (
    "Select the events and triplets that are most relevant to answering a given question./no_think "
    "Input:\n"
    "- Question: The question you want to answer.\n"
    "- Events: A list of textual events related to entities.\n"
    "- triplets: A list of (entity, relation, entity) triplets.\n"
    "Task:\n"
    "1. Identify only the events and triplets that provide evidence for answering the question.\n"
    "2. Maintain two separate lists: one for relevant events, one for relevant triplets.\n"
    "3. Keep the output concise and precise.\n"
    "4. Do not summarize, infer, or interpret—select only the supporting items.\n"
    "\nOutput Format:\n"
    "Relevant events: [ ... ]\n"
    "Relevant triplets: [ ... ]\n"
    "\nExample\nInput:\n"
    "Question: 'Which film was released first, Aas Ka Panchhi or Phoolwari?'\n"
    "Events: [\n"
    "    'Aas Ka Panchhi is a 1961 Hindi movie produced by J. Om Prakash.',\n"
    "    'It is written by Mohan Kumar and Rajinder Singh Bedi and directed by Mohan Kumar.',\n"
    "    'It was the third highest grossing Indian film of 1946.',\n"
    "    'Phoolwari is a 1946 Indian Bollywood film.'\n"
    "]\n"
    "triplets: [\n"
    "    ('Phoolwari', 'is', 'a 1946 Indian Bollywood film'),\n"
    "    ('Aas Ka Panchhi', 'is', 'a 1961 Hindi movie'),\n"
    "    ('Aas Ka Panchhi', 'directed by', 'Mohan Kumar'),\n"
    "    ('Phoolwari', 'was directed by', 'Chaturbhuj Doshi')\n"
    "]\n"
    "\nExpected Output:\n"
    "Relevant events: [\n"
    "    'Aas Ka Panchhi is a 1961 Hindi movie produced by J. Om Prakash.',\n"
    "    'Phoolwari is a 1946 Indian Bollywood film.'\n"
    "]\n"
    "Relevant triplets: [\n"
    "    ('Aas Ka Panchhi', 'is', 'a 1961 Hindi movie'),\n"
    "    ('Phoolwari', 'is', 'a 1946 Indian Bollywood film')\n"
    "]"
)

class LLMGenerator():
    def __init__(self, client, model_name):
        self.model_name = model_name
        self.client : OpenAI|Pipeline  = client
        if isinstance(client, OpenAI):
            self.inference_type = "openai"
        elif isinstance(client, Pipeline):
            self.inference_type = "pipeline"
        else:
            raise ValueError("Unsupported client type. Please provide either an OpenAI client or a Huggingface Pipeline Object.")
        self.cot_system_instruction = "".join(cot_system_instruction)
        self.cot_system_instruction_no_doc = "".join(cot_system_instruction_no_doc)
        self.cot_system_instruction_kg = "".join(cot_system_instruction_kg)
        self.cot_system_instruction_reason_passage = "".join(cot_system_instruction_reason_passage)
        self.filter_system_instruction = "".join(system_instruction_select_t_e)

    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(5))
    def _generate_response(self, messages, do_sample=True, 
                           max_new_tokens=32768,
                           temperature = 0.7,
                           frequency_penalty = None,
                           response_format = {"type": "text"}  # Default response format,
                           ):
        if self.inference_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                frequency_penalty= NOT_GIVEN if frequency_penalty is None else frequency_penalty,
                response_format = response_format if response_format is not None else {"type": "text"},
            )
            return response.choices[0].message.content
        elif self.inference_type == "pipeline":
            # Convert messages to a single string for Hugging Face
            input_text = " ".join([msg["content"] for msg in messages])
            response = self.client(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
            return response[0]["generated_text"]
        else:
            raise ValueError(f"Unsupported client type: {self.client_type}")
    
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(5))
    def filter_generation(self, messages):
        """
        Filter the generation using the configured client.

        Args:
            messages: The input messages for the model.

        Returns:
            The filtered response.
        """
        if self.inference_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                top_p=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"},
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].message.content
        elif self.inference_type == "huggingface":
            response = self.client(
                messages,
                max_new_tokens=4096,
                temperature=0.0,
                top_p=0.1,
                do_sample=True,
            )
            return response[0]["generated_text"]
        else:
            raise ValueError(f"Unsupported client type: {self.client_type}")

    # def _generate_batch_response(self, batch_messages, max_new_tokens=32768, temperature=0.7):
    #     # Use ThreadPoolExecutor for concurrent requests if using API
    #     with ThreadPoolExecutor() as executor:
    #         future_to_index = {
    #             executor.submit(self._generate_response, msg, max_new_tokens, temperature): idx 
    #             for idx, msg in enumerate(batch_messages)
    #         }
    #         results = [None] * len(batch_messages)  # Pre-allocate results list
    #         for future in as_completed(future_to_index):
    #             index = future_to_index[future]  # Get the original index
    #             results[index] = future.result()  # Place the result in the correct position
    #     return results

    def generate(self, question, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": self.cot_system_instruction_no_doc},
            {"role": "user", "content": question},
        ]
        return self._generate_response(messages, max_new_tokens=max_new_tokens)
    
    def generate_without_context_noT(self, question, max_new_tokens=1024, frequency_penalty=None, temperature = 0.7, seed = None):
        without_context_noT = ('Response questions with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations./no_think')
        messages = [
            {"role": "system", "content":"".join(without_context_noT)},
            {"role": "user", "content": f"{question}"},
        ]
        # print(f"messages: {messages}")
        # return self._generate_response(messages, max_new_tokens=max_new_tokens, frequency_penalty=frequency_penalty, temperature = temperature, seed = seed)
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)

    def generate_without_context_llama(self, question, max_new_tokens=1024, frequency_penalty=None, temperature = 0.7, seed = None):
        without_context_noT = ('Please answer questions. '
                                 'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                 'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')

        messages = [
            {"role": "system", "content":"".join(without_context_noT)},
            {"role": "user", "content": f"{question}"},
        ]
        # print(f"messages: {messages}")
        # return self._generate_response(messages, max_new_tokens=max_new_tokens, frequency_penalty=frequency_penalty, temperature = temperature, seed = seed)
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)   


    def generate_with_context_noT(self, question, max_new_tokens=1024, frequency_penalty=None, temperature = 0.7, seed = None):
        without_context_noT = ('Response questions with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')
        messages = [
            {"role": "system", "content":"".join(without_context_noT)},
            {"role": "user", "content": f"{question}"},
        ]
        # print(f"messages: {messages}")
        # return self._generate_response(messages, max_new_tokens=max_new_tokens, frequency_penalty=frequency_penalty, temperature = temperature, seed = seed)
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)

    def generate_with_context(self, question, context, max_new_tokens=1024, frequency_penalty=None, temperature = 0.7, seed = None, think = True):
        # print(context)
        if not think:
            cot_system_instruction = ('As an advanced reading comprehension assistant, your task is to analyze passages and corresponding questions meticulously. If the information is not enough, you can use your own knowledge to answer the question.'
                          'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations./no_think')
        else:
            cot_system_instruction = ('As an advanced reading comprehension assistant, your task is to analyze passages and corresponding questions meticulously. If the information is not enough, you can use your own knowledge to answer the question.'
                          'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')
        messages = [
            {"role": "system", "content": "".join(cot_system_instruction)},
            {"role": "user", "content": f"{context}\n\n{question}"},
        ]
        # print(f"messages: {messages}")
        # return self._generate_response(messages, max_new_tokens=max_new_tokens, frequency_penalty=frequency_penalty, temperature = temperature, seed = seed)
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)
    
    def generate_with_pass_te(self, question, context, max_new_tokens=1024, frequency_penalty=None, temperature = 0.7, seed = None):
        cot_system_instruction_reason_passage = (
            "You are an advanced reading comprehension assistant. "
            "Your task is to generate answers by following a hierarchical reasoning protocol: "
            "first, derive the answer from the provided passages; "
            "if the passages alone are not sufficient, extend the reasoning by integrating information from Relevant triplets and events; "
            "if both sources are still inadequate, resort to your own knowledge as supplementary evidence. "
            "Begin your response with 'Thought: ' to present the step-by-step reasoning process, "
            "and conclude with 'Answer: ' as a concise noun phrase."
        )

        messages = [
            {"role": "system", "content": "".join(cot_system_instruction_reason_passage)},
            {"role": "user", "content": f"{context}\nQuestion: {question}"},
        ]
        # print(f"messages: {messages}")
        # return self._generate_response(messages, max_new_tokens=max_new_tokens, frequency_penalty=frequency_penalty, temperature = temperature, seed = seed)
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)

    def generate_with_te(self, question, context, max_new_tokens=1024, frequency_penalty=None, temperature = 0.7, seed = None):
        cot_system_instruction = ('As an advanced reading comprehension assistant, your task is to analyze Relevant triplets, events and corresponding questions meticulously. If the information is not enough, you can use your own knowledge to answer the question.'
                          'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')
        messages = [
            {"role": "system", "content": "".join(cot_system_instruction)},
            {"role": "user", "content": f"{context}\nQuestion: {question}"},
        ]
        # print(f"messages: {messages}")
        # return self._generate_response(messages, max_new_tokens=max_new_tokens, frequency_penalty=frequency_penalty, temperature = temperature, seed = seed)
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)


    def filter_with_triples_event(self, question, events, triples, max_new_tokens=1024, frequency_penalty=None, temperature = 0.7, seed = None):
        messages = [
            {"role": "system", "content": self.filter_system_instruction},
            {"role": "user", "content": f"Question:{question}\nEvents: {events}\nTriples: {triples}"},
        ]
        # return self._generate_response(messages, max_new_tokens=max_new_tokens, frequency_penalty=frequency_penalty, temperature = temperature, seed = seed)
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)

    def generate_with_triples_event(self, question, context, max_new_tokens=1024, frequency_penalty=None, temperature = 0.7, seed = None):
        messages = [
            {"role": "system", "content": self.cot_system_instruction},
            {"role": "user", "content": f"{context}\nQuestion:{question}\n"},
        ]
        print(f"messages: {messages}")
        # return self._generate_response(messages, max_new_tokens=max_new_tokens, frequency_penalty=frequency_penalty, temperature = temperature, seed = seed)
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)

    def generate_with_reason_passage(self, question, reason_path, context, max_new_tokens=1024, frequency_penalty=None, temperature = 0.7, seed = None):
        messages = [
            {"role": "system", "content": self.cot_system_instruction_reason_passage},
            # {"role": "user", "content": f"reasoning paths:{reason_path}\npassages:{context}\n\n{question}\nThought:"},
            {"role": "user", "content": f"reasoning paths:{reason_path}\n\n{question}\nThought:"},
        ]
        # print(messages)
        # return self._generate_response(messages, max_new_tokens=max_new_tokens, frequency_penalty=frequency_penalty, temperature = temperature, seed = seed)
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)

    def generate_with_context_one_shot(self, question, context, max_new_tokens=4096, frequency_penalty=None, temperature = 0.7, seed = None):
        messages = deepcopy(prompt_template)
        messages.append(
            {"role": "user", "content": f"{context}\n\nQuestions:{question}\nThought:"},
            
        )
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)
    def generate_with_context_kg(self, question, context, max_new_tokens=1024, frequency_penalty=None, temperature = 0.7, seed = None):
        messages = [
            {"role": "system", "content": self.cot_system_instruction_kg},
            {"role": "user", "content": f"{context}\n\n{question}"},
        ]
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)

    def filter_triples_with_entity(self,question, nodes, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": """
            Your task is to filter text cnadidates based on their relevance to a given query.
            The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information.
            You must only select relevant texts from the provided candidate list that have connection to the query, aiding in reasoning and providing an accurate answer.
            The output should be in JSON format, e.g., {"nodes": [e1, e2, e3, e4]}, and if no texts are relevant, return an empty list, {"nodes": []}.
            Do not include any explanations, additional text, or context Only provide the JSON output.
            Do not change the content of each object in the list. You must only use text from the candidate list and cannot generate new text."""},

            {"role": "user", "content": f"""{question} \n Output Before Filter: {nodes} \n Output After Filter:"""}
        ]
        try:
            response = json.loads(self._generate_response(messages, max_new_tokens=max_new_tokens))
            # loop through the reponse json and check if all node is original nodes else go to exception
            return response
        except Exception as e:
            # If all retries fail, return the original triples
            return json.loads(nodes)

    def filter_triples_with_entity_event(self,question, triples):
        messages = deepcopy(filter_messages)
        messages.append(
            {"role": "user", "content": f"""[ ## question ## ]]
        {question}

        [[ ## fact_before_filter ## ]]
        {triples}"""})
        
        try:
            response = self.filter_generation(messages)
            cleaned_data = validate_filter_output(response)
            return cleaned_data['fact']
        except Exception as e:
            # If all retries fail, return the original triples
            return []
    def generate_with_custom_messages(self, custom_messages, do_sample=True, max_new_tokens=1024, temperature=0.8, frequency_penalty = None):
        return self._generate_response(custom_messages, do_sample, max_new_tokens, temperature, frequency_penalty)
    
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_filter_keywords_with_entity(self, question, keywords):
        messages = deepcopy(keyword_filtering_prompt)
        
        messages.append({
            "role": "user",
            "content": f"""[[ ## question ## ]]
            {question}
            [[ ## keywords_before_filter ## ]]
            {keywords}"""
        })
        
        try:
            response = self.generate_with_custom_messages(messages, response_format={"type": "json_object"}, temperature=0.0, max_new_tokens=2048)
            
            # Validate and clean the response
            cleaned_data = validate_keyword_output(response)
            
            return cleaned_data['keywords']
        except Exception as e:
            return keywords
    
    def ner(self, text):
        messages = [
            {"role": "system", "content": "Please extract the entities from the following question and output them separated by comma, in the following format: entity1, entity2, .../no_think"},
            {"role": "user", "content": f"Extract the named entities from: {text}"},
        ]
        # print(self._generate_response(messages))
        return self._generate_response(messages)
    
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_ner(self, text):
        messages = deepcopy(ner_prompt)
        messages.append(
            {
                "role": "user", 
                "content": f"[[ ## question ## ]]\n{text}" 
            }
        )
        
        # Generate raw response from LLM
        raw_response = self._generate_response(messages, max_new_tokens=4096, temperature=0.7, frequency_penalty=1.1, response_format={"type": "json_object"})
        
        try:
            # Validate and clean the response
            cleaned_data = validate_keyword_output(raw_response)
            return cleaned_data['keywords']
        
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            return []  # Fallback to empty list or raise custom exception
 
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_ner(self, text):
        messages = deepcopy(ner_prompt)
        messages.append(
            {
                "role": "user", 
                "content": f"[[ ## question ## ]]\n{text}" 
            }
        )
        
        # Generate raw response from LLM
        raw_response = self._generate_response(messages, max_new_tokens=4096, temperature=0.7, frequency_penalty=1.1, response_format={"type": "json_object"})
        
        try:
            # Validate and clean the response
            cleaned_data = validate_keyword_output(raw_response)
            return cleaned_data['keywords']
        
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            return []  # Fallback to empty list or raise custom exception

    def generate_with_react(self, question, context=None, max_new_tokens=1024, search_history=None, logger=None):
        react_system_instruction = (
            'You are an advanced AI assistant that uses the ReAct framework to solve problems through iterative search. '
            'Follow these steps in your response:\n'
            '1. Thought: Think step by step and analyze if the current context is sufficient to answer the question. If not, review the current context and think critically about what can be searched to help answer the question.\n'
            '   - Break down the question into *1-hop* sub-questions if necessary (e.g., identify key entities like people or places before addressing specific events).\n'
            '   - Use the available context to make inferences about key entities and their relationships.\n'
            '   - If a previous search query (prefix with "Previous search attempt") was not useful, reflect on why and adjust your strategy—avoid repeating similar queries and consider searching for general information about key entities or related concepts.\n'
            '2. Action: Choose one of:\n'
            '   - Search for [Query]: If you need more information, specify a new query. The [Query] must differ from previous searches in wording and direction to explore new angles.\n'
            '   - No Action: If the current context is sufficient.\n'
            '3. Answer: Provide one of:\n'
            '   - A concise, definitive response as a noun phrase if you can answer.\n'
            '   - "Need more information" if you need to search.\n\n'
            'Format your response exactly as:\n'
            'Thought: [your reasoning]\n'
            'Action: [Search for [Query] or No Action]\n'
            'Answer: [concise noun phrase if you can answer, or "Need more information" if you need to search]\n\n'
        )
        
        # Build context with search history if available
        full_context = []
        if search_history:
            for i, (thought, action, observation) in enumerate(search_history):
                search_history_text = f"\nPrevious search attempt {i}:\n"
                search_history_text += f"{action}\n  Result: {observation}\n"
                full_context.append(search_history_text)
        if context:
            full_context_text = f"Current Retrieved Context:\n{context}\n"
            full_context.append(full_context_text)
        if logger:
            logger.info(f"Full context for ReAct generation: {full_context}")
        
        # Combine few-shot examples with system instruction and user query
        messages = [
            {"role": "system", "content": react_system_instruction},
            {"role": "user", "content": f"Search History:\n\n{''.join(full_context)}\n\nQuestion: {question}" 
            if full_context else f"Question: {question}"}
        ]
        if logger:
            logger.info(f"Messages for ReAct generation: {search_history}Question: {question}")
        return self._generate_response(messages, max_new_tokens=max_new_tokens)

    def generate_with_rag_react(self, question: str, retriever: Union['BaseEdgeRetriever', 'BasePassageRetriever'], max_iterations: int = 5, max_new_tokens: int = 1024, logger: Logger = None):
        """
        Generate a response using RAG with ReAct framework, starting with an initial search using the original query.
        
        Args:
            question (str): The question to answer
            retriever: The retriever instance to use for searching
            max_iterations (int): Maximum number of ReAct iterations
            max_new_tokens (int): Maximum number of tokens to generate per iteration
            
        Returns:
            tuple: (final_answer, search_history)
                - final_answer: The final answer generated
                - search_history: List of (thought, action, observation) tuples
        """
        search_history = []
        
        # Perform initial search with the original query
        if isinstance(retriever, BaseEdgeRetriever):
            initial_context, _ = retriever.retrieve(question, topN=5)
            current_context = ". ".join(initial_context)
        elif isinstance(retriever, BasePassageRetriever):
            initial_context, _ = retriever.retrieve(question, topN=5)
            current_context = "\n".join(initial_context)
        
        # Start ReAct process with the initial context
        for iteration in range(max_iterations):
            # First, analyze if we can answer with current context
            analysis_response = self.generate_with_react(
                question=question,
                context=current_context,
                max_new_tokens=max_new_tokens,
                search_history=search_history,
                logger = logger
            )
            
            if logger:
                logger.info(f"Analysis response: {analysis_response}")
                
            try:
                # Parse the analysis response
                thought = analysis_response.split("Thought:")[1].split("\n")[0]
                if logger:
                    logger.info(f"Thought: {thought}")
                action = analysis_response.split("Action:")[1].split("\n")[0]
                answer = analysis_response.split("Answer:")[1].strip()
                
                # If the answer indicates we can answer with current context
                if answer.lower() != "need more information":
                    search_history.append((thought, action, "Using current context"))
                    return answer, search_history
                
                # If we need more information, perform the search
                if "search" in action.lower():
                    # Extract search query from the action
                    search_query = action.split("search for")[-1].strip()
                    
                    # Perform the search
                    if isinstance(retriever, BaseEdgeRetriever):
                        new_context, _ = retriever.retrieve(search_query, topN=3)
                        # Filter out contexts that are already in current_context
                        current_contexts = current_context.split(". ")
                        new_context = [ctx for ctx in new_context if ctx not in current_contexts]
                        new_context = ". ".join(new_context)
                    elif isinstance(retriever, BasePassageRetriever):
                        new_context, _ = retriever.retrieve(search_query, topN=3)
                        # Filter out contexts that are already in current_context
                        current_contexts = current_context.split("\n")
                        new_context = [ctx for ctx in new_context if ctx not in current_contexts]
                        new_context = "\n".join(new_context)
                    
                    # Store the search results as observation
                    if new_context:
                        observation = f"Found information: {new_context}"
                    else:
                        observation = "No new information found. Consider searching for related entities or events."
                    search_history.append((thought, action, observation))
                    
                    # Update context with new search results
                    if new_context:
                        current_context = f"{current_context}\n{new_context}"
                        if logger:
                            logger.info(f"New search results: {new_context}")
                    else:
                        if logger:
                            logger.info("No new information found, suggesting to try related entities")

                else:
                    # If no search is needed but we can't answer, something went wrong
                    search_history.append((thought, action, "No action taken but answer not found"))
                    return "Unable to find answer", search_history
                
            except Exception as e:
                if logger:
                    logger.error(f"Error parsing ReAct response: {e}")
                return analysis_response, search_history
        
        # If we've reached max iterations, return the last answer
        return answer, search_history