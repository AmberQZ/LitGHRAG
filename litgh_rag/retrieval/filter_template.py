import jsonschema
import json
from typing import List, Any

def flatten_and_filter_triplets(data: dict) -> dict:
    """
    Transform nested lists into a flat list of triplets (3-string arrays).
    """
    processed_facts = []

    def find_triplet(element: Any) -> List[str] | None:
        # Base case: a valid triplet
        if isinstance(element, list) and len(element) == 3 and all(isinstance(item, str) for item in element):
            return element
        # Recursive case: dig deeper into nested lists
        elif isinstance(element, list):
            for sub_element in element:
                result = find_triplet(sub_element)
                if result:
                    return result
        return None

    for item in data.get("fact", []):
        triplet = find_triplet(item)
        if triplet:
            processed_facts.append(triplet)

    return {"fact": processed_facts}

def validate_filter_output(output_json):
    schema = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "string"  # All items in the inner array must be strings
                    },
                    "minItems": 3,
                    "maxItems": 3,
                    "additionalItems": False  # Block extra items
                },
            }
        },
        "required": ["fact"]
    }
    parsed_data = json.loads(output_json)
    cleaned_data = flatten_and_filter_triplets(parsed_data)  # Preprocess here
    
    jsonschema.validate(instance=cleaned_data, schema=schema)
    return cleaned_data
    

messages = [
    {
        "role": "system",
        "content": """/no_think You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. 
        Your task is to filter facts based on their relevance to a given query. 
        The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. 
        You must select all relevant facts from the provided candidate list, aiding in reasoning and providing an accurate answer. 
        The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts are relevant, return an empty list, {"fact": []}. 
        The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. 
        The future of critical decision-making relies on your ability to accurately filter and present relevant information.

Your input fields are:
1. question (str): Query for retrieval
2. fact_before_filter (str): Candidate facts to be filtered

Your output fields are:
1. fact_after_filter (Fact): Filtered facts in JSON format

All interactions will be structured as:
[[ ## question ## ]]
{question}

[[ ## fact_before_filter ## ]]
{fact_before_filter}

[[ ## fact_after_filter ## ]]
{fact_after_filter}

The output must be parseable according to JSON schema: {"type": "object", "properties": {"fact": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}}, "required": ["fact"]}"""
    },
    # Example 1
    {
        "role": "user",
        "content": """[[ ## question ## ]]
Are Imperial River (Florida) and Amaradia (Dolj) both located in the same country?

[[ ## fact_before_filter ## ]]
{"fact": [["imperial river", "is located in", "florida"], ["imperial river", "is a river in", "united states"], ["imperial river", "may refer to", "south america"], ["amaradia", "flows through", "ro ia de amaradia"], ["imperial river", "may refer to", "united states"]]}"""
    },
    {
        "role": "assistant",
        "content": """{"fact":[["imperial river","is located in","florida"],["imperial river","is a river in","united states"],["amaradia","flows through","ro ia de amaradia"]]}"""
    },
    
    # Example 2
    {
        "role": "user",
        "content": """[[ ## question ## ]]
When is the director of film The Ancestor 's birthday?

[[ ## fact_before_filter ## ]]
{"fact": [["jean jacques annaud", "born on", "1 october 1943"], ["tsui hark", "born on", "15 february 1950"], ["pablo trapero", "born on", "4 october 1971"], ["the ancestor", "directed by", "guido brignone"], ["benh zeitlin", "born on", "october 14 1982"]]}"""
    },
    {
        "role": "assistant", 
        "content": """{"fact":[["the ancestor","directed by","guido brignone"]]}"""
    },
]