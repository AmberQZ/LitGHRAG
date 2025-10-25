import re
from collections import Counter
from typing import Tuple
import argparse
import json
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union, Callable
import numpy as np

class QAJudger:
    def __init__(self):
        pass
    
    def split_answer(self, generated_text):
        if "Answer:" in generated_text:
            generated_text = generated_text.split("Answer:")[-1]
        elif "answer:" in generated_text:
            generated_text = generated_text.split("answer:")[-1]
        # if answer is none
        if not generated_text:
            return "none"
        return generated_text

    def normalize_answer(self, answer: str) -> str:
        """Direct copy of the normalization from QAExactMatch/QAF1Score"""
        # Lowercase and normalize whitespace
        answer = answer.lower()
        # Replace hyphens with spaces
        answer = answer.replace('-', ' ')
        # Remove all other punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        # Standardize whitespace
        return ' '.join(answer.split())

    def judge(self, generated_text: str, reference_text: str) -> Tuple[int, float]:
        """Direct port of the original scoring logic"""
        # Extract answer from generated text
        pred_answer = self.split_answer(generated_text)
        
        # Normalize both answers
        pred_norm = self.normalize_answer(pred_answer)
        ref_norm = self.normalize_answer(reference_text)

        # Exact match calculation
        em = 1 if pred_norm == ref_norm else 0

        # F1 calculation (direct port from QAF1Score)
        pred_tokens = pred_norm.split()
        ref_tokens = ref_norm.split()
        
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return em, 0.0

        precision = num_same / len(pred_tokens) if pred_tokens else 0.0
        recall = num_same / len(ref_tokens) if ref_tokens else 0.0

        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        return em, f1

    def judge_simple_and_multi(self, generated_text: str, reference_text: List[str], aggregation_fn: Callable = np.max) -> Tuple[int, float]:
        """Direct port of the original scoring logic"""
        # Extract answer from generated text
        pred_answer = self.split_answer(generated_text)

        em_scores = [1.0 if self.normalize_answer(gold) == self.normalize_answer(pred_answer) else 0.0 for gold in reference_text]
        em = aggregation_fn(em_scores)

        def compute_f1(gold: str, predicted: str) -> float:
            gold_tokens = self.normalize_answer(gold).split()
            predicted_tokens = self.normalize_answer(predicted).split()

            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = 1.0 * num_same / len(predicted_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            return 2 * (precision * recall) / (precision + recall)

        f1_scores = [compute_f1(gold, pred_answer) for gold in reference_text]
        f1 = aggregation_fn(f1_scores)
        return em, f1

    def recall_at_k(self, retrieved_text: list, reference_text: list, k: int) -> float:
        """Calculates recall at k based on the top k retrieved texts."""
        successful_retrievals = 0

        # Limit the retrieved texts to the top k entries
        limited_retrieved_text = retrieved_text[:k]

        for ref_text in reference_text:
            for ret_text in limited_retrieved_text:
                if ref_text in ret_text:
                    successful_retrievals += 1
                    break

        recall = successful_retrievals / len(reference_text) if reference_text else 0
        return recall

    # recall for 1 answer
    def recall(self, retrieved_text: list, reference_text: list) -> dict:
        """Calculates recall values at different k levels."""
        recall_values = {
            'recall@2': self.recall_at_k(retrieved_text, reference_text, 2),
            'recall@5': self.recall_at_k(retrieved_text, reference_text, 5),
        }
        return recall_values['recall@2'], recall_values['recall@5']
    