from typing import List
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .model import ConversationModel
from dataclasses import dataclass, field

@dataclass
class BenchmarkDataset:
    name: str = "cais/mmlu"
    subtasks: List = field(default_factory=list)
    sample_size : int = 100
    seed : int = 42
    ANSWER_TOKEN_IDX_START = 49
    CHOICES : List = ["A", "B", "C", "D"]

    def __post_init__(self):
        self.dataset = load_dataset(self.name, "all")
        if len(self.subtasks) > 0:
            self.dataset = self.dataset.filter(lambda x: x["subject"] in self.subtasks)
        
        self.test_set = self.dataset['test'].shuffle(seed=self.seed)


    def load_model(self, hf_path):
        """
        Load the tokenizer and model from the given Hugging Face path.

        Args:
            hf_path (str): The path to the Hugging Face model.

        Returns:
            None
        """
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path)
        self.model = AutoModelForCausalLM.from_pretrained(hf_path)


    @staticmethod
    def _format_example(example):
        """
        Formats an example dictionary into a string representation.

        Args:
            example (dict): The example dictionary containing the question and choices.

        Returns:
            str: The formatted string representation of the example.

        """
        question = example['question']
        choices = [example[f'choices'][i] for i in range(4)]
        formatted = f"{question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer:"
        return formatted

    def _encode(self, examples):
            """
            Encodes a list of examples into tokenized inputs.

            Args:
                examples (list): List of examples to encode.

            Returns:
                list: List of tokenized inputs.
            """
            inputs = []
            for example in examples:
                question, choices = self.format_example(example)
                inputs.append(self.tokenizer([question] * len(choices), choices, truncation=True, padding=True))
            return inputs
    

    def evaluate(self):
            """
            Run an evaluation on the test set, on a subset of size
            self.sample_size

            Returns:
            - accuracy (float): The accuracy of the model on the test set.
            """
            # Ensure the model and tokenizer are loaded
            assert hasattr(self, 'model'), "Model not loaded. Run load_model() first."
            assert hasattr(self, 'tokenizer'), "Tokenizer not loaded. Run load_model() first."

            correct = 0
            total = 0

            # Shuffle and sample the test set
            sampled_subset = self.test_set.shuffle(seed=self.seed).select(range(self.sample_size))

            # TODO: Review whether the comparison is properly done. This means, what we look for as the 
            # predicted answer is the same as the ground truth answer. i.e. before, I had a typo and
            # I was comparing the predicted answer to " A", " B", " C", " D" instead of "A", "B", "C", "D"

            for example in tqdm(sampled_subset, desc=f"Evaluating - So far gotten {correct} out of {total}"):
                # Format the question and choices into a single prompt
                prompt = self._format_example(example)
                
                # Tokenize the prompt
                inputs = self.tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                # Compute log probabilities for each answer choice
                choice_log_probs = []
                
                # Pass through the model and compute log probs for the choice token
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[:, -1, :]  # Last token's logits
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # for idx in range(4):
                # choice_token = f"{chr(self.ANSWER_TOKEN_IDX_START + idx)}"  # " A", " B", " C", " D"
                for choice_token in self.CHOICES:
                    choice_ids = self.tokenizer(choice_token, return_tensors="pt")["input_ids"]
                        
                    # Get log probability for the specific choice (A, B, C, or D)
                    choice_log_prob = log_probs[0, choice_ids[0, 0]].item()
                    choice_log_probs.append(choice_log_prob)

                # Determine the predicted choice
                predicted_choice_idx = torch.tensor(choice_log_probs).argmax().item()
                predicted_choice = chr(self.ANSWER_TOKEN_IDX_START + predicted_choice_idx)

                # Check if prediction is correct
                correct_answer = example["answer"]
                if predicted_choice == correct_answer:
                    correct += 1
                total += 1

            # Calculate accuracy
            accuracy = correct / total if total > 0 else 0
            print(f"Accuracy: {accuracy * 100:.2f}%")
            return accuracy, sampled_subset