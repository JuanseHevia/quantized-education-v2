from typing import List
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from dataclasses import dataclass, field
from commons.retrieval import ChromaRetriever

QUANTIZATION_OPTIONS = {
'4bit': BitsAndBytesConfig(load_in_4bit=True),
'8bit': BitsAndBytesConfig(load_in_8bit=True)
}


@dataclass
class BenchmarkDataset:
    name: str = "cais/mmlu"
    subtasks: List = field(default_factory=list)
    sample_size : int = -1
    seed : int = 42
    CHOICES = ["A", "B", "C", "D"]
    device: str = "cpu"
    top_k: int = 2
    quantization: str = 'none' # change to 4bit or 8bit to quantize the model
    add_answer: bool = False
    rag: bool = False


    def __post_init__(self):
        self.dataset = load_dataset(self.name, "all").filter(lambda x: x["subject"] in self.subtasks)
        self.test_set = self.dataset["test"]
        if self.sample_size > 0:
            self.test_set = self.test_set.shuffle(seed=self.seed).select(list(range(self.sample_size)))
        
    def load_model(self, hf_path, rag=False, rag_path=None):
        """
        Load the tokenizer and model from the given Hugging Face path.

        Args:
            hf_path (str): The path to the Hugging Face model.

        Returns:
            None
        """

        self.tokenizer = AutoTokenizer.from_pretrained(hf_path)

        if self.quantization == 'none':
            self.model = AutoModelForCausalLM.from_pretrained(hf_path).to(self.device)
        elif self.quantization in QUANTIZATION_OPTIONS:
            config = QUANTIZATION_OPTIONS[self.quantization]
            self.model = AutoModelForCausalLM.from_pretrained(hf_path, quantization_config=config)
        else:
            raise Exception(f"Quantization option {self.quantization} not supported.")
        
        # store the choice IDs based on the chosen tokenizer
        self.choice_ids_mapping = {}
        for choice_token in self.CHOICES:
            choice_ids = self.tokenizer(choice_token, return_tensors="pt")["input_ids"]
            self.choice_ids_mapping[choice_token] = choice_ids[0,0]
        
        if self.rag:
            # self.rag_pipeline = Retriever(pdf_path, device=self.device)
            self.rag_pipeline = ChromaRetriever(knowledgebase_path=rag_path, )
            print("RAG pipeline loaded successfully.")

    def _format_example(self,example):
        """
        Formats an example dictionary into a string representation.

        Args:
            example (dict): The example dictionary containing the question and choices.

        Returns:
            str: The formatted string representation of the example.

        """
        question = example['question']
        choices = [example[f'choices'][i] for i in range(4)]
        prompt = f''''{question}
            CHOICES:
            A) {choices[0]}
            B) {choices[1]}
            C) {choices[2]}
            D) {choices[3]}

            Please select the correct answer by providing only the corresponding letter (A, B, C, or D).
            '''
        if self.rag:
            context = example.get('context', None)
            answer = example.get('answer', None)

            if self.add_answer:
                prompt += f'\nThe correct answer is: {self.CHOICES[answer]}'

            context_add = f'''\n
            USE THIS INFORMATION TO HELP YOU ANSWER THE QUESTION:\n
            {context}
            '''
            prompt = prompt + context_add

        prompt += "\nANSWER:"

        return prompt

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
            correct_qs = []

            for example in tqdm(self.test_set, desc=f"Evaluating - So far gotten {correct} out of {total}"):
                # Format the question and choices into a single prompt
                if self.rag:
                    example['context'] = self.rag_pipeline.retrieve_documents(example['question'], top_k=self.top_k)
                prompt = self._format_example(example)
                
                # Tokenize the prompt
                inputs = self.tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                # Compute log probabilities for each answer choice
                choice_log_probs = []
                
                # Pass through the model and compute log probs for the choice token
                with torch.no_grad():
                    # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=False)
                    # logits = outputs.logits[:, -1, :].cpu()  # Last token's logits
                    logits = self.model(input_ids=input_ids, 
                                        attention_mask=attention_mask, 
                                        use_cache=False, 
                                        output_hidden_states=False).\
                                logits[:, -1, :].cpu()  # Last token's logits

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                for choice in self.CHOICES:
                    # Get log probability for the specific choice (A, B, C, or D)
                    choice_log_prob = log_probs[0, self.choice_ids_mapping[choice]].item()
                    choice_log_probs.append(choice_log_prob)

                # Determine the predicted choice, by computing the arg max of hte 
                # log probs we just built. This is very greedy (we don't sample, just take the max logprob)
                predicted_choice_idx = torch.tensor(choice_log_probs).argmax().item()

                # Check if prediction is correct
                correct_answer = example["answer"]
                if predicted_choice_idx == correct_answer:
                    correct += 1
                    correct_qs.append(1)
                else:
                    correct_qs.append(0)

                total += 1

            # Calculate accuracy
            accuracy = correct / total if total > 0 else 0
            print(f"Accuracy: {accuracy * 100:.2f}%")
            return accuracy, self.test_set, correct_qs