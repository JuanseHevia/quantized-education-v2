from typing import List
import mlx_lm
import dataclasses
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

AVAILABLE_HFS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
]

@dataclasses.dataclass
class MlxModel:
    hf_path: str

    def load_model(self):
        # Load the model from hf_path
        model, tokenizer = mlx_lm.load(self.hf_path)
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, verbose: bool = False):
        return mlx_lm.generate(self.model, self.tokenizer,
                               prompt,
                               verbose=verbose)
    
@dataclasses.dataclass
class ConversationModel:
    hf_path: str
    device: str = "mps"
    messages: List = dataclasses.field(default_factory=list)
    temperature : float = 0.99
    max_tokens : int = 512
    anchor_prompt : str = None

    def __post_init__(self):
        """
        Downlaod weights if not present already and load models.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_path).to(self.device)

        if self.anchor_prompt:
            self.messages.append({"role": "system", "content": self.anchor_prompt})

    def format_messages(self):
        raise NotImplementedError
    
    def generate(self, prompt:str):
        """
        Get a single generation, without affecting the message history
        """
        _input = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(_input, max_length=self.max_tokens,
                                     do_sample=True, temperature=self.temperature)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def chat(self, prompt: str, verbose: bool = False):
        """
        Simulate a single chat interaction with the model
        """
        self.messages.append({"role" : "user", "content" : prompt})
        _input = self.tokenizer.apply_chat_template(conversation=self.messages,
                                                    tokenize=False, add_generation_prompt=True)
        _input = self.tokenizer.encode(_input, return_tensors="pt").to(self.device)

        # Generate response
        output = self.model.generate(_input, max_length=self.max_tokens,
                                     do_sample=True, temperature=self.temperature)
        
        # Decode output
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Update messages
        self.messages.append({"role": "model", "content": response})
        return response
    

    def print_message_history(self):
        """
        Get a quick, formatted view of the message history
        """
        for message in self.messages:
            print(f"{message['role']}: {message['content']}")
        

    def clean_history(self):
        """
        Clear the message history
        """
        self.messages = []


@dataclasses.dataclass
class TinyLlamaModel(ConversationModel):
    hf_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@dataclasses.dataclass    
class SmolModel(ConversationModel):
    hf_path: str = "HuggingFaceTB/SmolLM2-1.7B"
    is_chat : bool = False

    def __post_init__(self):
        if self.is_chat:
            self.hf_path = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        return super().__post_init__()