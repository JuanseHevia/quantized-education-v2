import torch
import model 
import retrieval

class Pipeline:
    """
    Pipeline class to manage the interaction between the model and the retriever.
    """
    
    ANCHOR_PROMPT = '''You are a biology assistant, please help me solve the following biology questions. I will
    provide you with a question and a paragraph from a biology textbook. You need to answer the question based on the
    information in the paragraph. If you cannot find the answer, please say "I don't know". Let's start!'''

    def __init__(self,
                 model_path: str,
                 pdf_path: str,
                 chunk_size: int = 500,
                 embedder_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 device: str = "cpu",
                 top_k: int = 2):
        """
        Initialize the Pipeline with a model and retriever.
        """
        # Initialize the model
        self.model = model.SmolModel(model_path, device=device, anchor_prompt=self.ANCHOR_PROMPT)
        
        # Initialize the retriever
        self.retriever = retrieval.Retriever(pdf_path, chunk_size, embedder_model, device, top_k)



    def ask(self, query: str, clean_on_exit: bool = False):
        """
        Ask a question and retrieve the relevant documents to generate the answer.
        """
        # Retrieve relevant documents
        documents = self.retriever.retrieve_documents(query)
        
        context = "\nHere are some relevant references to keep in mind: " + "\n".join(documents)

        # Generate the answer based on the question and retrieved documents
        response = self.model.chat(query + context)

        # Clean the history if needed
        if clean_on_exit:
            self.model.clean_history()

        return response
    
    def clean_history(self):
        """
        Clean the history of the model.
        """
        self.model.clean_history()

    def get_token_probs(self, query: str = None,
                             token_idx : int = -1):
            """
            Get token probabilities for a given query.
            This is commonly used for evaluation on MMLU.

            Args:
                query (str): The input query for which token probabilities are calculated.
                token_idx (int): The index of the token for which probabilities are calculated. Default is -1.

            Returns:
                log_probs (torch.Tensor): The log probabilities of the specified token.

            """
            encoded = self.tokenizer(query, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attn_masks = encoded["attention_mask"]

            # Get the model output
            with torch.no_grad():
                outputs = self.model.model(input_ids=input_ids.to(self.model.device),
                                          attention_mask=attn_masks.to(self.model.device))

                logits = outputs.logits[:, token_idx, :]  # Logits at token_idx
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            return log_probs
    
    def which_next_tokens(self, token_choices: list, query: str = None,
                              token_id_offset: int = 65, token_idx: int = -1):
            """
            Get the probabilities of the next tokens given a list of token choices.

            Args:
                token_choices (list): A list of token choices.
                query (str, optional): The query string. Defaults to None.
                token_id_offset (int, optional): The offset value for token IDs. Defaults to 65.
                token_idx (int, optional): The index of the token. Defaults to -1.

            Returns:
                tuple: A tuple containing the index of the chosen token and a list of log probabilities for each choice.
            """
            token_probs = self.get_token_probs(query=query, token_idx=token_idx)
            choice_log_probs = []

            for idx in range(len(token_choices)):
                # get index in the tokenizer vocabulary for the specific token
                choice_token = f"{chr(token_id_offset + idx)}"  # if token_id_offset = 65, this is" A", " B", " C", " D" , ...
                choice_ids = self.tokenizer(choice_token, return_tensors="pt")["input_ids"]
                    
                # Get log probability for the specific choice (A, B, C, or D)
                choice_log_prob = token_probs[0, choice_ids[0, 0]].item()
                choice_log_probs.append(choice_log_prob)

            # compute the chosen token from the list by taking argmax on the token probs
            pred_chosen_idx = torch.tensor(choice_log_probs).argmax().item()

            return pred_chosen_idx, choice_log_probs
