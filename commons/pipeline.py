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
