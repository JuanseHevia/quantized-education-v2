# rag_pipeline.py

import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import importlib
import numpy as np
import os
import commons.model as model
import chromadb

importlib.reload(model)

class ChromaRetriever:
    """
    Retriever implementation using a ChromaDB database.
    """

    def __init__(self, knowledgebase_path: str,
                 collection_name: str = "books") -> None:
        
        # initialize client
        self.client = chromadb.PersistentClient(path=knowledgebase_path)
        self.collection = self.client.get_collection(collection_name)

    def print_results(self, results: list) -> None:
        """
        Print the results of the query.
        """
        for idx, result in enumerate(results):
            print(f"Result {idx+1}: {result['document']}")

    def format_results(self, results: list) -> list:
        """
        Format the results of the query into a list of strings separated by newlines.
        """
        return '\n'.join([result for result in results])
    
    def retrieve_documents(self, query:str,
                           format_results:bool=True,
                           top_k: int = 2) -> str:
        """
        Retrieve relevant documents based on the query and return
        a formatted string representation of the results if needed.
        """
        results = self.collection.query(query_texts=[query], n_results=top_k)
        if format_results:
            return self.format_results(results["documents"][0])

class Retriever:
    def __init__(self, pdf_path,
                 chunk_size=300,
                 embedder_model='sentence-transformers/all-MiniLM-L6-v2',
                 device="cpu",
                 top_k=2):
        # Initialize the retriever
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size

        # Initialize the embedding model
        self.embedder = SentenceTransformer(embedder_model, device=device)

        # Load and embed the documents
        self.documents = self.extract_text_from_pdf()  # TODO: store in a vector database
        self.embeddings = self.embedder.encode(self.documents)

        # Initialize FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        self.top_k = top_k

    def extract_text_from_pdf(self):
        # Open the PDF and extract text in chunks
        pdf_document = fitz.open(self.pdf_path)
        text_chunks = []

        for page_num in range(len(pdf_document)):
            page_text = pdf_document[page_num].get_text("text")
            words = page_text.split()
            for i in range(0, len(words), self.chunk_size):
                chunk = " ".join(words[i:i + self.chunk_size])
                text_chunks.append(chunk)

        pdf_document.close()
        print(f"Extracted {len(text_chunks)} chunks from the PDF.")
        return text_chunks

    def retrieve_documents(self, query):
        # Retrieve relevant documents based on the query
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding, self.top_k)
        results = [self.documents[i] for i in indices[0]]
        return results

    # TODO: clean the implementation so we decouple the retrieval from the generation
    # def ask(self, query):
    #     # Use retrieved documents as context for generation
    #     retrieved_docs = self.retrieve_documents(query)
    #     context = " ".join(retrieved_docs)  # Limit context length if needed
    #     prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    #     # TODO: right now the generator is accumulating the context tokens
    #     response = self.generator.chat(prompt=prompt)
    #     return response
