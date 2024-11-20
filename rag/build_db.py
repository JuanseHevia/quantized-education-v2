import time
import chromadb
import os 
import torch 
import fitz
from tqdm import tqdm
from chromadb.utils import embedding_functions
import argparse


DATA_PATH = "./data"
KB_PATH = "knowledgebase"
DISTANCE_FN = "cosine" # options: "l2", "cosine", "ip"
EMBEDDING_FN = embedding_functions.DefaultEmbeddingFunction()

os.makedirs(KB_PATH, exist_ok=True)

def initialize_collection(kb_path:str , distance_fn='cosine', embedding_fn=embedding_functions.DefaultEmbeddingFunction()):
    """
    Initializes a collection in the ChromaDB database. By default, chroma uses the same embedding as original Retriever (all-MiniLM-L6-v2)

    Parameters:
    kb_path (str): The path to the ChromaDB database.
    distance_fn (str, optional): The distance function to use for embedding comparison. Defaults to 'cosine'.
    embedding_fn (object, optional): The embedding function to use for generating embeddings. Defaults to embedding_functions.DefaultEmbeddingFunction().

    Returns:
    collection (object): The initialized collection in the ChromaDB database.
    """

    client = chromadb.PersistentClient(path=kb_path)
    collection = client.get_or_create_collection("books",
                                                 embedding_function=embedding_fn,
                                                 metadata={"hnsw:space": distance_fn}) 
    return collection

def process_pdf_doc(chunks, preffix, summarize: bool=False):
    """
    Load PDF function and split in chunks.
    Optionally, summarize the content
    """

    if summarize:
        # TODO: implement LLM based summarization
        raise NotImplementedError("Summarization not implemented yet")
    
    # add IDs 
    ids = [f"{preffix}-{idx}" for idx in list(range(len(chunks)))]

    # add chunks to collection
    for chunk_id, chunk in tqdm(zip(ids, chunks), total=len(chunks)):
        # TODO: we can add metadata to this chunks, esto es genial
        collection.add(documents=[chunk],
                       ids=[chunk_id],)


def extract_text_from_pdf(pdf_path, chunk_size=300):
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    text_chunks = []
    fname = os.path.basename(pdf_path)
    print(f"Parsing file: {fname}")

    for page_num in tqdm(range(len(pdf_document))):
        page_text = pdf_document[page_num].get_text("text")
        # Split page text into chunks of 'chunk_size' words
        words = page_text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            text_chunks.append(chunk)
    
    pdf_document.close()
    return text_chunks


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--kb_path", type=str, default="./knowledgebase")
    parser.add_argument("--distance_fn", type=str, default="cosine")
    parser.add_argument("--chunk_size", type=int, default=500)
    # parse arguments
    args = parser.parse_args()

    # print number of documents found
    booknames = [name for name in os.listdir(args.data_path) if name.endswith(".pdf")]
    print("Number of documents found: ", len(booknames))

    # get chunks
    start = time.time()


    collection = initialize_collection(kb_path=args.kb_path,
                                       distance_fn=args.distance_fn)
    for bookname in booknames:
        _name = os.path.basename(bookname).split(".")[0]
        print(f"Processing: {_name}")
        chunks = extract_text_from_pdf(os.path.join(DATA_PATH, bookname),
                                        chunk_size=args.chunk_size)
        
        print(f"Adding {len(chunks)} chunks from {_name}!")
        process_pdf_doc(chunks=chunks, preffix=_name)

    end = time.time()

    print(f"Database built successfully! - {(start - end) / 60} seconds")