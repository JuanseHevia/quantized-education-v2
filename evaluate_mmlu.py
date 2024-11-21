import argparse
import os
from commons import benchmarks
import datetime
import json

# MODEL = "HuggingFaceTB/SmolLM2-1.7B"
# MODEL = "HuggingFaceTB/SmolLM2-135M"
MODEL = "HuggingFaceTB/SmolLM-135M"
SUBTASKS = ["college_biology", "high_school_biology"]

def evaluate_mmlu(args, use_rag: bool):
    """
    Evaluate the model on the MMLU benchmark.
    """

    if args.sample_size > 0:
        print(f"Using sample size: {args.sample_size}")
    else:
        print("Using full dataset.")


    # Instantiate dataset
    dataset = benchmarks.BenchmarkDataset(name="cais/mmlu", 
                                          subtasks=SUBTASKS,
                                          sample_size=args.sample_size, 
                                          seed=args.seed,
                                          device=args.device, 
                                          top_k=args.top_k,
                                          quantization=args.quantization)
    dataset.load_model(
            hf_path=MODEL,
            rag=args.rag,
            rag_path=args.kb_path)
    
    # run evaluation
    acc, subset, per_question_res = dataset.evaluate()
    print("Accuracy: ", acc)
    print("Subset: ", subset)

    # save results and score as JSON object
    today = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    with open(f"{args.result_path}/{today}-mmlu_results.json", "w") as f:
        _obj = {
            "accuracy": acc,
            "questions": subset["question"],
            "results": per_question_res
        }
        json.dump(_obj, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rag", type=str, default="True")
    parser.add_argument("--kb_path", type=str, default="./rag/knowledgebase")
    parser.add_argument("--result_path", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--quantization", type=str, default="none")
    args = parser.parse_args()

    _use_rag = args.rag.lower() == "true"

    if _use_rag:
        assert os.path.exists(args.kb_path), "Knowledgebase path does not exist."
        print("Using RAG pipeline.")

    os.makedirs(args.result_path, exist_ok=True)
    evaluate_mmlu(args, use_rag = _use_rag)

