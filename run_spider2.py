"""Spider 2.0-Snow evaluation runner: Generate SQL artifacts for official evaluation.

This script:
1. Loads benchmark samples using `Spider2SnowBenchmark`.
2. Formats instructions into Snowflake SQL generation prompts.
3. Performs deterministic LLM inference (greedy decoding, 4-bit quantization).
4. Saves each predicted query into an individual `<instance_id>.sql` file.
"""

import os
import argparse
from tqdm import tqdm
from src.methods.baseline import BaselineModel
from src.benchmarks.spider2.loader import Spider2SnowBenchmark

def main(args):
    # Directory where individual .sql prediction files will be stored
    pred_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    print("1. Loading Spider 2.0-Snow data...")
    bench = Spider2SnowBenchmark(args.jsonl_path, args.resource_dir)
    samples = bench.load_samples(limit=args.limit)
    print(f"-> {len(samples)} samples loaded.")

    print("2. Loading model...")
    model = BaselineModel(model_id=args.model_id, use_4bit=args.use_4bit)

    print("3. Generating SQL queries...")
    for item in tqdm(samples, desc="Spider 2.0 Inference"):
        instance_id = item["question_id"]
        output_file = os.path.join(pred_dir, f"{instance_id}.sql")

        # Resume mechanism: skip if file already exists
        if os.path.exists(output_file):
            continue

        sys_msg = (
            "You are an expert Snowflake SQL data analyst. "
            "Given the database metadata and user instruction, write a valid Snowflake SQL query. "
            "Wrap your SQL query in a markdown code block starting with ```sql and ending with ``` without explanation."
        )
        
        user_msg = (
            f"Database Schema:\n{item['schema']}\n\n"
            f"Hint / Context:\n{item['evidence']}\n\n"
            f"Instruction: {item['question']}\n\n"
            f"SQL:"
        )

        pred_sql = model.generate_sql(sys_msg, user_msg)

        # Save query in official evaluator format: <instance_id>.sql
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(pred_sql)

    print(f"\n Generation completed! SQL files saved to: {pred_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on Spider 2.0-Snow benchmark")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to spider2-snow.jsonl")
    parser.add_argument("--resource_dir", type=str, required=True, help="Path to spider2 resource/ directory")
    parser.add_argument("--output_dir", type=str, default="results/spider2_snow", help="Directory to save generated SQL queries")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Hugging Face model identifier")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to run (None runs the full benchmark)")
    parser.add_argument("--use_4bit", action="store_true", default=True, help="Enable 4-bit quantization with bitsandbytes")
    args = parser.parse_args()
    main(args)
