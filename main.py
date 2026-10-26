import os
import json
import argparse
from tqdm import tqdm
from src.data_loader import load_bird_samples, extract_schema_from_db
from src.prompts import create_text2sql_prompt
from src.methods.baseline import BaselineModel
from src.evaluation import evaluate_execution_accuracy

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("1. Chargement des données...")
    samples = load_bird_samples(args.dev_json, args.db_root, limit=args.limit)
    print(f"-> {len(samples)} exemples chargés.")

    print("2. Chargement du modèle...")
    model = BaselineModel(model_id=args.model_id, use_4bit=args.use_4bit)

    results = []
    correct_count = 0
    schemas_cache = {}

    print("3. Inférence et évaluation...")
    for item in tqdm(samples):
        db_path = item["db_path"]
        
        # Mise en cache du schéma pour ne pas relire SQLite à chaque fois
        if item["db_id"] not in schemas_cache:
            schemas_cache[item["db_id"]] = extract_schema_from_db(db_path)
        schema = schemas_cache[item["db_id"]]

        # Préparation du prompt et génération
        sys_msg, user_msg = create_text2sql_prompt(item["question"], schema, item["evidence"])
        pred_sql = model.generate_sql(sys_msg, user_msg)

        # Évaluation
        is_correct = evaluate_execution_accuracy(pred_sql, item["gold_sql"], db_path)
        if is_correct:
            correct_count += 1

        results.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "gold_sql": item["gold_sql"],
            "pred_sql": pred_sql,
            "is_correct": is_correct
        })

    accuracy = correct_count / len(samples) if samples else 0.0
    print(f"\n--- Résultat Baseline ---")
    print(f"Total testés: {len(samples)}")
    print(f"Execution Accuracy: {accuracy * 100:.2f}%")

    out_file = os.path.join(args.output_dir, "baseline_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "details": results}, f, indent=4)
    print(f"Résultats sauvegardés dans {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_json", type=str, required=True, help="Chemin vers dev.json de BIRD")
    parser.add_argument("--db_root", type=str, required=True, help="Chemin vers le dossier dev_databases")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    # parser.add_argument("--limit", type=int, default=20, help="Nombre d'exemples pour le premier test")
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
