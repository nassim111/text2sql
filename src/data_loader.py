import json
import sqlite3
from pathlib import Path
from typing import List, Dict

def extract_schema_from_db(db_path: str) -> str:
    """Extrait le DDL (CREATE TABLE) d'une base SQLite."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL;")
    tables = cursor.fetchall()
    conn.close()
    return "\n\n".join([t[0] for t in tables])

def load_bird_samples(dev_json_path: str, db_root_path: str, limit: int = None) -> List[Dict]:
    """
    Charge les exemples de BIRD (question, db_id, chemin db, gold SQL).
    """
    with open(dev_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    samples = []
    for item in data:
        db_id = item["db_id"]
        db_path = str(Path(db_root_path) / db_id / f"{db_id}.sqlite")
        
        samples.append({
            "question_id": item.get("question_id", ""),
            "question": item["question"],
            "evidence": item.get("evidence", ""),
            "gold_sql": item.get("SQL", ""),
            "db_id": db_id,
            "db_path": db_path
        })
    return samples
