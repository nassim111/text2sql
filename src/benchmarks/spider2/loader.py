"""Spider 2.0-Snow benchmark loader: Load tasks, schemas, and domain documentation.

This module:
1. Parses `spider2-snow.jsonl` containing user instructions and task IDs.
2. Reads and aggregates multi-file schemas (DDL, Markdown, JSON) from `resource/databases/`.
3. Retrieves external domain knowledge documents from `resource/documents/`.
4. Outputs standardized sample dictionaries for the evaluation pipeline.
"""

import json
from pathlib import Path
from typing import List, Dict

class Spider2SnowBenchmark:
    def __init__(self, jsonl_path: str, resource_dir: str):
        """
        Initializes paths to the Spider 2.0-Snow resources.
        Uses pathlib.Path to ensure cross-platform compatibility (Linux, Windows, macOS).
        """
        self.jsonl_path = Path(jsonl_path)
        self.resource_dir = Path(resource_dir)
        self.db_dir = self.resource_dir / "databases"    # Directory containing table metadata & DDLs
        self.doc_dir = self.resource_dir / "documents"   # Directory containing external domain documentation

    def load_samples(self, limit: int = None) -> List[Dict]:
        """
        Reads tasks and maps each instruction to its corresponding schema and external context.

        Args:
            limit (int, optional): Maximum number of samples to load. Useful for quick debugging.

        Returns:
            List[Dict]: Standardized list of evaluation samples.
        """
        samples = []

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit and idx >= limit:
                    break

                item = json.loads(line)

                # Official instance identifier (e.g., 'sf001') required for artifact generation
                instance_id = item.get("instance_id", f"task_{idx}")
                db_id = item.get("db_id", "")

                # 1. Load schema metadata
                schema_text = self._load_schema_for_db(db_id)

                # 2. Retrieve external domain knowledge if provided
                doc_text = ""
                if item.get("external_knowledge"):
                    doc_file = self.doc_dir / item["external_knowledge"]
                    if doc_file.exists():
                        with open(doc_file, "r", encoding="utf-8", errors="ignore") as df:
                            doc_text = df.read()

                # Unified sample dictionary contract for the pipeline
                samples.append({
                    "question_id": instance_id,       # Unique task ID
                    "question": item["instruction"],    # Natural language instruction
                    "evidence": doc_text,              # External knowledge / documentation
                    "schema": schema_text,             # Aggregated table schemas & descriptions
                    "db_id": db_id                     # Snowflake database name
                })

        return samples

    def _load_schema_for_db(self, db_id: str) -> str:
        """
        Collects and formats all schema metadata associated with the database.
        Files are sorted by name to ensure deterministic schema ordering across runs.
        """
        target_db_folder = self.db_dir / db_id
        if not target_db_folder.exists():
            return f"Database: {db_id}"

        schema_parts = [f"Database: {db_id}"]

        # sorted() guarantees consistent prompt context across different operating systems
        for file_path in sorted(target_db_folder.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix in [".sql", ".csv", ".md", ".json", ".txt"]:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as sf:
                        # Truncate large files to prevent exceeding model context windows
                        schema_parts.append(f"--- File: {file_path.name} ---\n{sf.read()[:2000]}")
                except Exception:
                    pass

        return "\n\n".join(schema_parts)
