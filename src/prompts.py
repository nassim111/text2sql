def create_text2sql_prompt(question: str, schema_ddl: str, evidence: str = "") -> str:
    """
    Formate le prompt pour le modèle.
    BIRD fournit souvent une 'evidence' (indication sémantique utile).
    """
    evidence_str = f"Hint / Evidence: {evidence}\n" if evidence else ""

    system_msg = (
        "You are an expert SQL engineer. Given a database schema and a user question, "
        "write a single, syntactically correct SQLite query to answer the question. "
        "Output ONLY the SQL query inside ```sql and ``` code blocks without explanation."
    )

    user_msg = f"""Database Schema:
{schema_ddl}

{evidence_str}Question: {question}

SQL Query:"""

    return system_msg, user_msg
