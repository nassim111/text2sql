import sqlite3

def execute_query(db_path: str, query: str):
    """Exécute une requête et retourne les résultats."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        res = cursor.fetchall()
        conn.close()
        return res, None
    except Exception as e:
        return None, str(e)

def evaluate_execution_accuracy(predicted_sql: str, gold_sql: str, db_path: str) -> bool:
    """Compare le résultat d'exécution de la requête prédite vs la requête de vérité."""
    pred_res, pred_err = execute_query(db_path, predicted_sql)
    gold_res, _ = execute_query(db_path, gold_sql)

    if pred_err is not None or gold_res is None:
        return False

    # Comparaison sous forme de set pour ignorer l'ordre (si pas de ORDER BY strict)
    return set(pred_res) == set(gold_res)
