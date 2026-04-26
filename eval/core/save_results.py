import pandas as pd
import json
from datetime import datetime
import os

def save_results(results: list, filename: str = None):
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval/results/eval_{timestamp}.csv"
    
    rows = []
    for r in results:
        scores = r.get("scores", {})
        rows.append({
            "scenario_id": r.get("scenario_id"),
            "scenario_name": r.get("scenario_name"),
            "failure_mode": r.get("failure_mode"),
            "model": r.get("model", "unknown"),
            "detects_issue": scores.get("detects_issue", 0),
            "explains_why": scores.get("explains_why", 0),
            "provides_fix": scores.get("provides_fix", 0),
            "quantifies_impact": scores.get("quantifies_impact", 0),
            "total": scores.get("total", 0),
            "comment": scores.get("comment", ""),
            "answer": r.get("answer", "")[:500], 
            "timestamp": datetime.now().isoformat()
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")
    return df