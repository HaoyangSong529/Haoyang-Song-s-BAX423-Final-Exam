from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import FEEDBACK_PATH

ACTION_WEIGHTS = {
    "accept": 1.0,
    "skip": -0.2,
    "reject": -1.0,
}


def load_feedback(path: Path = FEEDBACK_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["job_id", "action", "weight"])
    return pd.read_csv(path)


def save_feedback(job_id: str, action: str, path: Path = FEEDBACK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feedback = load_feedback(path)
    row = pd.DataFrame([{"job_id": job_id, "action": action, "weight": ACTION_WEIGHTS.get(action, 0.0)}])
    feedback = pd.concat([feedback, row], ignore_index=True)
    feedback.to_csv(path, index=False)


def feedback_weights(path: Path = FEEDBACK_PATH) -> dict[str, float]:
    feedback = load_feedback(path)
    if feedback.empty:
        return {}
    grouped = feedback.groupby("job_id")["weight"].mean().to_dict()
    return {str(k): float(v) for k, v in grouped.items()}


def simulate_feedback_improvement(scored_jobs: pd.DataFrame, rounds: int = 50) -> pd.DataFrame:
    simulation = scored_jobs.head(max(1, rounds)).copy()
    simulation["simulated_action"] = simulation["final_score"].apply(lambda x: "accept" if x >= scored_jobs["final_score"].median() else "reject")
    simulation["before_score"] = simulation["final_score"]
    simulation["after_score"] = simulation.apply(
        lambda row: row["final_score"] + (0.05 if row["simulated_action"] == "accept" else -0.05), axis=1
    )
    return pd.DataFrame(
        {
            "metric": ["mean_top_score_before", "mean_top_score_after", "simulated_rounds"],
            "value": [
                float(simulation["before_score"].head(10).mean()),
                float(simulation.sort_values("after_score", ascending=False)["after_score"].head(10).mean()),
                float(len(simulation)),
            ],
        }
    )
