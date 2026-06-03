from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class UserPreferences:
    target_roles: str
    skills: str
    location_preference: str
    minimum_salary: float
    dealbreakers: str
    visa_required: bool = False
    minimum_company_size: int = 0


def parse_skills(text: str) -> set[str]:
    tokens = re.split(r"[,;\n]| and ", str(text).lower())
    return {token.strip() for token in tokens if len(token.strip()) >= 2}


def contains_any(text: str, terms: list[str]) -> bool:
    lowered = str(text).lower()
    return any(term.lower() in lowered for term in terms)


def apply_hard_filters(jobs: pd.DataFrame, prefs: UserPreferences) -> pd.DataFrame:
    result = jobs.copy()
    dealbreakers = prefs.dealbreakers.lower()

    if "no senior" in dealbreakers or "no staff" in dealbreakers:
        result = result[~result["title"].str.lower().str.contains("senior|staff", na=False)]
    if "no junior" in dealbreakers:
        result = result[~result["title"].str.lower().str.contains("junior", na=False)]
    if "no contract" in dealbreakers or "contract" in dealbreakers:
        result = result[~result["employment_type"].str.lower().str.contains("contract|temporary", na=False)]
    if "no defense" in dealbreakers or "no defence" in dealbreakers:
        result = result[~result["company"].str.lower().str.contains("defense|defence|military", na=False)]

    if prefs.minimum_salary > 0:
        result = result[result["salary_max"].fillna(0) >= prefs.minimum_salary]
    if prefs.minimum_company_size > 0:
        result = result[result["company_size"].fillna(0) >= prefs.minimum_company_size]
    if prefs.visa_required:
        result = result[result["visa_sponsorship"].str.lower().isin(["yes", "unknown"])]

    location = prefs.location_preference.lower().strip()
    if location and location not in {"any", "any us", "us"}:
        if "remote" in location:
            result = result[result["location"].str.lower().str.contains("remote", na=False)]
        elif "bay area" in location:
            result = result[result["location"].str.lower().str.contains("san francisco|remote", na=False)]
        else:
            result = result[result["location"].str.lower().str.contains(location, na=False)]

    return result.copy()


def score_jobs(jobs: pd.DataFrame, prefs: UserPreferences, feedback_weight: dict[str, float] | None = None) -> pd.DataFrame:
    feedback_weight = feedback_weight or {}
    user_skills = parse_skills(prefs.skills)
    target_terms = parse_skills(prefs.target_roles)

    scored = jobs.copy()
    skill_scores = []
    role_scores = []
    salary_scores = []
    feedback_scores = []

    for _, row in scored.iterrows():
        job_text = f"{row.get('title', '')} {row.get('skills', '')} {row.get('description', '')}".lower()
        job_skills = parse_skills(row.get("skills", ""))
        overlap = len(user_skills.intersection(job_skills))
        skill_scores.append(overlap / max(1, len(user_skills)))

        role_match = max([1.0 if term in job_text else 0.0 for term in target_terms], default=0.0)
        role_scores.append(role_match)

        salary_max = float(row.get("salary_max", 0) or 0)
        salary_scores.append(min(1.0, salary_max / max(1.0, prefs.minimum_salary)) if prefs.minimum_salary else 0.5)

        job_id = str(row.get("job_id", ""))
        feedback_scores.append(feedback_weight.get(job_id, 0.0))

    scored["skill_score"] = skill_scores
    scored["role_score"] = role_scores
    scored["salary_score"] = salary_scores
    scored["feedback_score"] = feedback_scores
    scored["final_score"] = (
        0.45 * scored["embedding_score"].fillna(0)
        + 0.25 * scored["skill_score"]
        + 0.15 * scored["role_score"]
        + 0.10 * scored["salary_score"]
        + 0.05 * scored["feedback_score"]
    )
    scored = scored.sort_values("final_score", ascending=False).reset_index(drop=True)
    return scored


def explain_ranking(row: pd.Series) -> str:
    reasons = []
    reasons.append(f"Semantic match score: {row.get('embedding_score', 0):.3f}")
    reasons.append(f"Skill overlap score: {row.get('skill_score', 0):.3f}")
    reasons.append(f"Role match score: {row.get('role_score', 0):.3f}")
    reasons.append(f"Salary fit score: {row.get('salary_score', 0):.3f}")
    return "; ".join(reasons)


def ranking_metric_at_k(scored_jobs: pd.DataFrame, k: int = 10) -> dict[str, float]:
    top = scored_jobs.head(k).copy()
    if top.empty:
        return {"precision_at_k": 0.0, "average_score_at_k": 0.0}
    relevant = (top["final_score"] >= top["final_score"].median()).astype(int)
    return {
        "precision_at_k": float(relevant.mean()),
        "average_score_at_k": float(top["final_score"].mean()),
    }
