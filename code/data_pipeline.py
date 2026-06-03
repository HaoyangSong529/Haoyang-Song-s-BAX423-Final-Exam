from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import CLEAN_JOBS_PATH, DATA_DIR, JOBS_PATH, RANDOM_SEED, TARGET_SAMPLE_SIZE

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ROLE_FAMILIES = {
    "machine_learning": ["Machine Learning Engineer", "Applied Scientist", "AI Engineer", "Data Scientist"],
    "analytics": ["Data Analyst", "BI Analyst", "Analytics Engineer", "Junior Data Scientist"],
    "mlops": ["MLOps Engineer", "ML Platform Engineer", "Senior ML Engineer", "Data Platform Engineer"],
    "research": ["Research Scientist", "Computer Vision Engineer", "NLP Scientist", "Deep Learning Engineer"],
}

SKILL_BANK = {
    "machine_learning": ["Python", "SQL", "pandas", "scikit-learn", "PyTorch", "TensorFlow", "ML pipelines", "model evaluation"],
    "analytics": ["SQL", "Tableau", "Power BI", "R", "Python", "Excel", "statistics", "dashboarding"],
    "mlops": ["Python", "Kubernetes", "Docker", "Kafka", "Spark", "AWS", "CI/CD", "model serving"],
    "research": ["Python", "C++", "PyTorch", "NLP", "computer vision", "deep learning", "papers", "experimentation"],
}

LOCATIONS = ["Remote", "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX", "Boston, MA", "Chicago, IL"]
COMPANIES = [
    "Northstar AI", "DataBridge Health", "CloudMetric", "Nova Labs", "InsightWorks", "Vertex Analytics",
    "BlueRiver Tech", "BrightAI", "OmniData", "Atlas Systems", "Pioneer Research", "HelioCloud",
]
EMPLOYMENT_TYPES = ["Full-time", "Contract", "Temporary"]
VISA_SPONSORSHIP = ["Yes", "No", "Unknown"]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def stable_job_id(row: pd.Series) -> str:
    key = f"{row.get('title', '')}|{row.get('company', '')}|{row.get('location', '')}|{row.get('description', '')[:120]}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


def generate_synthetic_jobs(n_rows: int = TARGET_SAMPLE_SIZE) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        family = random.choice(list(ROLE_FAMILIES.keys()))
        title = random.choice(ROLE_FAMILIES[family])
        if random.random() < 0.08:
            title = "Senior " + title if "Senior" not in title else title
        if random.random() < 0.04:
            title = "Staff " + title
        if random.random() < 0.06:
            title = "Junior " + title if "Junior" not in title else title

        skills = random.sample(SKILL_BANK[family], k=min(len(SKILL_BANK[family]), random.randint(4, 7)))
        years = random.choice([0, 1, 2, 3, 4, 5, 6, 7])
        salary_min = random.choice([75000, 90000, 110000, 130000, 150000, 180000, 200000])
        salary_max = salary_min + random.choice([20000, 30000, 40000, 50000])
        company_size = random.choice([20, 50, 80, 120, 300, 1000, 5000, 20000])
        employment_type = random.choices(EMPLOYMENT_TYPES, weights=[0.86, 0.10, 0.04])[0]
        sponsorship = random.choices(VISA_SPONSORSHIP, weights=[0.35, 0.45, 0.20])[0]
        defense = random.random() < 0.03
        description = (
            f"We are hiring a {title} to work on {family.replace('_', ' ')} products. "
            f"The role requires {', '.join(skills)}. "
            f"Preferred experience is {years}+ years. "
            f"Responsibilities include building reliable data products, collaborating with cross-functional teams, "
            f"and communicating technical trade-offs clearly."
        )
        rows.append(
            {
                "title": title,
                "company": random.choice(COMPANIES) + (" Defense" if defense else ""),
                "location": random.choice(LOCATIONS),
                "salary_min": salary_min,
                "salary_max": salary_max,
                "employment_type": employment_type,
                "company_size": company_size,
                "visa_sponsorship": sponsorship,
                "years_required": years,
                "skills": ", ".join(skills),
                "description": description,
                "url": f"https://example.com/jobs/{i}",
                "source": "synthetic_snapshot",
            }
        )
    df = pd.DataFrame(rows)
    df["job_id"] = df.apply(stable_job_id, axis=1)
    return df


def load_raw_jobs(path: Path = JOBS_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_jobs()
    df.to_csv(path, index=False)
    return df


def clean_jobs(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["title", "company", "location", "description"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    default_values = {
        "salary_min": 0,
        "salary_max": 0,
        "employment_type": "Unknown",
        "company_size": 0,
        "visa_sponsorship": "Unknown",
        "years_required": 0,
        "skills": "",
        "url": "",
        "source": "uploaded_or_generated",
    }
    for col, value in default_values.items():
        if col not in df.columns:
            df[col] = value

    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["company"] = df["company"].fillna("").astype(str)
    df["location"] = df["location"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["skills"] = df["skills"].fillna("").astype(str)
    df["job_text"] = (
        df["title"] + " at " + df["company"] + ". Skills: " + df["skills"] + ". Description: " + df["description"]
    )
    df["job_id"] = df.apply(stable_job_id, axis=1)
    df = df.drop_duplicates(subset=["job_id"]).reset_index(drop=True)

    numeric_cols = ["salary_min", "salary_max", "company_size", "years_required"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def build_clean_dataset() -> pd.DataFrame:
    df = load_raw_jobs()
    clean = clean_jobs(df)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_JOBS_PATH, index=False)
    return clean


if __name__ == "__main__":
    output = build_clean_dataset()
    print(f"Saved {len(output):,} clean job postings to {CLEAN_JOBS_PATH}")
