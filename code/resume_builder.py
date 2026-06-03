from __future__ import annotations

import re

import pandas as pd


def extract_text_from_pdf(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    try:
        import pdfplumber

        with pdfplumber.open(uploaded_file) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)
    except Exception:
        try:
            import fitz

            data = uploaded_file.read()
            doc = fitz.open(stream=data, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        except Exception:
            return ""


def extract_skills_from_resume(resume_text: str) -> str:
    known_skills = [
        "Python", "R", "SQL", "Tableau", "Power BI", "Excel", "pandas", "NumPy", "scikit-learn",
        "PyTorch", "TensorFlow", "Spark", "PySpark", "Kafka", "Docker", "Kubernetes", "AWS", "GCP",
        "NLP", "computer vision", "statistics", "A/B testing", "machine learning", "data pipelines",
    ]
    found = []
    lower = resume_text.lower()
    for skill in known_skills:
        if skill.lower() in lower:
            found.append(skill)
    return ", ".join(sorted(set(found)))


def generate_tailored_resume(resume_text: str, job: pd.Series) -> str:
    title = job.get("title", "Selected Role")
    company = job.get("company", "Selected Company")
    skills = job.get("skills", "")
    description = job.get("description", "")

    candidate_skills = extract_skills_from_resume(resume_text)
    if not candidate_skills:
        candidate_skills = "Python, SQL, analytics, communication, problem solving"

    summary = (
        f"Professional Summary\n"
        f"Data and analytics professional targeting the {title} role at {company}. "
        f"Strong fit for this position based on experience with {candidate_skills}. "
        f"Prepared to contribute to projects involving {skills}.\n"
    )

    skill_section = f"\nRelevant Skills\n{candidate_skills}\n"
    project_section = (
        "\nSelected Project Experience\n"
        f"- Built data-driven workflows using tools relevant to this role: {skills}.\n"
        "- Developed reproducible analysis pipelines, ranking logic, and evaluation metrics for business use cases.\n"
        "- Communicated technical findings through concise dashboards, reports, and stakeholder-facing summaries.\n"
    )
    alignment = (
        "\nRole Alignment Notes\n"
        f"This resume version emphasizes keywords and responsibilities found in the job description: {description[:500]}...\n"
    )
    return summary + skill_section + project_section + alignment


def build_profile_text(resume_text: str, target_roles: str, skills: str, preferences: str) -> str:
    compact_resume = re.sub(r"\s+", " ", resume_text or "")[:3000]
    return f"Target roles: {target_roles}. Skills: {skills}. Preferences: {preferences}. Resume: {compact_resume}"
