# JobPilot: Smart Job Matcher and Resume Builder

This project implements a BAX-423 final project application for intelligent job matching. It includes data ingestion, deduplication, embedding-based retrieval, multi-stage ranking, adaptive feedback learning, batch analytics, CSV export, and tailored resume generation.

## Features

- Generates or loads a structured job posting dataset.
- Deduplicates postings using a stable hash ID.
- Embeds job descriptions using SentenceTransformers.
- Builds a FAISS approximate nearest-neighbor retrieval index.
- Applies hard filters for salary, location, seniority, contract status, defense companies, visa sponsorship, and company size.
- Scores jobs using semantic similarity, skill overlap, role match, salary fit, and feedback weight.
- Saves accept, skip, and reject feedback for adaptive re-ranking.
- Creates a tailored resume draft for a selected job.
- Provides market analytics and downloadable CSV output.

## Folder Structure

```text
jobpilot_bax423_final/
  code/
    app.py
    config.py
    data_pipeline.py
    feedback.py
    ranking.py
    resume_builder.py
    retrieval.py
    run_pipeline.py
  data/
  requirements.txt
  README.md
  prompts.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Pipeline

```bash
cd code
python run_pipeline.py
```

The first run may take several minutes because the app creates embeddings and a FAISS index.

## Run the App

```bash
cd code
streamlit run app.py
```

## Data Notes

If `data/jobs_sample.csv` is not present, the pipeline generates a synthetic offline dataset. To use a Kaggle or API dataset, place a CSV at `data/jobs_sample.csv` with these recommended columns:

- title
- company
- location
- salary_min
- salary_max
- employment_type
- company_size
- visa_sponsorship
- years_required
- skills
- description
- url
- source

The cleaning pipeline will fill missing optional columns with defaults.

## Suggested Deployment

Streamlit Community Cloud, Render, or Google Cloud Run can host this app. For a simple demo, Streamlit Community Cloud is the fastest option. For Google Cloud, containerize the app and run `streamlit run code/app.py --server.port 8080 --server.address 0.0.0.0`.
