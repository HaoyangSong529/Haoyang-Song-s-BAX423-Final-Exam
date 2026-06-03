from __future__ import annotations

import pandas as pd
import streamlit as st

from data_pipeline import build_clean_dataset
from feedback import feedback_weights, save_feedback, simulate_feedback_improvement
from ranking import UserPreferences, apply_hard_filters, explain_ranking, ranking_metric_at_k, score_jobs
from resume_builder import build_profile_text, extract_skills_from_resume, extract_text_from_pdf, generate_tailored_resume
from retrieval import build_faiss_index, retrieve_jobs

st.set_page_config(page_title="JobPilot", page_icon="💼", layout="wide")

st.title("JobPilot: Smart Job Matcher and Resume Builder")
st.caption("BAX-423 final project application: ingestion, embedding retrieval, multi-stage ranking, feedback learning, analytics, and resume tailoring.")

with st.sidebar:
    st.header("Data and Index")
    if st.button("Build or Refresh Dataset"):
        with st.spinner("Building clean job dataset..."):
            jobs = build_clean_dataset()
        st.success(f"Dataset ready with {len(jobs):,} postings.")

    if st.button("Build or Refresh FAISS Index"):
        with st.spinner("Embedding jobs and building FAISS index. This may take several minutes on first run."):
            jobs, _ = build_faiss_index(force_rebuild=True)
        st.success(f"Index ready for {len(jobs):,} postings.")

st.header("1. Candidate Profile")
col1, col2 = st.columns(2)
with col1:
    uploaded_resume = st.file_uploader("Upload resume PDF", type=["pdf"])
    manual_resume = st.text_area("Or paste resume/profile text", height=180)
    target_roles = st.text_input("Target roles", "Data Analyst, Analytics Engineer, Junior Data Scientist")
    skills = st.text_input("Skills", "Python, SQL, Tableau, PySpark, NLP")
with col2:
    location_preference = st.text_input("Location preference", "Any US")
    minimum_salary = st.number_input("Minimum salary", min_value=0, value=80000, step=5000)
    dealbreakers = st.text_area("Dealbreakers", "No 3+ years experience required. No contract-only roles.", height=100)
    visa_required = st.checkbox("Needs H-1B / visa sponsorship", value=False)
    minimum_company_size = st.number_input("Minimum company size", min_value=0, value=0, step=50)
    top_k = st.slider("Top jobs to display", min_value=5, max_value=50, value=10)

resume_text = ""
if uploaded_resume is not None:
    resume_text = extract_text_from_pdf(uploaded_resume)
if manual_resume.strip():
    resume_text = manual_resume.strip()

if not skills.strip() and resume_text:
    skills = extract_skills_from_resume(resume_text)

prefs = UserPreferences(
    target_roles=target_roles,
    skills=skills,
    location_preference=location_preference,
    minimum_salary=float(minimum_salary),
    dealbreakers=dealbreakers,
    visa_required=visa_required,
    minimum_company_size=int(minimum_company_size),
)

st.header("2. Match and Rank Jobs")
if st.button("Find Best Matches", type="primary"):
    if not target_roles.strip() and not skills.strip() and not resume_text.strip():
        st.error("Please provide a resume, target roles, or skills before searching.")
    else:
        profile_text = build_profile_text(
            resume_text=resume_text,
            target_roles=target_roles,
            skills=skills,
            preferences=f"{location_preference}; salary >= {minimum_salary}; {dealbreakers}",
        )
        with st.spinner("Retrieving semantically similar jobs..."):
            candidates = retrieve_jobs(profile_text, top_n=250)
        filtered = apply_hard_filters(candidates, prefs)
        if filtered.empty:
            st.warning("No jobs remained after hard filters. Try relaxing salary, location, or dealbreakers.")
        else:
            scored = score_jobs(filtered, prefs, feedback_weights())
            scored["why_ranked_here"] = scored.apply(explain_ranking, axis=1)
            st.session_state["scored_jobs"] = scored
            st.success(f"Found {len(scored):,} ranked jobs after filtering.")

if "scored_jobs" in st.session_state:
    scored = st.session_state["scored_jobs"]
    metrics = ranking_metric_at_k(scored, k=min(10, len(scored)))
    st.subheader("Ranking Metrics")
    metric_cols = st.columns(2)
    metric_cols[0].metric("Precision@10 proxy", f"{metrics['precision_at_k']:.2f}")
    metric_cols[1].metric("Average top score", f"{metrics['average_score_at_k']:.3f}")

    display_cols = [
        "title", "company", "location", "salary_min", "salary_max", "employment_type",
        "years_required", "visa_sponsorship", "final_score", "why_ranked_here", "url",
    ]
    st.subheader("Top Ranked Jobs")
    st.dataframe(scored[display_cols].head(top_k), use_container_width=True)

    csv_data = scored[display_cols + ["description", "skills"]].head(top_k).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Top Jobs CSV",
        data=csv_data,
        file_name="jobpilot_top_jobs.csv",
        mime="text/csv",
    )

    st.subheader("3. Feedback Learning")
    selected_index = st.selectbox("Select a job for feedback or resume generation", list(range(min(top_k, len(scored)))))
    selected_job = scored.iloc[int(selected_index)]
    st.write(f"Selected job: **{selected_job['title']}** at **{selected_job['company']}**")
    feedback_cols = st.columns(3)
    if feedback_cols[0].button("Accept"):
        save_feedback(str(selected_job["job_id"]), "accept")
        st.success("Feedback saved. Run matching again to apply learning weight.")
    if feedback_cols[1].button("Skip"):
        save_feedback(str(selected_job["job_id"]), "skip")
        st.info("Feedback saved. Run matching again to apply learning weight.")
    if feedback_cols[2].button("Reject"):
        save_feedback(str(selected_job["job_id"]), "reject")
        st.warning("Feedback saved. Run matching again to apply learning weight.")

    sim = simulate_feedback_improvement(scored, rounds=50)
    st.write("Simulated feedback improvement over 50 rounds")
    st.dataframe(sim, use_container_width=True)

    st.subheader("4. Generate Tailored Resume")
    tailored_resume = generate_tailored_resume(resume_text, selected_job)
    st.text_area("Tailored resume draft", tailored_resume, height=350)
    st.download_button(
        label="Download Tailored Resume TXT",
        data=tailored_resume.encode("utf-8"),
        file_name="tailored_resume.txt",
        mime="text/plain",
    )

    st.header("5. Batch Market Analytics")
    analytics_cols = st.columns(3)
    analytics_cols[0].bar_chart(scored["location"].value_counts().head(10))
    analytics_cols[1].bar_chart(scored["employment_type"].value_counts())
    salary_df = scored[["salary_min", "salary_max"]].copy()
    analytics_cols[2].line_chart(salary_df.head(100))
else:
    st.info("Build the dataset/index first if needed, then click Find Best Matches.")
