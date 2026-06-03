from __future__ import annotations

import numpy as np
import pandas as pd

from config import CLEAN_JOBS_PATH, EMBEDDING_MODEL_NAME, EMBEDDINGS_PATH, FAISS_INDEX_PATH
from data_pipeline import build_clean_dataset


def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(force_rebuild: bool = False) -> tuple[pd.DataFrame, object]:
    import faiss

    if CLEAN_JOBS_PATH.exists():
        jobs = pd.read_csv(CLEAN_JOBS_PATH)
    else:
        jobs = build_clean_dataset()

    if force_rebuild or not EMBEDDINGS_PATH.exists() or not FAISS_INDEX_PATH.exists():
        embeddings = embed_texts(jobs["job_text"].fillna("").astype(str).tolist())
        np.save(EMBEDDINGS_PATH, embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(FAISS_INDEX_PATH))
    else:
        index = faiss.read_index(str(FAISS_INDEX_PATH))

    return jobs, index


def retrieve_jobs(profile_text: str, top_n: int = 100) -> pd.DataFrame:
    jobs, index = build_faiss_index()
    query = embed_texts([profile_text])
    scores, indices = index.search(query, min(top_n, len(jobs)))
    result = jobs.iloc[indices[0]].copy()
    result["embedding_score"] = scores[0]
    return result.reset_index(drop=True)


if __name__ == "__main__":
    build_faiss_index(force_rebuild=True)
    print("FAISS index and embeddings were created successfully.")
