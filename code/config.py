from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JOBS_PATH = DATA_DIR / "jobs_sample.csv"
CLEAN_JOBS_PATH = DATA_DIR / "jobs_clean.csv"
EMBEDDINGS_PATH = DATA_DIR / "job_embeddings.npy"
FAISS_INDEX_PATH = DATA_DIR / "job_index.faiss"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"

DEFAULT_TOP_K = 10
RANDOM_SEED = 423
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TARGET_SAMPLE_SIZE = 5000
