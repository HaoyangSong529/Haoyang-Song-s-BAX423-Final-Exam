from data_pipeline import build_clean_dataset
from retrieval import build_faiss_index


def main() -> None:
    jobs = build_clean_dataset()
    print(f"Clean dataset contains {len(jobs):,} postings.")
    build_faiss_index(force_rebuild=True)
    print("Embedding index completed.")


if __name__ == "__main__":
    main()
