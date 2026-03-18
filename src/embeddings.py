import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "models/embeddings.npy"
PROCESSED_CSV = "data/processed_papers.csv"


def generate_embeddings():
    if not os.path.exists(PROCESSED_CSV):
        print(f"ERROR: {PROCESSED_CSV} not found. Run preprocessing.py first.")
        return None, None

    print("Loading preprocessed data...")
    df = pd.read_csv(PROCESSED_CSV)
    df = df.dropna(subset=["clean_abstract"])
    df = df[df["clean_abstract"].str.strip() != ""]

    print(f"Total papers to embed: {len(df)}")

    print(f"\nLoading SentenceTransformer model: {MODEL_NAME}")
    print("(First run will download the model ~90MB)")
    model = SentenceTransformer(MODEL_NAME)

    abstracts = df["clean_abstract"].tolist()

    print("\nGenerating embeddings... (this may take a few minutes)")
    embeddings = model.encode(
        abstracts,
        show_progress_bar=True,
        batch_size=64
    )

    os.makedirs("models", exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"\nEmbeddings saved to: {EMBEDDINGS_PATH}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Each paper represented as {embeddings.shape[1]}-dimensional vector")

    return embeddings, df


def load_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        print("Embeddings not found. Generating now...")
        return generate_embeddings()

    print("Loading existing embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(PROCESSED_CSV)
    df = df.dropna(subset=["clean_abstract"])
    df = df[df["clean_abstract"].str.strip() != ""]
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings, df


if __name__ == "__main__":
    generate_embeddings()
