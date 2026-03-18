import numpy as np
import pandas as pd
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

CLUSTERS_PATH = "models/cluster_labels.npy"
TFIDF_PATH = "models/tfidf_vectorizer.pkl"
TOPICS_PATH = "models/topics.pkl"

N_CLUSTERS = 10


def discover_topics(encoded=None, df=None):
    """
    Uses KMeans clustering on ANN-encoded embeddings to discover research topics.
    Then uses TF-IDF to label each cluster with meaningful keywords.
    """
    from src.autoencoder import load_encoded

    if encoded is None:
        encoded = load_encoded()

    if df is None:
        df = pd.read_csv("data/processed_papers.csv")
        df = df.dropna(subset=["clean_abstract"])
        df = df[df["clean_abstract"].str.strip() != ""]

    # Ensure sizes match
    min_len = min(len(encoded), len(df))
    encoded = encoded[:min_len]
    df = df.iloc[:min_len].copy()

    print(f"Running KMeans clustering with {N_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(encoded)

    df["cluster"] = cluster_labels

    print("Computing TF-IDF topic keywords per cluster...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df["clean_abstract"])
    feature_names = vectorizer.get_feature_names_out()

    topics = {}
    for cluster_id in range(N_CLUSTERS):
        cluster_mask = (df["cluster"] == cluster_id).values
        cluster_tfidf = tfidf_matrix[cluster_mask]

        if cluster_tfidf.shape[0] == 0:
            topics[cluster_id] = ["unknown"]
            continue

        mean_tfidf = cluster_tfidf.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[::-1][:10]
        top_keywords = [feature_names[i] for i in top_indices]
        topics[cluster_id] = top_keywords

        count = cluster_mask.sum()
        print(f"Cluster {cluster_id} ({count} papers): {', '.join(top_keywords[:5])}")

    # Save
    os.makedirs("models", exist_ok=True)
    np.save(CLUSTERS_PATH, cluster_labels)

    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(TOPICS_PATH, "wb") as f:
        pickle.dump(topics, f)

    df.to_csv("data/clustered_papers.csv", index=False)

    print(f"\nTopic modeling complete! Saved to models/")
    return topics, cluster_labels, df


def load_topics():
    if not os.path.exists(TOPICS_PATH):
        print("Topics not found. Running topic modeling...")
        return discover_topics()

    with open(TOPICS_PATH, "rb") as f:
        topics = pickle.load(f)
    cluster_labels = np.load(CLUSTERS_PATH)
    df = pd.read_csv("data/clustered_papers.csv")
    return topics, cluster_labels, df


if __name__ == "__main__":
    discover_topics()
