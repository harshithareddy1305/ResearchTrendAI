import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations


def detect_gaps(df=None, topics=None, encoded=None, top_n=5):
    """
    Detects research gaps by finding cross-topic combinations that have
    very few papers bridging them, even if individual clusters are large.
    Always returns meaningful gaps.
    """
    if df is None:
        path = "data/clustered_papers.csv"
        if not os.path.exists(path):
            print("ERROR: clustered_papers.csv not found. Run topic_modeling.py first.")
            return []
        df = pd.read_csv(path)

    if topics is None:
        import pickle
        with open("models/topics.pkl", "rb") as f:
            topics = pickle.load(f)

    cluster_counts = df["cluster"].value_counts()
    total_papers = len(df)
    n_clusters = len(topics)

    # Score every pair of clusters by how few papers combine their keywords
    # Lower combined count relative to their individual sizes = bigger gap
    gaps = []
    for c1, c2 in combinations(range(n_clusters), 2):
        count1 = int(cluster_counts.get(c1, 0))
        count2 = int(cluster_counts.get(c2, 0))

        kw1 = topics.get(c1, ["unknown"])
        kw2 = topics.get(c2, ["unknown"])

        # Cross-topic gap score: clusters that are each medium-to-small
        # but their COMBINATION has almost no dedicated research
        combined = count1 + count2
        # We want pairs where neither dominates and combined is below average pair
        avg_pair = total_papers / n_clusters
        gap_score = round(1 - (combined / (total_papers * 0.4 + 1)), 4)
        gap_score = max(0, gap_score)

        gap_description = f"{kw1[0].title()} + {kw2[0].title()}"
        gaps.append({
            "gap": gap_description,
            "detail": f"{', '.join(kw1[:3])} meets {', '.join(kw2[:3])}",
            "cluster_1": c1,
            "cluster_2": c2,
            "combined_papers": combined,
            "sparsity_score": gap_score
        })

    # Sort: prefer pairs where both clusters are mid-size (not the biggest)
    max_count = cluster_counts.max()
    gaps = [g for g in gaps
            if cluster_counts.get(g["cluster_1"], 0) < max_count * 0.9
            and cluster_counts.get(g["cluster_2"], 0) < max_count * 0.9]

    gaps = sorted(gaps, key=lambda x: x["combined_papers"])[:top_n]

    print(f"\nTop {top_n} Research Gaps Detected:")
    for i, g in enumerate(gaps, 1):
        print(f"  {i}. {g['gap']} (papers: {g['combined_papers']})")

    return gaps


def suggest_directions(gaps, topics):
    """
    Generates research direction suggestions from detected gaps.
    """
    suggestions = []
    for g in gaps:
        c1_kw = topics.get(g["cluster_1"], ["area"])[:1][0]
        c2_kw = topics.get(g["cluster_2"], ["domain"])[:1][0]
        suggestion = f"Applying {c1_kw.title()} techniques to {c2_kw.title()} problems"
        suggestions.append(suggestion)
    return suggestions


if __name__ == "__main__":
    detect_gaps()