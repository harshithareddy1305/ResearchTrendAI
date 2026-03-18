import pandas as pd
import numpy as np
import os


def analyze_trends(df=None):
    """
    Counts research papers per cluster per year to detect trending topics.
    Returns a pivot table: rows=year, columns=cluster, values=paper_count
    """
    if df is None:
        path = "data/clustered_papers.csv"
        if not os.path.exists(path):
            print("ERROR: clustered_papers.csv not found. Run topic_modeling.py first.")
            return None
        df = pd.read_csv(path)

    # Clean year column
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "cluster"])
    df["year"] = df["year"].astype(int)
    df["cluster"] = df["cluster"].astype(int)

    # Filter reasonable years
    df = df[(df["year"] >= 2015) & (df["year"] <= 2024)]

    # Count papers per cluster per year
    trend_data = df.groupby(["year", "cluster"]).size().reset_index(name="count")
    trend_pivot = trend_data.pivot(index="year", columns="cluster", values="count").fillna(0)

    print("Trend analysis complete!")
    print(trend_pivot)

    trend_pivot.to_csv("data/trends.csv")
    return trend_pivot


def get_trending_clusters(trend_pivot, top_n=5):
    """
    Detect fastest-growing clusters using recent 3-year growth rate.
    """
    years = sorted(trend_pivot.index.tolist())

    if len(years) < 2:
        print("Not enough yearly data for trend detection.")
        return []

    recent_years = years[-3:]
    early_years = years[:3]

    recent_counts = trend_pivot.loc[trend_pivot.index.isin(recent_years)].mean()
    early_counts = trend_pivot.loc[trend_pivot.index.isin(early_years)].mean()

    # Growth rate
    growth = (recent_counts - early_counts) / (early_counts + 1)
    trending = growth.sort_values(ascending=False).head(top_n)

    print("\nTop Trending Clusters (Growth Rate):")
    for cluster_id, rate in trending.items():
        print(f"  Cluster {int(cluster_id)}: +{rate:.1%} growth")

    return trending.index.tolist()


if __name__ == "__main__":
    trend_pivot = analyze_trends()
    if trend_pivot is not None:
        get_trending_clusters(trend_pivot)
