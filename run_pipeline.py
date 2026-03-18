"""
run_pipeline.py — Run the full ResearchTrendAI pipeline from terminal.

Usage:
    python run_pipeline.py

This runs:
    Step 1: Preprocessing
    Step 2: Neural Embeddings
    Step 3: ANN Autoencoder
    Step 4: Topic Modeling + Clustering
    Step 5: Trend Analysis
    Step 6: Gap Detection

After this, launch the dashboard with:
    streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("\n" + "="*55)
print("   ResearchTrendAI — Full Pipeline Runner")
print("="*55)


# ── Step 1: Preprocessing ──
print("\n[Step 1] NLP Preprocessing...")
from preprocessing import preprocess_dataset
preprocess_dataset()


# ── Step 2: Generate Embeddings ──
print("\n[Step 2] Generating Neural Embeddings...")
from embeddings import generate_embeddings
embeddings, df = generate_embeddings()


# ── Step 3: Autoencoder ──
print("\n[Step 3] Running ANN Autoencoder...")
from autoencoder import encode_embeddings
encoded = encode_embeddings(embeddings)


# ── Step 4: Topic Modeling ──
print("\n[Step 4] Discovering Research Topics...")
from topic_modeling import discover_topics
topics, cluster_labels, df_clustered = discover_topics(encoded=encoded, df=df)


# ── Step 5: Trend Analysis ──
print("\n[Step 5] Analyzing Research Trends...")
from trend_analysis import analyze_trends, get_trending_clusters
trend_pivot = analyze_trends(df=df_clustered)
if trend_pivot is not None:
    trending = get_trending_clusters(trend_pivot)


# ── Step 6: Gap Detection ──
print("\n[Step 6] Detecting Research Gaps...")
from gap_detection import detect_gaps, suggest_directions
gaps = detect_gaps(df=df_clustered, topics=topics)
suggestions = suggest_directions(gaps, topics)

print("\n💡 Suggested Research Directions:")
for i, s in enumerate(suggestions, 1):
    print(f"  {i}. {s}")


print("\n" + "="*55)
print("   Pipeline Complete!")
print("   Launch dashboard: streamlit run app.py")
print("="*55 + "\n")
