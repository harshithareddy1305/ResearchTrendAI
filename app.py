import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(
    page_title="ResearchTrendAI",
    page_icon=None,
    layout="wide"
)

st.markdown("""
<style>
    /* Page background */
    .stApp { background-color: #f7f5f2; }
    .block-container { padding: 2.5rem 3rem 2rem 3rem; max-width: 1200px; }

    /* Hide sidebar toggle and default decorations */
    [data-testid="stSidebarNav"] { display: none; }

    /* Sidebar */
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }

    /* Typography */
    h1 { font-size: 1.6rem !important; font-weight: 600 !important; color: #1a1a1a !important; letter-spacing: -0.3px; }
    h2 { font-size: 1.1rem !important; font-weight: 600 !important; color: #2c2c2c !important; margin-top: 2rem !important; }
    h3 { font-size: 0.95rem !important; font-weight: 500 !important; color: #444 !important; }
    p, li { color: #444; font-size: 0.9rem; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8e4df;
        border-radius: 8px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #888 !important; text-transform: uppercase; letter-spacing: 0.5px; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #1a1a1a !important; font-weight: 600 !important; }

    /* Divider */
    hr { border: none; border-top: 1px solid #e8e4df; margin: 1.5rem 0; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border: 1px solid #e8e4df; border-radius: 8px; overflow: hidden; }

    /* Buttons */
    .stButton > button {
        background: #ffffff;
        border: 1px solid #d4cfc9;
        color: #333;
        border-radius: 6px;
        font-size: 0.82rem;
        padding: 0.4rem 0.9rem;
        width: 100%;
        text-align: left;
        transition: background 0.15s;
    }
    .stButton > button:hover { background: #f0ece8; border-color: #b0a9a0; }

    /* Selectbox */
    [data-testid="stSelectbox"] select { font-size: 0.88rem; }

    /* Caption */
    .caption-text { font-size: 0.78rem; color: #999; margin-top: 2rem; border-top: 1px solid #e8e4df; padding-top: 0.8rem; }

    /* Section label */
    .section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #aaa;
        margin-bottom: 0.5rem;
    }

    /* Gap card */
    .gap-card {
        background: #fff;
        border: 1px solid #e8e4df;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        color: #2c2c2c;
    }
    .gap-meta { font-size: 0.75rem; color: #aaa; margin-top: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("<p style='font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;color:#aaa;margin-bottom:0.8rem'>Pipeline</p>", unsafe_allow_html=True)

    has_raw = os.path.exists("data/research_papers.csv")
    has_processed = os.path.exists("data/processed_papers.csv")
    has_embeddings = os.path.exists("models/embeddings.npy")
    has_encoded = os.path.exists("models/encoded_embeddings.npy")
    has_clustered = os.path.exists("data/clustered_papers.csv")
    has_topics = os.path.exists("models/topics.pkl")

    def status(flag): return "<span style='color:#5cb85c;font-size:0.75rem'>ready</span>" if flag else "<span style='color:#ccc;font-size:0.75rem'>pending</span>"

    st.markdown(f"<p style='font-size:0.82rem;color:#555;margin:2px 0'>Papers CSV &nbsp; {status(has_raw)}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.82rem;color:#555;margin:2px 0'>Preprocessed &nbsp; {status(has_processed)}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.82rem;color:#555;margin:2px 0'>Embeddings &nbsp; {status(has_embeddings)}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.82rem;color:#555;margin:2px 0'>Encoded &nbsp; {status(has_encoded)}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.82rem;color:#555;margin:2px 0'>Clustered &nbsp; {status(has_clustered)}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.82rem;color:#555;margin:2px 0'>Topics &nbsp; {status(has_topics)}</p>", unsafe_allow_html=True)



# ── Main ──
st.markdown("<h1>ResearchTrendAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#888;font-size:0.9rem;margin-top:-0.5rem;margin-bottom:1.5rem'>Research Trend and Gap Analyzer</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

if not has_topics:
    st.markdown("""
    <div style='background:#fff;border:1px solid #e8e4df;border-radius:8px;padding:1.5rem 2rem;'>
    <p style='font-weight:600;color:#2c2c2c;margin-bottom:0.5rem'>Getting Started</p>
    <p style='color:#666;font-size:0.88rem'>Run the four pipeline steps in the sidebar to analyze the dataset. Each step builds on the previous one.</p>
    <ol style='color:#666;font-size:0.88rem;margin-top:0.8rem'>
    <li>Preprocess Data</li>
    <li>Generate Embeddings</li>
    <li>Run Autoencoder</li>
    <li>Discover Topics</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
else:
    df = pd.read_csv("data/clustered_papers.csv")
    with open("models/topics.pkl", "rb") as f:
        topics = pickle.load(f)

    cluster_counts = df["cluster"].value_counts()

    # ── Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Papers", f"{len(df):,}")
    col2.metric("Topics Found", len(topics))
    col3.metric("Year Range", f"{int(df['year'].min())} – {int(df['year'].max())}" if "year" in df.columns else "N/A")
    col4.metric("Avg Papers / Topic", f"{len(df)//len(topics)}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Topics Table ──
    st.markdown("## Research Topics")
    topic_rows = []
    for cid, kws in topics.items():
        topic_rows.append({
            "Topic": f"Topic {cid}",
            "Keywords": ", ".join(kws[:6]),
            "Papers": int(cluster_counts.get(cid, 0))
        })
    topic_df = pd.DataFrame(topic_rows).sort_values("Papers", ascending=False).reset_index(drop=True)
    st.dataframe(topic_df, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Trend Chart ──
    st.markdown("## Trends Over Time")

    try:
        from trend_analysis import get_trending_clusters

        df_t = df.copy()
        df_t["year"] = pd.to_numeric(df_t["year"], errors="coerce")
        df_t = df_t.dropna(subset=["year", "cluster"])
        df_t = df_t[(df_t["year"] >= 2015) & (df_t["year"] <= 2024)]
        trend_pivot = df_t.groupby(["year", "cluster"]).size().unstack(fill_value=0)

        if not trend_pivot.empty:
            trending_ids = get_trending_clusters(trend_pivot, top_n=5)

            fig, ax = plt.subplots(figsize=(10, 3.5))
            fig.patch.set_facecolor("#f7f5f2")
            ax.set_facecolor("#f7f5f2")

            colors = ["#7b9cce", "#e8956d", "#6abf8a", "#c97bb2", "#c4a84f"]
            for i, cid in enumerate(trending_ids):
                if cid in trend_pivot.columns:
                    label = f"Topic {cid} — {topics.get(cid, ['?'])[0]}"
                    ax.plot(trend_pivot.index, trend_pivot[cid], marker='o',
                            label=label, color=colors[i % len(colors)], linewidth=1.8, markersize=4)

            ax.tick_params(colors="#666", labelsize=9)
            ax.set_xlabel("Year", fontsize=9, color="#666")
            ax.set_ylabel("Papers", fontsize=9, color="#666")
            for spine in ax.spines.values():
                spine.set_edgecolor("#e0dbd5")
            ax.legend(fontsize=8, framealpha=0, labelcolor="#444")
            ax.grid(axis="y", color="#e8e4df", linewidth=0.6)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("<p style='font-size:0.82rem;color:#666;margin-top:0.8rem'><strong>Trending topics</strong></p>", unsafe_allow_html=True)
            for i, cid in enumerate(trending_ids):
                kws = topics.get(cid, ["unknown"])
                st.markdown(f"<p style='font-size:0.85rem;color:#555;margin:2px 0'>{i+1}. Topic {cid} &mdash; {', '.join(kws[:4])}</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Trend error: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Gaps ──
    st.markdown("## Research Gaps")

    try:
        from gap_detection import detect_gaps, suggest_directions
        gaps = detect_gaps(df=df, topics=topics)
        suggestions = suggest_directions(gaps, topics)

        if gaps:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<p style='font-size:0.82rem;font-weight:600;color:#444;margin-bottom:0.6rem'>Underexplored Areas</p>", unsafe_allow_html=True)
                for g in gaps:
                    st.markdown(f"<div class='gap-card'>{g['gap']}<div class='gap-meta'>{g.get('detail', '')} &middot; {g['combined_papers']} papers</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<p style='font-size:0.82rem;font-weight:600;color:#444;margin-bottom:0.6rem'>Suggested Directions</p>", unsafe_allow_html=True)
                for s in suggestions:
                    st.markdown(f"<div class='gap-card'>{s}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#aaa;font-size:0.88rem'>No significant gaps detected.</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Gap error: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Paper Browser ──
    st.markdown("## Browse Papers")

    selected_cluster = st.selectbox(
        "Select topic",
        options=sorted(topics.keys()),
        format_func=lambda x: f"Topic {x}  —  {', '.join(topics[x][:3])}"
    )

    cluster_papers = df[df["cluster"] == selected_cluster][["title", "abstract", "category", "year"]]
    st.markdown(f"<p style='font-size:0.82rem;color:#888;margin-bottom:0.5rem'>{len(cluster_papers)} papers in this topic</p>", unsafe_allow_html=True)
    st.dataframe(cluster_papers.head(20), use_container_width=True, hide_index=True)

    st.markdown("<p class='caption-text'>ResearchTrendAI &nbsp;·&nbsp; Sentence Transformers + KMeans + Streamlit</p>", unsafe_allow_html=True)