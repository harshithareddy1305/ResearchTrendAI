#  ResearchTrendAI
**Intelligent Research Trend & Gap Analyzer**

An AI-powered system that analyzes thousands of research paper abstracts to identify major topics, detect trending research areas, and find underexplored research gaps.

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| NLP Preprocessing | NLTK, regex |
| Neural Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| ANN Dimensionality Reduction | PCA-based Autoencoder (384→128→64→32) |
| Topic Modeling | KMeans Clustering + TF-IDF |
| Trend Detection | Pandas time-series analysis |
| Gap Detection | Cluster sparsity analysis |
| Dashboard | Streamlit |

---

## 📁 Folder Structure

```
ResearchTrendAI/
│
├── data/                        # Datasets
│   ├── arxiv-metadata-oai-snapshot.json   ← you download this
│   ├── research_papers.csv
│   ├── processed_papers.csv
│   ├── clustered_papers.csv
│   └── trends.csv
│
├── models/                      # Saved ML artifacts
│   ├── embeddings.npy
│   ├── encoded_embeddings.npy
│   ├── cluster_labels.npy
│   ├── tfidf_vectorizer.pkl
│   └── topics.pkl
│
├── src/
│   ├── data_loader.py           # Step 0: Convert JSON → CSV
│   ├── preprocessing.py         # Step 1: NLP cleaning
│   ├── embeddings.py            # Step 2: Neural embeddings
│   ├── autoencoder.py           # Step 3: ANN dimensionality reduction
│   ├── topic_modeling.py        # Step 4: Clustering + topic discovery
│   ├── trend_analysis.py        # Step 5: Trend detection
│   └── gap_detection.py         # Step 6: Research gap detection
│
├── app.py                       # Streamlit dashboard
├── run_pipeline.py              # One-shot pipeline runner
└── requirements.txt
```

---

## Setup & Running

### 1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
- Go to: https://www.kaggle.com/datasets/Cornell-University/arxiv
- Download: `arxiv-metadata-oai-snapshot.json`
- Place it inside: `data/`

### 4. Convert dataset
```bash
python src/data_loader.py
```

### 5. Run full pipeline
```bash
python run_pipeline.py
```

### 6. Launch dashboard
```bash
streamlit run app.py
```

---

## What we get to See

- **Top Research Topics** — clustered from paper abstracts
- **Trending Topics** — growth rate detection across years
- **Research Gaps** — sparse cluster combinations
- **Research Direction Suggestions** — AI-generated ideas
- **Paper Browser** — explore papers by topic cluster
