import json
import pandas as pd
import os

# Path to the downloaded arxiv JSON file
# Place your arxiv-metadata-oai-snapshot.json inside the data/ folder
data_file = "data/arxiv-metadata-oai-snapshot.json"

def load_arxiv_data(limit=5000):
    papers = []

    print(f"Loading data from {data_file} ...")

    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            paper = json.loads(line)

            # Extract year from update_date e.g. "2021-03-15" -> "2021"
            update_date = paper.get("update_date", "")
            year = update_date[:4] if update_date else "unknown"

            papers.append({
                "title": paper.get("title", "").replace("\n", " ").strip(),
                "abstract": paper.get("abstract", "").replace("\n", " ").strip(),
                "category": paper.get("categories", ""),
                "year": year
            })

            if i >= limit - 1:
                break

    df = pd.DataFrame(papers)

    # Drop rows where title or abstract is empty
    df = df.dropna(subset=["title", "abstract"])
    df = df[df["abstract"].str.strip() != ""]

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/research_papers.csv", index=False)

    print(f"Dataset created with {len(df)} papers!")
    print(df.head(3))
    return df


if __name__ == "__main__":
    load_arxiv_data(limit=5000)
