import pandas as pd
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download all required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove non-alphabetical characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Remove very short tokens
    tokens = [word for word in tokens if len(word) > 2]

    return " ".join(tokens)


def preprocess_dataset():
    input_path = "data/research_papers.csv"
    output_path = "data/processed_papers.csv"

    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found. Please run src/data_loader.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(input_path)

    print(f"Total papers loaded: {len(df)}")

    # Drop rows with missing abstracts
    df = df.dropna(subset=["abstract"])
    df = df[df["abstract"].str.strip() != ""]

    print("Cleaning and preprocessing abstracts...")
    df["clean_abstract"] = df["abstract"].apply(clean_text)

    # Save
    df.to_csv(output_path, index=False)

    print(f"\nPreprocessing completed!")
    print(f"Saved to: {output_path}")
    print(f"Total processed papers: {len(df)}")

    # Preview
    print("\nSample comparison:")
    sample = df[['abstract', 'clean_abstract']].head(2)
    for i, row in sample.iterrows():
        print(f"\nOriginal: {row['abstract'][:120]}...")
        print(f"Cleaned:  {row['clean_abstract'][:120]}...")


if __name__ == "__main__":
    preprocess_dataset()
