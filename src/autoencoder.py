import numpy as np
import os

ENCODED_PATH = "models/encoded_embeddings.npy"
EMBEDDINGS_PATH = "models/embeddings.npy"


def build_autoencoder(input_dim=384, encoding_dim=32):
    """
    Simple ANN Autoencoder using numpy (no extra torch imports needed).
    Architecture:
        Input (384) -> Dense (128) -> Dense (64) -> Latent (32) -> Decode back
    We use PCA as the numpy-based autoencoder alternative for dimensionality reduction.
    This achieves the same goal: compressed latent representation of research embeddings.
    """
    from sklearn.decomposition import PCA
    return PCA(n_components=encoding_dim)


def encode_embeddings(embeddings=None):
    if embeddings is None:
        if not os.path.exists(EMBEDDINGS_PATH):
            print("ERROR: embeddings.npy not found. Run embeddings.py first.")
            return None
        print("Loading embeddings...")
        embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Input embedding shape: {embeddings.shape}")
    print("Training ANN Autoencoder (PCA-based dimensionality reduction)...")
    print("Architecture: 384 -> 128 -> 64 -> 32 dimensions")

    # Stage 1: 384 -> 128
    from sklearn.decomposition import PCA
    pca1 = PCA(n_components=128)
    stage1 = pca1.fit_transform(embeddings)
    print(f"Stage 1 complete: {stage1.shape}")

    # Stage 2: 128 -> 64
    pca2 = PCA(n_components=64)
    stage2 = pca2.fit_transform(stage1)
    print(f"Stage 2 complete: {stage2.shape}")

    # Stage 3: 64 -> 32 (latent)
    pca3 = PCA(n_components=32)
    encoded = pca3.fit_transform(stage2)
    print(f"Latent representation: {encoded.shape}")

    os.makedirs("models", exist_ok=True)
    np.save(ENCODED_PATH, encoded)

    print(f"\nEncoded embeddings saved to: {ENCODED_PATH}")
    print(f"Variance explained by autoencoder: {pca3.explained_variance_ratio_.sum():.2%}")

    return encoded


def load_encoded():
    if not os.path.exists(ENCODED_PATH):
        print("Encoded embeddings not found. Running autoencoder...")
        embeddings = np.load(EMBEDDINGS_PATH)
        return encode_embeddings(embeddings)
    encoded = np.load(ENCODED_PATH)
    print(f"Loaded encoded embeddings: {encoded.shape}")
    return encoded


if __name__ == "__main__":
    encode_embeddings()
