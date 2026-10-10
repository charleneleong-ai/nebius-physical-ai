"""Best-of-N action chunks without a learned verifier: a flow-matching policy draws a different chunk per noise seed, and
the one closest to all the others (the medoid) is the consensus. RoboMonkey (arXiv 2506.17811) samples and picks with a
trained VLM verifier; this is its verifier-free first step, chosen because it needs no reward model."""

import numpy as np


def medoid(chunks: np.ndarray) -> np.ndarray:
    """(N, B, T, D) candidate chunks -> (B, T, D): per batch row, the candidate with the least summed distance to the
    other candidates (Euclidean over the whole chunk)."""
    flat = chunks.reshape(chunks.shape[0], chunks.shape[1], -1)  # (N, B, T*D)
    distances = np.linalg.norm(flat[:, None] - flat[None], axis=-1).sum(axis=1)  # (N, B): summed over the others
    best = distances.argmin(axis=0)  # (B,)
    return chunks[best, np.arange(chunks.shape[1])]
