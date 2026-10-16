"""Best-of-N action chunks: a flow-matching policy draws a different chunk per noise seed, and one is executed.

`medoid_index` picks the consensus (the chunk closest to all the others), RoboMonkey's (arXiv 2506.17811) verifier-free
first step. The rest are the torch-free parts of the learned verifier (`policy.verifier`, after DiVeR, arXiv
2610.04933): each candidate's features, and how much a step's candidates disagree."""

import numpy as np


def medoid_index(chunks: np.ndarray) -> np.ndarray:
    """(N, B, T, D) candidate chunks -> (B,): per batch row, the candidate with the least summed distance to the
    other candidates (Euclidean over the whole chunk)."""
    flat = chunks.reshape(chunks.shape[0], chunks.shape[1], -1)  # (N, B, T*D)
    return np.linalg.norm(flat[:, None] - flat[None], axis=-1).sum(axis=1).argmin(axis=0)


def candidate_features(
    features: np.ndarray, joints: np.ndarray, chunks: np.ndarray, expert: np.ndarray | None = None
) -> np.ndarray:
    """(B, D) pooled observation features, (B, 6) joints and (B, N, T, 6) candidate chunks -> (B, N, D + 6 + T*6):
    one row per candidate, the chunk as offsets from the current joints. `expert` (B, N, E), the action expert's own
    pooled state for each candidate, is appended when given: it differs per candidate where the shared observation
    features cannot."""
    batch, n = chunks.shape[:2]
    offsets = (chunks - joints[:, None, None, :]).reshape(batch, n, -1)
    shared = np.concatenate([features, joints], axis=-1)[:, None].repeat(n, axis=1)
    parts = [shared, offsets] if expert is None else [shared, offsets, expert]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def criticality(chunks: np.ndarray) -> np.ndarray:
    """(B, N, T, 6) candidates -> (B,) mean per-coordinate variance across the N candidates (DiVeR eq. 8)."""
    flat = chunks.reshape(chunks.shape[0], chunks.shape[1], -1)
    return flat.var(axis=1).mean(axis=-1)


def decision_weights(scores: np.ndarray, beta: float) -> np.ndarray:
    """One trajectory's criticality scores -> positive weights exp(beta * standardised score) (DiVeR eq. 9)."""
    spread = scores.std()
    return np.exp(beta * (scores - scores.mean()) / spread) if spread > 0 else np.ones_like(scores)
