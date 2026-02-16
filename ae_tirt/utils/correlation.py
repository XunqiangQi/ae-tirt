"""Correlation matrix helpers."""

from __future__ import annotations

import numpy as np


def is_positive_definite(mat: np.ndarray | list, tol: float = 1e-10) -> bool:
    """Check if matrix is positive definite (eigenvalues > tol)."""
    mat = np.asarray(mat, dtype=float)
    return bool(np.all(np.linalg.eigvalsh(mat) > tol))


def sample_random_correlation_matrix(
    ntraits: int,
    low: float = -0.5,
    high: float = 0.5,
    max_tries: int = 50,
    epsilon: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a valid positive-definite correlation matrix."""
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(max_tries):
        mat = rng.uniform(low, high, size=(ntraits, ntraits))
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, 1.0)

        min_eig = np.linalg.eigvalsh(mat).min()
        if min_eig <= 0:
            mat += np.eye(ntraits) * (-min_eig + epsilon)

        d = np.sqrt(np.diag(mat))
        mat = mat / d[:, None] / d[None, :]
        mat = np.clip(mat, -0.999, 0.999)
        np.fill_diagonal(mat, 1.0)

        if np.linalg.eigvalsh(mat).min() > 0:
            return mat

    raise RuntimeError("Failed to sample a positive-definite correlation matrix.")
