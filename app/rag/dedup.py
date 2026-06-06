"""Deduplication helpers for the RAG layer.

Provides exact-dedup via a stable content hash and near-duplicate detection
via cosine similarity over embedding vectors.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

# Pre-compiled pattern collapsing any run of whitespace into a single space.
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for hashing/comparison.

    Lowercases, collapses all runs of whitespace to a single space, and strips
    leading/trailing whitespace. ``None``-ish/empty input yields an empty
    string.
    """
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def content_hash(title: str, content: str, url: str = "") -> str:
    """Return a stable sha256 hex digest for exact-dedup.

    The digest is computed from the normalized (lowercased, whitespace-
    collapsed, stripped) title, content, and optional url joined with a
    delimiter that cannot appear after normalization. The same logical
    document always produces the same hash regardless of incidental
    whitespace or letter casing.
    """
    parts = [
        normalize_text(title),
        normalize_text(content),
        normalize_text(url),
    ]
    # Newline is a safe separator: normalize_text collapses newlines to spaces,
    # so it never appears inside the parts themselves.
    joined = "\n".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def is_near_duplicate(
    vec_a: list[float],
    vec_b: list[float],
    threshold: float = 0.92,
) -> bool:
    """Return True if cosine similarity of two vectors is >= ``threshold``.

    Uses numpy. Zero-length or empty vectors, and any vector with zero norm,
    are treated as non-duplicates (similarity 0) to avoid division by zero.
    """
    if not vec_a or not vec_b:
        return False

    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)

    if a.shape != b.shape:
        return False

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return False

    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    return similarity >= threshold
