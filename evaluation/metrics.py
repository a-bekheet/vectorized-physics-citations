"""Evaluation metrics for the citation search system."""
import time
from typing import Optional
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def compute_silhouette_score(
    embeddings: np.ndarray,
    n_clusters: int = 8,
    sample_size: Optional[int] = 5000,
) -> float:
    """Compute silhouette score for clustering quality."""
    if sample_size and len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    score = silhouette_score(embeddings, labels, metric="cosine")
    return score


def compute_cluster_stats(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    n_clusters: int = 8,
) -> dict:
    """Compute statistics for each cluster."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    stats = {}
    for i in range(n_clusters):
        mask = labels == i
        cluster_df = metadata[mask]

        all_categories = []
        for cats in cluster_df["categories"].str.split("|"):
            if cats:
                all_categories.extend(cats)

        top_categories = Counter(all_categories).most_common(5)

        stats[f"cluster_{i}"] = {
            "size": int(mask.sum()),
            "percentage": float(mask.sum() / len(labels) * 100),
            "top_categories": top_categories,
            "sample_titles": cluster_df["title"].head(3).tolist(),
        }

    return stats


def evaluate_search_quality(
    search_index,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    n_queries: int = 100,
    top_k: int = 10,
) -> dict:
    """Evaluate search quality and performance."""
    query_indices = np.random.choice(len(embeddings), n_queries, replace=False)

    latencies = []
    category_overlaps = []

    for idx in query_indices:
        query_embedding = embeddings[idx]
        query_categories = set(metadata.iloc[idx]["categories"].split("|"))

        start = time.time()
        result_indices, scores = search_index.search(query_embedding, top_k + 1)
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        result_indices = result_indices[result_indices != idx][:top_k]

        overlaps = []
        for res_idx in result_indices:
            res_categories = set(metadata.iloc[res_idx]["categories"].split("|"))
            overlap = len(query_categories & res_categories) / len(
                query_categories | res_categories
            )
            overlaps.append(overlap)

        if overlaps:
            category_overlaps.append(np.mean(overlaps))

    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "mean_category_overlap": float(np.mean(category_overlaps)),
        "std_category_overlap": float(np.std(category_overlaps)),
    }


def compute_embedding_stats(embeddings: np.ndarray) -> dict:
    """Compute statistics about the embeddings."""
    norms = np.linalg.norm(embeddings, axis=1)

    sample_size = min(1000, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sample = embeddings[sample_indices]
    similarities = np.dot(sample, sample.T)
    np.fill_diagonal(similarities, 0)
    pairwise_sims = similarities[np.triu_indices(sample_size, k=1)]

    return {
        "n_embeddings": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "mean_pairwise_similarity": float(np.mean(pairwise_sims)),
        "std_pairwise_similarity": float(np.std(pairwise_sims)),
        "min_pairwise_similarity": float(np.min(pairwise_sims)),
        "max_pairwise_similarity": float(np.max(pairwise_sims)),
    }
