"""Visualization utilities for paper embeddings."""
from typing import Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
from sklearn.cluster import KMeans


def create_umap_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Project embeddings to 2D using UMAP."""
    print(f"Computing UMAP projection ({embeddings.shape[0]} vectors)...")
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    projection = reducer.fit_transform(embeddings)
    print(f"UMAP projection complete: {projection.shape}")
    return projection


def plot_embedding_space(
    projection: np.ndarray,
    labels: Optional[np.ndarray] = None,
    titles: Optional[list[str]] = None,
    citation_counts: Optional[np.ndarray] = None,
    figsize: tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Paper Embedding Space",
    alpha: float = 0.7,
    s: int = 15,
) -> plt.Figure:
    """Create static scatter plot of embedding space with citation-based sizing."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Citation-based sizing
    if citation_counts is not None:
        citation_counts = np.array(citation_counts)
        citation_counts = np.where(citation_counts < 1, 1, citation_counts)
        min_size, max_size = 8, 200
        log_citations = np.log1p(citation_counts)
        normalized = (log_citations - log_citations.min()) / (log_citations.max() - log_citations.min() + 1e-8)
        sizes = min_size + normalized * (max_size - min_size)
    else:
        sizes = s

    if labels is not None:
        scatter = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            c=labels,
            cmap="Set2",
            alpha=alpha,
            s=sizes,
            edgecolors='white',
            linewidths=0.3,
        )
        cbar = plt.colorbar(scatter, ax=ax, label="Cluster", shrink=0.8)
        cbar.ax.tick_params(labelsize=9)
    else:
        ax.scatter(
            projection[:, 0],
            projection[:, 1],
            alpha=alpha,
            s=sizes,
            c="steelblue",
            edgecolors='white',
            linewidths=0.3,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)

    # Add subtitle
    if citation_counts is not None:
        ax.text(0.5, -0.08, f"Node size ~ log(citations) | {len(projection)} papers",
                transform=ax.transAxes, ha='center', fontsize=10, color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


def plot_paper_clusters(
    projection: np.ndarray,
    n_clusters: int = 8,
    citation_counts: Optional[np.ndarray] = None,
    figsize: tuple[int, int] = (14, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Cluster papers and visualize with citation-based sizing."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(projection)

    fig = plot_embedding_space(
        projection,
        labels=labels,
        citation_counts=citation_counts,
        figsize=figsize,
        save_path=save_path,
        title=f"Plasma Physics Paper Clusters (k={n_clusters})",
    )

    return fig, labels


def create_interactive_plot(
    projection: np.ndarray,
    metadata: pd.DataFrame,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Interactive Paper Space",
) -> go.Figure:
    """Create interactive Plotly visualization with citation-based node sizing."""
    df = metadata.copy()
    df["x"] = projection[:, 0]
    df["y"] = projection[:, 1]

    # Citation-based node sizing
    citation_counts = df.get("citation_count", pd.Series([1] * len(df))).fillna(1)
    citation_counts = citation_counts.replace(0, 1)
    min_size, max_size = 6, 45
    log_citations = np.log1p(citation_counts)
    normalized = (log_citations - log_citations.min()) / (log_citations.max() - log_citations.min() + 1e-8)
    df["node_size"] = min_size + normalized * (max_size - min_size)

    if labels is not None:
        df["cluster"] = labels.astype(str)

    df["title_short"] = df["title"].str[:80] + "..."
    df["abstract_short"] = df["abstract"].str[:200] + "..."
    df["citation_display"] = citation_counts.astype(int)

    # Build custom hover text
    hover_text = [
        f"<b>{row['title'][:70]}...</b><br><br>"
        f"<b>ArXiv:</b> {row['arxiv_id']}<br>"
        f"<b>Citations:</b> {int(row.get('citation_count', 0))}<br>"
        f"<b>Published:</b> {str(row['published'])[:10]}<br>"
        f"<b>Cluster:</b> {row.get('cluster', 'N/A') if labels is not None else 'N/A'}<br><br>"
        f"<i>{row['abstract'][:250]}...</i>"
        for _, row in df.iterrows()
    ]

    fig = go.Figure()

    if labels is not None:
        # Color by cluster with distinct palette
        cluster_colors = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1
        unique_clusters = sorted(df["cluster"].unique(), key=lambda x: int(x))

        for i, cluster in enumerate(unique_clusters):
            mask = df["cluster"] == cluster
            cluster_df = df[mask]
            cluster_hover = [hover_text[j] for j in df.index[mask]]

            fig.add_trace(go.Scatter(
                x=cluster_df["x"],
                y=cluster_df["y"],
                mode='markers',
                marker=dict(
                    size=cluster_df["node_size"],
                    color=cluster_colors[i % len(cluster_colors)],
                    line=dict(width=0.5, color='white'),
                    opacity=0.8,
                ),
                text=cluster_hover,
                hoverinfo='text',
                name=f'Cluster {cluster}',
            ))
    else:
        years = pd.to_datetime(df["published"]).dt.year
        fig.add_trace(go.Scatter(
            x=df["x"],
            y=df["y"],
            mode='markers',
            marker=dict(
                size=df["node_size"],
                color=years,
                colorscale='Viridis',
                colorbar=dict(title="Year", thickness=15),
                line=dict(width=0.5, color='white'),
                opacity=0.8,
            ),
            text=hover_text,
            hoverinfo='text',
            name='Papers',
        ))

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>Node size ~ log(citations) | {len(df)} papers</sup>",
            font=dict(size=18),
            x=0.5,
        ),
        width=1400,
        height=900,
        font=dict(size=12, family="Arial"),
        legend=dict(
            title="Clusters",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        xaxis=dict(
            title="UMAP Dimension 1",
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            zeroline=False,
        ),
        yaxis=dict(
            title="UMAP Dimension 2",
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            zeroline=False,
        ),
        plot_bgcolor='rgba(250,250,252,1)',
        hovermode='closest',
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved interactive plot to {save_path}")

    return fig


def plot_similarity_distribution(
    similarities: np.ndarray,
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot distribution of similarity scores."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    ax.hist(similarities, bins=50, edgecolor="white", alpha=0.7, color="steelblue")
    ax.axvline(
        similarities.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {similarities.mean():.3f}",
    )
    ax.axvline(
        np.median(similarities),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(similarities):.3f}",
    )

    ax.set_title("Similarity Score Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


def plot_query_results(
    query_text: str,
    results: list[dict],
    projection: np.ndarray,
    query_idx: Optional[int] = None,
    figsize: tuple[int, int] = (14, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Visualize search query results."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150)

    ax1 = axes[0]
    ax1.scatter(
        projection[:, 0], projection[:, 1],
        alpha=0.3, s=5, c="gray", label="All papers"
    )

    result_indices = [r["index"] for r in results]
    ax1.scatter(
        projection[result_indices, 0],
        projection[result_indices, 1],
        c="red", s=50, marker="*", label="Search results", zorder=5
    )

    if query_idx is not None:
        ax1.scatter(
            projection[query_idx, 0],
            projection[query_idx, 1],
            c="blue", s=100, marker="X", label="Query paper", zorder=10
        )

    ax1.set_title("Search Results in Embedding Space", fontsize=12, fontweight="bold")
    ax1.set_xlabel("UMAP 1", fontsize=10)
    ax1.set_ylabel("UMAP 2", fontsize=10)
    ax1.legend(fontsize=9)

    ax2 = axes[1]
    scores = [r["score"] for r in results]
    titles = [r["title"][:40] + "..." for r in results]
    colors = plt.cm.RdYlGn(np.array(scores))

    bars = ax2.barh(range(len(results)), scores, color=colors)
    ax2.set_yticks(range(len(results)))
    ax2.set_yticklabels(titles, fontsize=8)
    ax2.set_xlabel("Similarity Score", fontsize=10)
    ax2.set_title("Top Results by Similarity", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()

    for bar, score in zip(bars, scores):
        ax2.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", fontsize=8
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig
