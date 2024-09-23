"""Citation network visualization with node sizes based on citation count."""
from typing import Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_citation_network(
    projection: np.ndarray,
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    similarity_threshold: float = 0.7,
    max_edges: int = 2000,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Citation Network - Node Size = Citation Count",
) -> go.Figure:
    """
    Create interactive network visualization where:
    - Node position: UMAP projection
    - Node size: Citation count (log-scaled)
    - Node color: Cluster or year
    - Edges: High-similarity connections
    """
    n_papers = len(projection)

    citation_counts = metadata.get("citation_count", pd.Series([1] * n_papers)).fillna(1)
    citation_counts = citation_counts.replace(0, 1)

    min_size, max_size = 5, 50
    log_citations = np.log1p(citation_counts)
    normalized = (log_citations - log_citations.min()) / (log_citations.max() - log_citations.min() + 1e-8)
    node_sizes = min_size + normalized * (max_size - min_size)

    print(f"Computing similarity edges (threshold={similarity_threshold})...")
    similarities = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(similarities, 0)

    edge_indices = np.where(similarities > similarity_threshold)
    edge_weights = similarities[edge_indices]

    if len(edge_weights) > max_edges * 2:
        top_indices = np.argsort(edge_weights)[-max_edges * 2:]
        edge_indices = (edge_indices[0][top_indices], edge_indices[1][top_indices])
        edge_weights = edge_weights[top_indices]

    edge_x = []
    edge_y = []
    for i, j in zip(edge_indices[0], edge_indices[1]):
        if i < j:
            edge_x.extend([projection[i, 0], projection[j, 0], None])
            edge_y.extend([projection[i, 1], projection[j, 1], None])

    print(f"Creating network with {n_papers} nodes and {len(edge_x)//3} edges...")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=0.3, color='rgba(150,150,150,0.3)'),
        hoverinfo='none',
        name='Similarity Links'
    ))

    years = pd.to_datetime(metadata["published"]).dt.year
    color_values = years.fillna(years.median())

    hover_text = [
        f"<b>{row['title'][:60]}...</b><br>"
        f"ArXiv: {row['arxiv_id']}<br>"
        f"Citations: {int(row.get('citation_count', 0))}<br>"
        f"Year: {str(row['published'])[:4]}<br>"
        f"Categories: {row['categories'][:50]}"
        for _, row in metadata.iterrows()
    ]

    fig.add_trace(go.Scatter(
        x=projection[:, 0],
        y=projection[:, 1],
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=color_values,
            colorscale='Viridis',
            colorbar=dict(title="Year", thickness=15),
            line=dict(width=0.5, color='white'),
            opacity=0.8,
        ),
        text=hover_text,
        hoverinfo='text',
        name='Papers'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
        plot_bgcolor='rgba(250,250,250,1)',
        width=1400,
        height=900,
        annotations=[
            dict(
                text=f"Node size ∝ log(citations) | {n_papers} papers | {len(edge_x)//3} similarity links",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.02,
                font=dict(size=12, color="gray")
            )
        ]
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved network visualization to {save_path}")

    return fig


def create_citation_scatter(
    projection: np.ndarray,
    metadata: pd.DataFrame,
    cluster_labels: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create scatter plot with citation-sized markers and cluster colors."""
    citation_counts = metadata.get("citation_count", pd.Series([1] * len(metadata))).fillna(1)
    citation_counts = citation_counts.replace(0, 1)

    min_size, max_size = 4, 40
    log_citations = np.log1p(citation_counts)
    normalized = (log_citations - log_citations.min()) / (log_citations.max() - log_citations.min() + 1e-8)
    node_sizes = min_size + normalized * (max_size - min_size)

    if cluster_labels is not None:
        color_values = cluster_labels
        colorscale = 'Portland'
        colorbar_title = 'Cluster'
    else:
        years = pd.to_datetime(metadata["published"]).dt.year
        color_values = years.fillna(years.median())
        colorscale = 'Viridis'
        colorbar_title = 'Year'

    hover_text = [
        f"<b>{row['title'][:50]}...</b><br>"
        f"Citations: {int(row.get('citation_count', 0))}<br>"
        f"ArXiv: {row['arxiv_id']}"
        for _, row in metadata.iterrows()
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=projection[:, 0],
        y=projection[:, 1],
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=color_values,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title, thickness=15),
            line=dict(width=0.3, color='darkgray'),
            opacity=0.75,
        ),
        text=hover_text,
        hoverinfo='text',
    ))

    fig.update_layout(
        title=dict(
            text="Plasma Physics Papers - Size = Citations, Color = Cluster",
            font=dict(size=16)
        ),
        xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, title='UMAP 1'),
        yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, title='UMAP 2'),
        plot_bgcolor='white',
        width=1200,
        height=800,
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved citation scatter to {save_path}")

    return fig


def create_citation_dashboard(
    projection: np.ndarray,
    metadata: pd.DataFrame,
    cluster_labels: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create a multi-panel dashboard with citation statistics."""
    citation_counts = metadata.get("citation_count", pd.Series([0] * len(metadata))).fillna(0)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Paper Space (size=citations)",
            "Citation Distribution",
            "Citations by Year",
            "Top Cited Papers"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "box"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    min_size, max_size = 3, 25
    log_citations = np.log1p(citation_counts)
    normalized = (log_citations - log_citations.min()) / (log_citations.max() - log_citations.min() + 1e-8)
    node_sizes = min_size + normalized * (max_size - min_size)

    fig.add_trace(
        go.Scatter(
            x=projection[:, 0],
            y=projection[:, 1],
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=cluster_labels,
                colorscale='Portland',
                opacity=0.7,
            ),
            hovertext=metadata["title"].str[:40] + "...",
            hoverinfo='text',
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Histogram(
            x=np.log1p(citation_counts),
            nbinsx=50,
            marker_color='steelblue',
            name='Citations',
        ),
        row=1, col=2
    )

    years = pd.to_datetime(metadata["published"]).dt.year
    for year in sorted(years.unique())[-8:]:
        year_citations = citation_counts[years == year]
        fig.add_trace(
            go.Box(y=year_citations, name=str(year), marker_color='steelblue'),
            row=2, col=1
        )

    top_n = 15
    top_indices = citation_counts.nlargest(top_n).index
    top_titles = metadata.loc[top_indices, "title"].str[:30] + "..."
    top_citations = citation_counts.loc[top_indices]

    fig.add_trace(
        go.Bar(
            y=top_titles[::-1],
            x=top_citations[::-1],
            orientation='h',
            marker_color='steelblue',
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=900,
        width=1400,
        showlegend=False,
        title_text="Citation Analysis Dashboard",
        title_font_size=18,
    )

    fig.update_xaxes(title_text="UMAP 1", row=1, col=1)
    fig.update_yaxes(title_text="UMAP 2", row=1, col=1)
    fig.update_xaxes(title_text="log(1 + citations)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Citations", row=2, col=1)
    fig.update_xaxes(title_text="Citations", row=2, col=2)

    if save_path:
        fig.write_html(save_path)
        print(f"Saved dashboard to {save_path}")

    return fig
