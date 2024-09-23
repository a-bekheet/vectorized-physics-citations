"""Command-line interface for paper search and exploration."""
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from data import ArxivFetcher, TextPreprocessor
from data.preprocessor import preprocess_papers
from models import PaperEmbedder, SearchIndex
from utils import load_config


console = Console()


def load_data(config_path: str = "configs/default.yaml"):
    """Load embeddings, metadata, and search index."""
    config = load_config(config_path)

    metadata_path = Path(config["paths"]["metadata_file"])
    embeddings_path = Path(config["paths"]["embeddings_file"])
    index_path = Path(config["paths"]["index_dir"]) / "search.index"

    if not all(p.exists() for p in [metadata_path, embeddings_path, index_path]):
        console.print(
            "[red]Data not found. Run 'python experiments/run_experiment.py' first.[/red]"
        )
        sys.exit(1)

    fetcher = ArxivFetcher.load(str(metadata_path))
    metadata = fetcher.to_dataframe()
    embeddings = PaperEmbedder.load_embeddings(str(embeddings_path))

    index = SearchIndex(dimension=embeddings.shape[1])
    index.load(str(index_path))

    return config, metadata, embeddings, index, fetcher


@click.group()
def cli():
    """Vectorized Physics Citations - Semantic search for ArXiv papers."""
    pass


@cli.command()
@click.argument("query", type=str)
@click.option("-k", "--top-k", default=10, help="Number of results to return")
@click.option("-c", "--config", default="configs/default.yaml", help="Config file path")
def search(query: str, top_k: int, config: str):
    """Search for similar papers using a text query."""
    console.print(f"\n[bold]Searching for:[/bold] {query}\n")

    cfg, metadata, embeddings, index, _ = load_data(config)

    embedder = PaperEmbedder(
        model_name=cfg["embedding"]["model_name"],
        device=cfg["embedding"]["device"],
    )

    query_embedding = embedder.embed_single(query, normalize=True)

    indices, scores = index.search(query_embedding, top_k)

    table = Table(title=f"Top {top_k} Results", show_lines=True)
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("ArXiv ID", style="green", width=12)
    table.add_column("Score", style="yellow", width=8)
    table.add_column("Title", style="white")

    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        row = metadata.iloc[idx]
        table.add_row(
            str(rank),
            row["arxiv_id"],
            f"{score:.4f}",
            row["title"][:80] + "..." if len(row["title"]) > 80 else row["title"],
        )

    console.print(table)


@cli.command()
@click.argument("arxiv_id", type=str)
@click.option("-k", "--top-k", default=10, help="Number of similar papers")
@click.option("-c", "--config", default="configs/default.yaml", help="Config file path")
def similar(arxiv_id: str, top_k: int, config: str):
    """Find papers similar to a given ArXiv ID."""
    cfg, metadata, embeddings, index, _ = load_data(config)

    matches = metadata[metadata["arxiv_id"] == arxiv_id]
    if matches.empty:
        console.print(f"[red]Paper {arxiv_id} not found in database.[/red]")
        return

    paper_idx = matches.index[0]
    paper = metadata.iloc[paper_idx]

    console.print(Panel(
        f"[bold]{paper['title']}[/bold]\n\n"
        f"[dim]ArXiv ID:[/dim] {paper['arxiv_id']}\n"
        f"[dim]Published:[/dim] {paper['published'][:10]}\n"
        f"[dim]Categories:[/dim] {paper['categories']}",
        title="Query Paper",
        border_style="blue",
    ))

    query_embedding = embeddings[paper_idx]
    indices, scores = index.search(query_embedding, top_k + 1)

    mask = indices != paper_idx
    indices = indices[mask][:top_k]
    scores = scores[mask][:top_k]

    table = Table(title=f"Papers Similar to {arxiv_id}", show_lines=True)
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("ArXiv ID", style="green", width=12)
    table.add_column("Score", style="yellow", width=8)
    table.add_column("Title", style="white")

    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        row = metadata.iloc[idx]
        table.add_row(
            str(rank),
            row["arxiv_id"],
            f"{score:.4f}",
            row["title"][:80] + "..." if len(row["title"]) > 80 else row["title"],
        )

    console.print(table)


@cli.command()
@click.argument("arxiv_id", type=str)
@click.option("-c", "--config", default="configs/default.yaml", help="Config file path")
def info(arxiv_id: str, config: str):
    """Show detailed information about a paper."""
    _, metadata, _, _, _ = load_data(config)

    matches = metadata[metadata["arxiv_id"] == arxiv_id]
    if matches.empty:
        console.print(f"[red]Paper {arxiv_id} not found in database.[/red]")
        return

    paper = matches.iloc[0]

    console.print(Panel(
        f"[bold]{paper['title']}[/bold]\n\n"
        f"[dim]ArXiv ID:[/dim] {paper['arxiv_id']}\n"
        f"[dim]Published:[/dim] {paper['published'][:10]}\n"
        f"[dim]Updated:[/dim] {paper['updated'][:10]}\n"
        f"[dim]Categories:[/dim] {paper['categories']}\n"
        f"[dim]Authors:[/dim] {paper['authors'][:100]}...\n\n"
        f"[bold]Abstract:[/bold]\n{paper['abstract'][:500]}...\n\n"
        f"[dim]PDF:[/dim] {paper['pdf_url']}",
        title=f"Paper Details: {arxiv_id}",
        border_style="blue",
    ))


@cli.command()
@click.option("-c", "--config", default="configs/default.yaml", help="Config file path")
def stats(config: str):
    """Show database statistics."""
    cfg, metadata, embeddings, index, _ = load_data(config)

    console.print(Panel(
        f"[bold]Database Statistics[/bold]\n\n"
        f"[dim]Total Papers:[/dim] {len(metadata):,}\n"
        f"[dim]Embedding Dimension:[/dim] {embeddings.shape[1]}\n"
        f"[dim]Index Size:[/dim] {index.size:,}\n"
        f"[dim]Category:[/dim] {cfg['arxiv']['category']}\n"
        f"[dim]Model:[/dim] {cfg['embedding']['model_name']}",
        title="Vectorized Physics Citations",
        border_style="green",
    ))

    years = pd.to_datetime(metadata["published"]).dt.year
    year_counts = years.value_counts().sort_index()

    table = Table(title="Papers by Year")
    table.add_column("Year", style="cyan")
    table.add_column("Count", style="yellow")

    for year, count in year_counts.tail(10).items():
        table.add_row(str(year), f"{count:,}")

    console.print(table)


@cli.command()
@click.option("-n", "--num-samples", default=5, help="Number of random papers")
@click.option("-c", "--config", default="configs/default.yaml", help="Config file path")
def random(num_samples: int, config: str):
    """Show random papers from the database."""
    _, metadata, _, _, _ = load_data(config)

    samples = metadata.sample(num_samples)

    table = Table(title=f"{num_samples} Random Papers", show_lines=True)
    table.add_column("ArXiv ID", style="green", width=12)
    table.add_column("Year", style="cyan", width=6)
    table.add_column("Title", style="white")

    for _, paper in samples.iterrows():
        year = paper["published"][:4]
        table.add_row(
            paper["arxiv_id"],
            year,
            paper["title"][:80] + "..." if len(paper["title"]) > 80 else paper["title"],
        )

    console.print(table)


if __name__ == "__main__":
    cli()
