"""Main experiment script for paper embedding and visualization."""
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from data import ArxivFetcher, TextPreprocessor, enrich_metadata_with_citations
from data.preprocessor import preprocess_papers
from models import PaperEmbedder, SearchIndex
from evaluation.metrics import (
    compute_silhouette_score,
    compute_cluster_stats,
    evaluate_search_quality,
    compute_embedding_stats,
)
from utils import (
    load_config,
    create_umap_projection,
    plot_embedding_space,
    plot_paper_clusters,
    create_interactive_plot,
    create_citation_network,
    create_citation_scatter,
    create_citation_dashboard,
)
from utils.visualization import plot_similarity_distribution


def setup_directories(config: dict) -> None:
    """Create output directories."""
    for key in ["data_dir", "index_dir", "plots_dir"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def run_data_collection(config: dict) -> tuple[ArxivFetcher, pd.DataFrame]:
    """Fetch papers from ArXiv."""
    print("\n" + "=" * 60)
    print("PHASE 1: Data Collection")
    print("=" * 60)

    metadata_path = Path(config["paths"]["metadata_file"])

    if metadata_path.exists():
        print(f"Loading cached metadata from {metadata_path}")
        fetcher = ArxivFetcher.load(str(metadata_path))
        df = fetcher.to_dataframe()
    else:
        fetcher = ArxivFetcher(
            category=config["arxiv"]["category"],
            max_results=config["arxiv"]["max_results"],
            batch_size=config["arxiv"]["batch_size"],
            delay_seconds=config["arxiv"]["delay_seconds"],
            date_from=config["arxiv"].get("date_from"),
            date_to=config["arxiv"].get("date_to"),
        )
        fetcher.fetch_all(progress=True)
        fetcher.save(str(metadata_path))
        df = fetcher.to_dataframe()

    print(f"Total papers: {len(df)}")
    return fetcher, df


def run_citation_enrichment(
    config: dict,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch citation counts from Semantic Scholar."""
    print("\n" + "=" * 60)
    print("PHASE 1.5: Citation Enrichment (Semantic Scholar)")
    print("=" * 60)

    citation_cache = Path(config["paths"]["data_dir"]) / "citations.parquet"

    metadata_enriched, _ = enrich_metadata_with_citations(
        metadata,
        citation_cache_path=str(citation_cache),
    )

    total_citations = metadata_enriched["citation_count"].sum()
    max_citations = metadata_enriched["citation_count"].max()
    papers_with_citations = (metadata_enriched["citation_count"] > 0).sum()

    print(f"Papers with citation data: {papers_with_citations}/{len(metadata_enriched)}")
    print(f"Total citations: {total_citations:,}")
    print(f"Most cited paper: {max_citations:,} citations")

    return metadata_enriched


def run_embedding_generation(
    config: dict,
    fetcher: ArxivFetcher,
) -> np.ndarray:
    """Generate embeddings for all papers."""
    print("\n" + "=" * 60)
    print("PHASE 2: Embedding Generation")
    print("=" * 60)

    embeddings_path = Path(config["paths"]["embeddings_file"])

    if embeddings_path.exists():
        print(f"Loading cached embeddings from {embeddings_path}")
        embeddings = PaperEmbedder.load_embeddings(str(embeddings_path))
    else:
        preprocessor = TextPreprocessor(remove_latex=True)
        texts = preprocess_papers(fetcher.papers, preprocessor)

        embedder = PaperEmbedder(
            model_name=config["embedding"]["model_name"],
            batch_size=config["embedding"]["batch_size"],
            device=config["embedding"]["device"],
        )
        print(f"Using embedder: {embedder}")

        start = time.time()
        embeddings = embedder.embed(texts, normalize=True)
        elapsed = time.time() - start

        print(f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
        print(f"Embedding rate: {len(embeddings) / elapsed:.1f} papers/sec")

        embedder.save_embeddings(embeddings, str(embeddings_path))

    return embeddings


def run_index_building(
    config: dict,
    embeddings: np.ndarray,
) -> SearchIndex:
    """Build FAISS search index."""
    print("\n" + "=" * 60)
    print("PHASE 3: Index Building")
    print("=" * 60)

    index_path = Path(config["paths"]["index_dir"]) / "search.index"

    index = SearchIndex(
        dimension=embeddings.shape[1],
        index_type=config["search"]["index_type"],
        nlist=config["search"]["nlist"],
        nprobe=config["search"]["nprobe"],
    )

    if index_path.exists():
        print(f"Loading cached index from {index_path}")
        index.load(str(index_path))
    else:
        start = time.time()
        index.build(embeddings)
        elapsed = time.time() - start
        print(f"Built index in {elapsed:.2f}s")
        index.save(str(index_path))

    return index


def run_evaluation(
    config: dict,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    search_index: SearchIndex,
) -> dict:
    """Run evaluation metrics."""
    print("\n" + "=" * 60)
    print("PHASE 4: Evaluation")
    print("=" * 60)

    results = {}

    print("Computing embedding statistics...")
    results["embedding_stats"] = compute_embedding_stats(embeddings)

    print("Computing silhouette score...")
    results["silhouette_score"] = compute_silhouette_score(embeddings, n_clusters=8)
    print(f"Silhouette score (k=8): {results['silhouette_score']:.4f}")

    print("Evaluating search quality...")
    results["search_metrics"] = evaluate_search_quality(
        search_index, embeddings, metadata, n_queries=100
    )
    print(f"Mean search latency: {results['search_metrics']['mean_latency_ms']:.2f}ms")
    print(f"Mean category overlap: {results['search_metrics']['mean_category_overlap']:.4f}")

    print("Computing cluster statistics...")
    results["cluster_stats"] = compute_cluster_stats(embeddings, metadata, n_clusters=8)

    results_path = Path(config["paths"]["plots_dir"]) / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved evaluation results to {results_path}")

    return results


def run_visualization(
    config: dict,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate visualizations."""
    print("\n" + "=" * 60)
    print("PHASE 5: Visualization")
    print("=" * 60)

    plots_dir = Path(config["paths"]["plots_dir"])
    projection_path = plots_dir / "umap_projection.npy"

    if projection_path.exists():
        print(f"Loading cached projection from {projection_path}")
        projection = np.load(projection_path)
    else:
        projection = create_umap_projection(
            embeddings,
            n_neighbors=config["visualization"]["umap_neighbors"],
            min_dist=config["visualization"]["umap_min_dist"],
            metric=config["visualization"]["umap_metric"],
            random_state=config["visualization"]["random_state"],
        )
        np.save(projection_path, projection)

    # Get citation counts for sizing
    citation_counts = metadata.get("citation_count", pd.Series([1] * len(metadata))).fillna(1).values

    print("Generating static embedding space plot...")
    plot_embedding_space(
        projection,
        citation_counts=citation_counts,
        figsize=(12, 8),
        save_path=plots_dir / "embedding_space.png",
        title="Plasma Physics Paper Embedding Space",
    )

    print("Generating cluster visualization...")
    _, labels = plot_paper_clusters(
        projection,
        n_clusters=8,
        citation_counts=citation_counts,
        figsize=(14, 10),
        save_path=plots_dir / "paper_clusters.png",
    )

    print("Generating interactive plot...")
    create_interactive_plot(
        projection,
        metadata,
        labels=labels,
        save_path=plots_dir / "interactive_space.html",
        title="Interactive Plasma Physics Paper Space",
    )

    print("Generating citation network visualization...")
    create_citation_network(
        projection,
        metadata,
        embeddings,
        similarity_threshold=0.65,
        max_edges=1500,
        save_path=plots_dir / "citation_network.html",
        title="Citation Network - Node Size = Citation Count",
    )

    print("Generating citation scatter plot...")
    create_citation_scatter(
        projection,
        metadata,
        cluster_labels=labels,
        save_path=plots_dir / "citation_scatter.html",
    )

    print("Generating citation dashboard...")
    create_citation_dashboard(
        projection,
        metadata,
        cluster_labels=labels,
        save_path=plots_dir / "citation_dashboard.html",
    )

    sample_size = min(1000, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sample = embeddings[sample_indices]
    similarities = np.dot(sample, sample.T)
    np.fill_diagonal(similarities, 0)
    pairwise_sims = similarities[np.triu_indices(sample_size, k=1)]

    plot_similarity_distribution(
        pairwise_sims,
        save_path=plots_dir / "similarity_distribution.png",
    )

    return projection, labels


def main():
    """Run complete experiment pipeline."""
    print("=" * 60)
    print("VECTORIZED PHYSICS CITATIONS - EXPERIMENT RUNNER")
    print("=" * 60)

    config = load_config("configs/default.yaml")
    setup_directories(config)

    fetcher, metadata = run_data_collection(config)

    metadata = run_citation_enrichment(config, metadata)

    embeddings = run_embedding_generation(config, fetcher)

    search_index = run_index_building(config, embeddings)

    results = run_evaluation(config, embeddings, metadata, search_index)

    projection, labels = run_visualization(config, embeddings, metadata)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Total papers processed: {len(metadata)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Total citations in dataset: {metadata['citation_count'].sum():,}")
    print(f"Silhouette score: {results['silhouette_score']:.4f}")
    print(f"Mean search latency: {results['search_metrics']['mean_latency_ms']:.2f}ms")


if __name__ == "__main__":
    main()
