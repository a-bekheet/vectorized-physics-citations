"""Compare Python (sentence-transformers) and C++ embedder implementations."""
import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data import ArxivFetcher, TextPreprocessor
from data.preprocessor import preprocess_papers
from models import PaperEmbedder
from utils import load_config


class CppEmbedder:
    """C++ embedder wrapper matching PaperEmbedder interface."""

    def __init__(
        self,
        weights_path: str,
        vocab_path: str = None,
        batch_size: int = 64,
    ):
        self.weights_path = str(weights_path)
        self.vocab_path = str(vocab_path) if vocab_path else None
        self.batch_size = batch_size
        self.embedding_dim = 384

        # Load the shared library
        self._load_library()

    def _load_library(self):
        """Load the C++ shared library."""
        import ctypes

        # Find the library
        lib_paths = [
            Path(__file__).parent.parent / "cpp_embedder/build/lib/libcpp_embedder.dylib",
            Path(__file__).parent.parent / "cpp_embedder/build/lib/libcpp_embedder.so",
        ]

        lib_path = None
        for p in lib_paths:
            if p.exists():
                lib_path = p
                break

        if lib_path is None:
            raise RuntimeError("Could not find libcpp_embedder shared library")

        self._lib = ctypes.CDLL(str(lib_path))

        # Define function signatures
        self._lib.embedder_create.argtypes = [ctypes.c_char_p]
        self._lib.embedder_create.restype = ctypes.c_void_p

        self._lib.embedder_create_with_vocab.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._lib.embedder_create_with_vocab.restype = ctypes.c_void_p

        self._lib.embedder_destroy.argtypes = [ctypes.c_void_p]
        self._lib.embedder_destroy.restype = None

        self._lib.embedder_embed.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_char_p,  # text
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int,  # output_size
        ]
        self._lib.embedder_embed.restype = ctypes.c_int

        self._lib.embedder_get_error.argtypes = []
        self._lib.embedder_get_error.restype = ctypes.c_char_p

        # Create embedder instance with explicit vocab path
        if self.vocab_path:
            self._handle = self._lib.embedder_create_with_vocab(
                self.weights_path.encode('utf-8'),
                self.vocab_path.encode('utf-8')
            )
        else:
            self._handle = self._lib.embedder_create(self.weights_path.encode('utf-8'))

        if not self._handle:
            error = self._lib.embedder_get_error()
            raise RuntimeError(f"Failed to create embedder: {error.decode() if error else 'unknown error'}")

    def embed(
        self,
        texts: list[str],
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        import ctypes

        embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        output_buffer = (ctypes.c_float * self.embedding_dim)()

        iterator = tqdm(texts, desc="C++ embedding") if show_progress else texts

        for i, text in enumerate(iterator):
            result = self._lib.embedder_embed(
                self._handle,
                text.encode('utf-8'),
                output_buffer,
                self.embedding_dim,
            )

            if result != 0:
                error = self._lib.embedder_get_error()
                raise RuntimeError(f"Embedding failed: {error.decode() if error else 'unknown error'}")

            embeddings[i] = np.array(output_buffer)

            # Normalize if requested (C++ already normalizes, but be explicit)
            if normalize:
                norm = np.linalg.norm(embeddings[i])
                if norm > 0:
                    embeddings[i] /= norm

        return embeddings

    def embed_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed([text], normalize=normalize, show_progress=False)[0]

    def __del__(self):
        """Clean up C++ resources."""
        if hasattr(self, '_handle') and self._handle:
            self._lib.embedder_destroy(self._handle)

    def __repr__(self) -> str:
        return f"CppEmbedder(weights={self.weights_path}, dim={self.embedding_dim})"


def benchmark_embedders(texts: list[str], n_runs: int = 3) -> dict:
    """Benchmark Python vs C++ embedder performance."""
    print("\n" + "=" * 60)
    print("BENCHMARKING EMBEDDERS")
    print("=" * 60)

    results = {
        "python": {"times": [], "throughput": []},
        "cpp": {"times": [], "throughput": []},
    }

    # Test subset for benchmarking
    # Note: C++ implementation is pure/unoptimized, use small sample
    test_texts = texts[:10]  # Small sample for C++ benchmark

    # Python embedder
    print("\nBenchmarking Python (sentence-transformers)...")
    py_embedder = PaperEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")

    for run in range(n_runs):
        start = time.perf_counter()
        _ = py_embedder.embed(test_texts, show_progress=False)
        elapsed = time.perf_counter() - start
        results["python"]["times"].append(elapsed)
        results["python"]["throughput"].append(len(test_texts) / elapsed)
        print(f"  Run {run + 1}: {elapsed:.3f}s ({len(test_texts) / elapsed:.1f} texts/sec)")

    # C++ embedder
    print("\nBenchmarking C++ embedder...")
    weights_path = Path(__file__).parent.parent / "data/models/all-MiniLM-L6-v2.bin"
    vocab_path = Path(__file__).parent.parent / "data/models/all-MiniLM-L6-v2.vocab"
    cpp_embedder = CppEmbedder(str(weights_path), str(vocab_path))

    for run in range(n_runs):
        start = time.perf_counter()
        _ = cpp_embedder.embed(test_texts, show_progress=False)
        elapsed = time.perf_counter() - start
        results["cpp"]["times"].append(elapsed)
        results["cpp"]["throughput"].append(len(test_texts) / elapsed)
        print(f"  Run {run + 1}: {elapsed:.3f}s ({len(test_texts) / elapsed:.1f} texts/sec)")

    # Summary statistics
    results["python"]["mean_time"] = np.mean(results["python"]["times"])
    results["python"]["mean_throughput"] = np.mean(results["python"]["throughput"])
    results["cpp"]["mean_time"] = np.mean(results["cpp"]["times"])
    results["cpp"]["mean_throughput"] = np.mean(results["cpp"]["throughput"])

    print("\n--- Summary ---")
    print(f"Python: {results['python']['mean_time']:.3f}s avg, {results['python']['mean_throughput']:.1f} texts/sec")
    print(f"C++:    {results['cpp']['mean_time']:.3f}s avg, {results['cpp']['mean_throughput']:.1f} texts/sec")
    print(f"Speedup: {results['python']['mean_time'] / results['cpp']['mean_time']:.2f}x")

    return results


def compare_embeddings(texts: list[str], n_samples: int = 20) -> dict:
    """Compare embedding quality between Python and C++ implementations."""
    print("\n" + "=" * 60)
    print("COMPARING EMBEDDING QUALITY")
    print("=" * 60)

    # Use subset for comparison
    sample_texts = texts[:n_samples]

    # Generate embeddings with both
    print("\nGenerating Python embeddings...")
    py_embedder = PaperEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")
    py_embeddings = py_embedder.embed(sample_texts, show_progress=True)

    print("\nGenerating C++ embeddings...")
    weights_path = Path(__file__).parent.parent / "data/models/all-MiniLM-L6-v2.bin"
    vocab_path = Path(__file__).parent.parent / "data/models/all-MiniLM-L6-v2.vocab"
    cpp_embedder = CppEmbedder(str(weights_path), str(vocab_path))
    cpp_embeddings = cpp_embedder.embed(sample_texts, show_progress=True)

    results = {}

    # 1. Cosine similarity between corresponding embeddings
    print("\nComputing cosine similarities...")
    cosine_sims = np.sum(py_embeddings * cpp_embeddings, axis=1)
    results["cosine_similarities"] = {
        "mean": float(np.mean(cosine_sims)),
        "std": float(np.std(cosine_sims)),
        "min": float(np.min(cosine_sims)),
        "max": float(np.max(cosine_sims)),
        "values": cosine_sims.tolist(),
    }
    print(f"  Mean cosine similarity: {results['cosine_similarities']['mean']:.6f}")
    print(f"  Std: {results['cosine_similarities']['std']:.6f}")
    print(f"  Range: [{results['cosine_similarities']['min']:.6f}, {results['cosine_similarities']['max']:.6f}]")

    # 2. L2 distance between embeddings
    l2_distances = np.linalg.norm(py_embeddings - cpp_embeddings, axis=1)
    results["l2_distances"] = {
        "mean": float(np.mean(l2_distances)),
        "std": float(np.std(l2_distances)),
        "min": float(np.min(l2_distances)),
        "max": float(np.max(l2_distances)),
    }
    print(f"  Mean L2 distance: {results['l2_distances']['mean']:.6f}")

    # 3. Ranking correlation (do similar texts get similar rankings?)
    print("\nComputing ranking correlation...")
    n_queries = 50
    query_indices = np.random.choice(n_samples, n_queries, replace=False)

    ranking_correlations = []
    for qi in query_indices:
        py_sims = np.dot(py_embeddings, py_embeddings[qi])
        cpp_sims = np.dot(cpp_embeddings, cpp_embeddings[qi])

        py_ranks = np.argsort(-py_sims)
        cpp_ranks = np.argsort(-cpp_sims)

        # Compute rank correlation for top-20
        top_k = 20
        py_top = set(py_ranks[:top_k])
        cpp_top = set(cpp_ranks[:top_k])
        overlap = len(py_top & cpp_top) / top_k
        ranking_correlations.append(overlap)

    results["ranking_correlation"] = {
        "mean": float(np.mean(ranking_correlations)),
        "std": float(np.std(ranking_correlations)),
    }
    print(f"  Mean top-20 ranking overlap: {results['ranking_correlation']['mean']:.4f}")

    # 4. Nearest neighbor consistency
    print("\nComputing nearest neighbor consistency...")
    nn_matches = 0
    for i in range(n_samples):
        py_sims = np.dot(py_embeddings, py_embeddings[i])
        cpp_sims = np.dot(cpp_embeddings, cpp_embeddings[i])
        py_sims[i] = -np.inf
        cpp_sims[i] = -np.inf

        if np.argmax(py_sims) == np.argmax(cpp_sims):
            nn_matches += 1

    results["nearest_neighbor_agreement"] = nn_matches / n_samples
    print(f"  Nearest neighbor agreement: {results['nearest_neighbor_agreement']:.4f}")

    return results, py_embeddings, cpp_embeddings


def create_comparison_plots(
    comparison_results: dict,
    benchmark_results: dict,
    py_embeddings: np.ndarray,
    cpp_embeddings: np.ndarray,
    save_dir: Path,
) -> None:
    """Generate comparison visualization plots."""
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 60)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Cosine similarity distribution
    print("  Creating cosine similarity distribution plot...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    cosine_sims = comparison_results["cosine_similarities"]["values"]
    ax.hist(cosine_sims, bins=50, edgecolor='white', alpha=0.8, color='steelblue')
    ax.axvline(np.mean(cosine_sims), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(cosine_sims):.4f}')
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Python vs C++ Embedding Cosine Similarity Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_dir / "cosine_similarity_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Performance comparison bar chart
    print("  Creating performance comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Throughput comparison
    backends = ['Python\n(sentence-transformers)', 'C++\n(pure implementation)']
    throughputs = [benchmark_results["python"]["mean_throughput"],
                   benchmark_results["cpp"]["mean_throughput"]]
    colors = ['#3498db', '#e74c3c']

    axes[0].bar(backends, throughputs, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_ylabel('Throughput (texts/second)', fontsize=12)
    axes[0].set_title('Embedding Throughput Comparison', fontsize=14, fontweight='bold')
    for i, v in enumerate(throughputs):
        axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=11, fontweight='bold')

    # Time comparison
    times = [benchmark_results["python"]["mean_time"],
             benchmark_results["cpp"]["mean_time"]]
    axes[1].bar(backends, times, color=colors, edgecolor='white', linewidth=2)
    axes[1].set_ylabel('Time (seconds) for 100 texts', fontsize=12)
    axes[1].set_title('Embedding Time Comparison', fontsize=14, fontweight='bold')
    for i, v in enumerate(times):
        axes[1].text(i, v + 0.01, f'{v:.3f}s', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_dir / "performance_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Embedding space comparison (UMAP)
    print("  Creating embedding space comparison plot...")
    try:
        import umap

        n_samples = min(500, len(py_embeddings))
        indices = np.random.choice(len(py_embeddings), n_samples, replace=False)

        combined = np.vstack([py_embeddings[indices], cpp_embeddings[indices]])
        labels = ['Python'] * n_samples + ['C++'] * n_samples

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        projection = reducer.fit_transform(combined)

        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

        py_proj = projection[:n_samples]
        cpp_proj = projection[n_samples:]

        ax.scatter(py_proj[:, 0], py_proj[:, 1], c='#3498db', alpha=0.5, s=30, label='Python')
        ax.scatter(cpp_proj[:, 0], cpp_proj[:, 1], c='#e74c3c', alpha=0.5, s=30, label='C++')

        # Draw lines connecting corresponding points
        for i in range(0, n_samples, 10):  # Every 10th point to avoid clutter
            ax.plot([py_proj[i, 0], cpp_proj[i, 0]],
                   [py_proj[i, 1], cpp_proj[i, 1]],
                   'gray', alpha=0.2, linewidth=0.5)

        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_title('Embedding Space Comparison (UMAP Projection)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(save_dir / "embedding_space_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError:
        print("    Warning: umap-learn not available, skipping UMAP plot")

    # 4. Quality metrics summary
    print("  Creating quality metrics summary plot...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    metrics = ['Cosine\nSimilarity', 'Ranking\nOverlap', 'NN\nAgreement']
    values = [
        comparison_results["cosine_similarities"]["mean"],
        comparison_results["ranking_correlation"]["mean"],
        comparison_results["nearest_neighbor_agreement"],
    ]

    bars = ax.bar(metrics, values, color=['#2ecc71', '#9b59b6', '#f39c12'],
                  edgecolor='white', linewidth=2)
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect agreement')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Python vs C++ Embedding Quality Metrics', fontsize=14, fontweight='bold')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_dir / "quality_metrics_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll plots saved to {save_dir}")


def main():
    """Run embedder comparison experiment."""
    print("=" * 60)
    print("EMBEDDER COMPARISON: Python vs C++")
    print("=" * 60)

    config = load_config("configs/default.yaml")
    plots_dir = Path(config["paths"]["plots_dir"])

    # Load papers
    print("\nLoading paper data...")
    metadata_path = Path(config["paths"]["metadata_file"])
    if not metadata_path.exists():
        raise RuntimeError(f"Metadata not found at {metadata_path}. Run main experiment first.")

    fetcher = ArxivFetcher.load(str(metadata_path))
    preprocessor = TextPreprocessor(remove_latex=True)
    texts = preprocess_papers(fetcher.papers, preprocessor)
    print(f"Loaded {len(texts)} papers")

    # Run benchmark
    benchmark_results = benchmark_embedders(texts)

    # Run quality comparison
    comparison_results, py_emb, cpp_emb = compare_embeddings(texts)

    # Create plots
    create_comparison_plots(
        comparison_results,
        benchmark_results,
        py_emb,
        cpp_emb,
        plots_dir / "comparison",
    )

    # Save results
    results = {
        "benchmark": benchmark_results,
        "comparison": {k: v for k, v in comparison_results.items() if k != "cosine_similarities"},
    }
    results["comparison"]["cosine_similarity_mean"] = comparison_results["cosine_similarities"]["mean"]
    results["comparison"]["cosine_similarity_std"] = comparison_results["cosine_similarities"]["std"]

    results_path = plots_dir / "comparison" / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
