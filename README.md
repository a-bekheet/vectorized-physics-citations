# Vectorized Physics Citations

Semantic search and visualization for Plasma Physics papers from ArXiv using vector embeddings.

## Overview

This project implements a vector-based semantic search system for academic papers from the ArXiv physics.plasm-ph (Plasma Physics) category. Using Sentence-BERT embeddings and FAISS indexing, it enables:

- **Semantic Search**: Find papers by meaning, not just keywords
- **Similar Paper Discovery**: Given a paper, find conceptually related work
- **Visual Exploration**: Interactive 2D visualization of the research landscape
- **Fast Queries**: Sub-100ms search across 10,000+ papers

## Installation

```bash
# Clone the repository
git clone https://github.com/a-bekheet/vectorized-physics-citations.git
cd vectorized-physics-citations

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run the Experiment Pipeline

```bash
python experiments/run_experiment.py
```

This will:
- Fetch ~10,000 Plasma Physics papers from ArXiv
- Generate embeddings using Sentence-BERT
- Build a FAISS search index
- Create visualizations

### 2. Use the CLI

```bash
# Search by text query
python cli.py search "tokamak plasma instabilities"

# Find papers similar to a specific paper
python cli.py similar 2310.12345

# Show paper details
python cli.py info 2310.12345

# View database statistics
python cli.py stats
```

## Project Structure

```
vectorized-physics-citations/
├── configs/default.yaml      # Configuration
├── data/                     # Data fetching and preprocessing
│   ├── arxiv_fetcher.py      # ArXiv API client
│   └── preprocessor.py       # Text cleaning
├── models/                   # Embedding and search
│   ├── embedder.py           # Sentence-BERT wrapper
│   └── search_index.py       # FAISS index
├── evaluation/               # Metrics and analysis
│   └── metrics.py            # Clustering, search quality
├── utils/                    # Utilities
│   ├── visualization.py      # UMAP, plots
│   └── config.py             # Config loading
├── experiments/              # Experiment scripts
│   └── run_experiment.py     # Main pipeline
├── cli.py                    # Command-line interface
└── results/                  # Generated outputs
```

## Technical Details

### Embedding Model
- **Model**: `all-MiniLM-L6-v2` (Sentence-BERT)
- **Dimension**: 384
- **Input**: Title + Abstract concatenation

### Search Index
- **Library**: FAISS
- **Index Type**: Flat (exact) or IVF (approximate)
- **Similarity**: Cosine (via inner product on normalized vectors)

### Visualization
- **Dimensionality Reduction**: UMAP
- **Parameters**: 15 neighbors, 0.1 min_dist, cosine metric

## Results

After running the experiment:
- `results/plots/embedding_space.png` - Static visualization
- `results/plots/paper_clusters.png` - Clustered view
- `results/plots/interactive_space.html` - Interactive exploration
- `results/plots/evaluation_results.json` - Metrics

## License

MIT License - see LICENSE file for details.

## Author

Ali Bekheet - [GitHub](https://github.com/a-bekheet)
