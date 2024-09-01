# Vectorized Physics Citations – PRD v0.1
Author: Ali Bekheet
Last Updated: November 2024

## 1. Executive Summary

This project implements a vector-based semantic search and visualization system for Plasma Physics papers from ArXiv. Using modern NLP embeddings (Sentence-BERT), we create dense vector representations of paper abstracts and metadata, enabling similarity search, citation network exploration, and interactive 2D/3D visualizations of the research landscape.

## 2. Motivation

Physics researchers often struggle to discover related work outside their immediate citation networks. Traditional keyword search misses semantic connections between papers. This project demonstrates:
- Practical application of transformer embeddings to academic literature
- Interactive visualization of high-dimensional semantic spaces
- Building a search system that finds "conceptually similar" papers, not just keyword matches

## 3. Goals and Non-Goals

### Goals
- Fetch and process Plasma Physics papers from ArXiv API (physics.plasm-ph category)
- Generate semantic embeddings using pre-trained Sentence-BERT models
- Implement fast approximate nearest neighbor search for similar papers
- Create interactive 2D visualizations using UMAP dimensionality reduction
- Build a CLI tool for paper discovery and exploration
- Visualize citation relationships and semantic clusters

### Non-Goals
- Full-text PDF parsing (abstracts only)
- Training custom embedding models
- Real-time paper monitoring/alerting
- Citation count prediction or impact analysis
- Production deployment infrastructure

## 4. Success Criteria

### Technical
- Successfully embed 5,000+ papers with < 1 second per query response time
- Achieve meaningful semantic clustering visible in UMAP projections
- Cosine similarity search returning relevant papers (qualitative evaluation)
- Clean separation of research topics in visualization

### Portfolio
A reviewer should conclude that the author can:
- Work with REST APIs and data pipelines
- Apply NLP/embedding techniques to real-world problems
- Create effective scientific visualizations
- Build usable command-line tools

## 5. Data Domain

### Data Source
- **ArXiv API**: OAI-PMH protocol for metadata harvesting
- **Category**: physics.plasm-ph (Plasma Physics)
- **Fields**: paper ID, title, abstract, authors, submission date, categories

### Data Processing
1. Fetch paper metadata via ArXiv API
2. Clean and normalize text (remove LaTeX, special characters)
3. Generate embeddings using `all-MiniLM-L6-v2` model
4. Store vectors in FAISS index for efficient search

### Expected Dataset Size
- ~10,000-15,000 papers from physics.plasm-ph
- ~50MB raw metadata
- ~100MB embedding vectors

## 6. Repository Structure

```
vectorized-physics-citations/
├── instructions.md
├── requirements.txt
├── README.md
├── setup.py
├── configs/
│   └── default.yaml
├── data/
│   ├── __init__.py
│   ├── arxiv_fetcher.py
│   └── preprocessor.py
├── models/
│   ├── __init__.py
│   └── embedder.py
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
├── experiments/
│   └── run_experiment.py
├── utils/
│   ├── __init__.py
│   └── visualization.py
├── results/
│   └── (generated outputs)
├── images/
│   └── (plots for documentation)
└── projects/
    └── vectorized-physics-citations.html
```

## 7. Model Architecture

### Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding Dimension**: 384
- **Input**: Paper title + abstract concatenation
- **Output**: Dense vector representation

### Search Index
- **Library**: FAISS (Facebook AI Similarity Search)
- **Index Type**: `IndexFlatIP` (inner product for cosine similarity)
- **Approximate Search**: `IndexIVFFlat` for larger datasets

## 8. Mathematical Formulation

### Cosine Similarity
For two paper embeddings $v_i$ and $v_j$:
$$\text{sim}(v_i, v_j) = \frac{v_i \cdot v_j}{||v_i|| \cdot ||v_j||}$$

### UMAP Projection
Dimensionality reduction from 384D to 2D using:
- `n_neighbors=15`: Local structure preservation
- `min_dist=0.1`: Point spacing in low-dimensional space
- `metric='cosine'`: Semantic distance measure

### Clustering
Optional K-means clustering on embeddings to identify research themes:
$$\min_{\mu} \sum_{i=1}^{n} \min_{k} ||v_i - \mu_k||^2$$

## 9. Training Configuration

No training required (using pre-trained embeddings). Key parameters:

```yaml
# configs/default.yaml
arxiv:
  category: "physics.plasm-ph"
  max_results: 10000
  batch_size: 100

embedding:
  model_name: "all-MiniLM-L6-v2"
  batch_size: 64
  max_seq_length: 512

search:
  index_type: "flat"  # or "ivf" for large datasets
  nprobe: 10
  top_k: 20

visualization:
  umap_neighbors: 15
  umap_min_dist: 0.1
  umap_metric: "cosine"
```

## 10. Evaluation Metrics

### Quantitative
- **Index Build Time**: Time to generate embeddings and build FAISS index
- **Query Latency**: Time for k-NN search (target: <100ms)
- **Memory Usage**: RAM footprint for index and embeddings

### Qualitative
- **Semantic Coherence**: Do similar papers share topics/methods?
- **Cluster Quality**: Visual inspection of UMAP clusters
- **Search Relevance**: Manual evaluation of top-k results

## 11. Experiment Plan

1. **Data Collection**: Fetch 10,000 papers from physics.plasm-ph
2. **Preprocessing**: Clean abstracts, handle LaTeX, tokenize
3. **Embedding Generation**: Compute vectors for all papers
4. **Index Building**: Create FAISS index with different configurations
5. **Visualization**: Generate UMAP projections with cluster coloring
6. **Search Evaluation**: Test query performance and result quality
7. **Documentation**: Create plots and write findings

## 12. Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| ArXiv API rate limiting | Implement exponential backoff, cache responses |
| Memory constraints for large datasets | Use FAISS IVF index, batch processing |
| Poor cluster separation | Tune UMAP parameters, try different embeddings |
| Slow embedding generation | Use GPU if available, batch processing |

## 13. Final Deliverables

- [x] Working Python package with CLI interface
- [x] 10,000+ embedded Plasma Physics papers
- [x] Interactive UMAP visualization (HTML/static)
- [x] Similar paper search functionality
- [x] Academic-quality plots and figures
- [x] Project documentation (README, HTML page)
- [x] Comprehensive instructions.md (this document)
