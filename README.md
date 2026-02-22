# Retrieval Sensitivity Analysis

## Overview

This project evaluates the sensitivity of information retrieval methods to document transformations. It compares how different retrieval approaches (BM25 and dense retrieval with FAISS) handle clinical text documents that have been augmented with additional entities such as names and locations.

## Project Structure

- **retrieval.py** - Main script for evaluating retrieval methods
- **check.py** - Utility script for checking PyTorch and CUDA configuration
- **data.txt** - JSON dataset containing original and transformed clinical documents with token-level labels

## Dataset Format

The dataset consists of clinical text examples with the following structure:

```json
{
  "original": ["word", "tokens", "..."],
  "originalLabels": [0, 0, ...],
  "transformed": ["word", "tokens", "with", "additions", "..."],
  "transformedLabels": [0, 0, ...]
}
```

- **original**: Clean clinical text tokenized into words
- **transformed**: Same text augmented with fictional names and hospital locations
- **Labels**: Token-level classification (0=neutral, 1=medical treatment, 2=medical condition, 5-6=person entities, 7-9=location entities)

## Features

### Retrieval Methods

1. **BM25 Retrieval** - Keyword-based sparse retrieval using BM25 scoring
2. **Dense Retrieval** - Neural embedding-based retrieval using:
   - SentenceTransformer (`all-MiniLM-L6-v2` model)
   - FAISS for efficient similarity search
   - GPU acceleration (CUDA) when available

### Evaluation Metrics

- **Recall@5** - Whether the relevant document appears in top 5 results
- **MRR@10** - Mean Reciprocal Rank at 10 results

## Requirements

```
torch
numpy
rank-bm25
sentence-transformers
faiss-cpu (or faiss-gpu)
```

## Usage

1. **Check your environment:**

   ```bash
   python check.py
   ```

   This verifies PyTorch installation and CUDA availability.

2. **Run retrieval evaluation:**
   ```bash
   python retrieval.py
   ```
   The script will:
   - Load the clinical dataset
   - Evaluate BM25 performance on clean documents
   - Evaluate BM25 performance on transformed documents
   - Evaluate dense retrieval on both versions
   - Calculate and compare metrics

## Purpose

This analysis helps understand how document modifications (entity injections, location additions) affect retrieval system performance and robustness in clinical text processing.

## License

[Add your license information here]
