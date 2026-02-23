# Retrieval Sensitivity Analysis

## Overview

This project evaluates the sensitivity of information retrieval methods to document transformations using **120 comprehensive clinical queries**. It compares three retrieval approaches (BM25, Dense embeddings with FAISS, and Hybrid) on clinical text documents that have been augmented with Protected Health Information (PHI) such as fictional patient names and hospital locations. The analysis measures performance degradation across different retrieval methods when documents contain injected PHI.

## Project Structure

- **retrieval.py** - Main evaluation script implementing three retrieval methods and visualization
- **check.py** - Utility script for checking PyTorch and CUDA configuration
- **data.txt** - JSON dataset containing 50 clinical documents with original/transformed pairs and token-level labels

## Dataset Format

The dataset consists of 50 clinical case documents with the following structure:

```json
{
  "original": ["word", "tokens", "..."],
  "originalLabels": [0, 0, ...],
  "transformed": ["word", "tokens", "with", "fictional", "names", "..."],
  "transformedLabels": [0, 0, ...]
}
```

- **original**: Clean clinical text (e.g., "severe neutropenia sepsis")
- **transformed**: Same text with injected fictional patient names and hospital locations
- **Labels**: Token classification (0=neutral, 1=treatment, 2=condition, 5-6=person entities, 7-9=location entities)
- **698 total tokenized examples** spanning treatment protocols, diagnoses, outcomes, and complications

## Query Dataset

The evaluation uses **120 specific clinical queries** organized across major medical categories:

- Drug-disease combinations (omeprazole hemolytic anemia, simvastatin liver failure, etc.)
- Adverse drug reactions (cardiac, hepatic, renal, hematologic)
- Clinical conditions and management (seizures, heart failure, kidney disease)
- Laboratory findings (elevated markers, enzyme levels, viral load)
- Multi-system presentations and severe outcomes
- Drug interactions and synergistic effects
- Patient demographics and special populations
- Study design and therapeutic outcomes
- Pharmacological classes and special populations

### Sample Queries

- "severe neutropenia sepsis mortality patients clinical outcome"
- "omeprazole hemolytic anemia drug-induced immune mechanism"
- "cyclosporine-induced TMA renal transplant plasmapheresis coagulation disorder"
- "carbamazepine neurotoxicity verapamil calcium channel blocker drug interaction"
- "propylthiouracil hyperthyroidism pericarditis fever glomerulonephritis autoimmune reaction"
- "interferon alpha ribavirin hepatitis C diplopia ophthalmologic ptosis"

## Features

### Retrieval Methods

1. **BM25 Retrieval** - Keyword-based sparse retrieval using Okapi BM25 scoring
2. **Dense Retrieval** - Neural embedding-based retrieval using:
   - SentenceTransformer (`all-MiniLM-L6-v2` model)
   - FAISS IndexFlatIP for efficient similarity search
   - L2 normalization for cosine similarity
   - GPU acceleration (CUDA) when available
3. **Hybrid Retrieval** - Ensemble combining:
   - BM25 results (top 10)
   - Dense retrieval results (top 10)
   - Cross-Encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
   - Intelligent result deduplication and combination

### Ground Truth Mapping

Intelligent multi-document relevance mapping using:

- Term overlap analysis between queries and documents
- Minimum 2-3 term overlap threshold for relevance detection
- Maps each query to multiple relevant documents (not 1:1)
- Handles sparse and dense query-document relationships

### Evaluation Metrics

- **Recall@3** - Whether any relevant document appears in top 3 results
- **MRR@3** - Mean Reciprocal Rank at 3 results (first relevant document position)
- **Sensitivity Drops** - Performance degradation comparing clean vs PHI-augmented documents
- **Visualization** - Dynamic axis scaling with 6-decimal precision for micro-differences

## Requirements

```
torch>=1.9.0
numpy
rank-bm25
sentence-transformers
faiss-cpu (or faiss-gpu for GPU acceleration)
matplotlib
```

Install with:

```bash
pip install torch numpy rank-bm25 sentence-transformers faiss-cpu matplotlib
```

## Usage

1. **Check your environment:**

   ```bash
   python check.py
   ```

   This verifies PyTorch, CUDA availability, and GPU memory.

2. **Run retrieval evaluation:**

   ```bash
   python retrieval.py
   ```

   The script will:
   - Load 50 clinical documents and 120 specialized queries
   - Build intelligent multi-document ground truth mapping
   - Evaluate BM25 on clean and PHI-augmented documents
   - Evaluate Dense retrieval (FAISS) on both versions
   - Evaluate Hybrid method combining all three approaches
   - Calculate Recall@5 and MRR@10 metrics
   - Measure sensitivity drops due to PHI injection
   - Generate 4-panel visualization comparing all methods
   - Save analysis plot as `retrieval_sensitivity_analysis.png` (300 DPI)

## Output

The evaluation generates:

- **Console output**: Performance metrics for each retrieval method

  ```
  === BM25 ===
  Clean   - Recall@3: X.XXX, MRR@10: X.XXX
  PHI     - Recall@3: X.XXX, MRR@10: X.XXX
  Drop    - Recall Δ: 0.XXXXXX, MRR Δ: 0.XXXXXX
  ```

- **Visualization**: 4-panel plot showing:
  - Panel 1: Recall@3 comparison
  - Panel 2: MRR@3 comparison
  - Panel 3: Recall@3 drop
  - Panel 4: MRR@3 drop

## Purpose

This analysis evaluates how information retrieval systems handle document transformations with injected Protected Health Information (PHI). It helps understand:

- Robustness of different retrieval methods to document augmentation
- Sensitivity of BM25, dense embeddings, and hybrid approaches
- Clinical text retrieval performance under realistic PHI scenarios
- Comparative effectiveness of sparse vs dense vs hybrid methods
- Quantifiable performance degradation from document transformations

## Clinical Application

The project models real-world scenarios where clinical documents may contain:

- Patient names and identifiers
- Hospital and facility locations
- Other Protected Health Information

Understanding how retrieval systems handle these transformations is critical for clinical NLP, medical record management, and healthcare information systems.
