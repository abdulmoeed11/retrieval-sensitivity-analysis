import json
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from collections import defaultdict

############################
# 1. LOAD YOUR DATASET
############################

# Replace with your dataset loading logic
# Assuming dataset is a list of dicts like the example you shared

with open("data.txt", "r") as f:
    dataset = json.load(f)

clean_docs = [" ".join(sample["original"]) for sample in dataset]
phi_docs = [" ".join(sample["transformed"]) for sample in dataset]

num_docs = len(clean_docs)

############################
# 2. DEFINE CLINICAL QUERIES
############################

queries = [
    "severe neutropenia sepsis",
    "treated with methylprednisolone",
    "gradually improved after treatment",
    "neutropenia mortality",
]

# Ground truth mapping
# For simplicity: assume query i is relevant to document i
# You should adjust this mapping properly
ground_truth = {i: i for i in range(len(queries))}


############################
# 3. METRICS
############################

def recall_at_k(ranked_indices, true_index, k=5):
    return int(true_index in ranked_indices[:k])

def mrr_at_k(ranked_indices, true_index, k=10):
    for rank, idx in enumerate(ranked_indices[:k], start=1):
        if idx == true_index:
            return 1.0 / rank
    return 0.0


############################
# 4. BM25 RETRIEVAL
############################

def evaluate_bm25(docs, queries, ground_truth):
    tokenized_corpus = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    recalls = []
    mrrs = []

    for qid, query in enumerate(queries):
        scores = bm25.get_scores(query.split())
        ranked_indices = np.argsort(scores)[::-1]

        true_doc = ground_truth[qid]

        recalls.append(recall_at_k(ranked_indices, true_doc, k=5))
        mrrs.append(mrr_at_k(ranked_indices, true_doc, k=10))

    return np.mean(recalls), np.mean(mrrs)


############################
# 5. DENSE RETRIEVAL (FAISS)
############################

def evaluate_dense(docs, queries, ground_truth):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    doc_embeddings = model.encode(docs, convert_to_numpy=True)
    query_embeddings = model.encode(queries, convert_to_numpy=True)

    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # Normalize for cosine similarity
    faiss.normalize_L2(doc_embeddings)
    faiss.normalize_L2(query_embeddings)

    index.add(doc_embeddings)

    recalls = []
    mrrs = []

    D, I = index.search(query_embeddings, 10)

    for qid in range(len(queries)):
        ranked_indices = I[qid]
        true_doc = ground_truth[qid]

        recalls.append(recall_at_k(ranked_indices, true_doc, k=5))
        mrrs.append(mrr_at_k(ranked_indices, true_doc, k=10))

    return np.mean(recalls), np.mean(mrrs)


############################
# 6. RUN EXPERIMENT
############################

print("=== BM25 ===")
clean_recall, clean_mrr = evaluate_bm25(clean_docs, queries, ground_truth)
phi_recall, phi_mrr = evaluate_bm25(phi_docs, queries, ground_truth)

print(f"Clean   - Recall@5: {clean_recall:.3f}, MRR@10: {clean_mrr:.3f}")
print(f"PHI     - Recall@5: {phi_recall:.3f}, MRR@10: {phi_mrr:.3f}")
print(f"Drop    - Recall Δ: {clean_recall - phi_recall:.3f}, MRR Δ: {clean_mrr - phi_mrr:.3f}")

print("\n=== Dense (FAISS) ===")
clean_recall, clean_mrr = evaluate_dense(clean_docs, queries, ground_truth)
phi_recall, phi_mrr = evaluate_dense(phi_docs, queries, ground_truth)

print(f"Clean   - Recall@5: {clean_recall:.3f}, MRR@10: {clean_mrr:.3f}")
print(f"PHI     - Recall@5: {phi_recall:.3f}, MRR@10: {phi_mrr:.3f}")
print(f"Drop    - Recall Δ: {clean_recall - phi_recall:.3f}, MRR Δ: {clean_mrr - phi_mrr:.3f}")