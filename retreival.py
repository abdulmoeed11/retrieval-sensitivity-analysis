import json
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from collections import defaultdict
import matplotlib.pyplot as plt

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
    "severe neutropenia sepsis mortality patients clinical outcome",
    "methylprednisolone treatment immunosuppression gradual improvement recovery",
    "intensive medical support patient recovery hospitalization care",
    "hypotensive patients shock treatment hemodynamic instability management",
    "elevated plasma deoxycorticosterone abnormal cortisol metabolism hormone levels",
    "asymptomatic aneurysm routine control incidental finding detection",
    "severe distress recovery phase post-operative complications",
    "spontaneous recovery hospital surveillance three hour timeline",
    "risperidone reduction withdrawal syndrome drug discontinuation effect",
    "elevated creatine kinase elevated creatinine kidney injury markers",
    "pyridoxine supplementation pharmacologic dosing long-term therapy",
    "omeprazole hemolytic anemia drug-induced immune mechanism",
    "mild severe neutropenia retrospective study adverse event",
    "venlafaxine incontinence side effects urinary dysfunction males",
    "HBV DNA levels hepatitis B viral replication follow-up monitoring",
    "focal seizures transient hemiparesis neurological complications recovery",
    "paracetamol toxicity serum level non-toxic range overdose",
    "parenteral corticosteroids paramethasone dexamethasone intravenous administration",
    "congestive heart failure chemotherapy completion cardiac toxicity",
    "FK506 MAHA microangiopathic hemolytic anemia drug rechallenge recurrence",
    "extubated patient mechanical ventilation weaning peripheral ward transfer",
    "severe headache abdominal pain study discontinuation adverse events",
    "losartan anuria solitary kidney renal insufficiency angiotensin blocker",
    "petechiae purpura thrombocytopenia bleeding manifestations thorax limbs",
    "creatine monohydrate herbal supplement elevated kidney function",
    "intubated bradyarrhythmic patient hypotension blood pressure monitoring",
    "angiography preangiographic testing cardiac catheterization decision-making",
    "chronic active hepatitis diclofenac sodium drug-induced liver injury",
    "sensorineural hearing loss drug ototoxicity irreversible auditory damage",
    "simvastatin hepatotoxicity lipid panel statin therapy monitoring",
    "renal function impairment dialysis recovery post-treatment",
    "paramethasone skin test ELISA allergy hypersensitivity reaction",
    "methyldopa hepatic dysfunction hepatitis drug causality recurrence",
    "troleandomycin hepatitis prolonged anicteric cholestasis biliary obstruction",
    "venlafaxine overdose generalized seizure toxicity convulsion",
    "HIV-infected immunocompromised severe intrahepatic cholestasis opportunistic infection",
    "bradyarrhythmias lidocaine idiosyncratic reaction cardiac conduction block",
    "coronary artery disease shock cardiac intensive care unit admission",
    "glyburide rapid recovery hypoglycemia therapy discontinuation one year follow-up",
    "imipramine syndrome antidepressant psychiatric medication adverse reaction",
    "labetalol metoprolol midazolam sedation consciousness altered mental status",
    "antithymocyte globulin aplastic anemia D-penicillamine drug-induced bone marrow failure",
    "pseudoacromegaly minoxidil hair loss medication systemic effect",
    "tingling burning sensations heat exposure sunburn erythema neuropathy dermatological",
    "pharmacological supportive interventions laboratory parameters ICU mortality",
    "carbamazepine absence epilepsy vigabatrin seizure disorder onset",
    "etoposide cyclosporin synergistic effect leukemia chemotherapy relapse",
    "weakness lethargy shortness breath omeprazole respiratory symptoms",
    "cholestatic hepatitis carbimazole thyroid hormone mixed hepatitis pattern",
    "dysphonia acitretin skin condition retinoid side effect voice changes",
    "apheresis platelet donor transfusion reaction collection procedure",
    "blurred vision infertility clinic visual dysfunction reproductive evaluation",
    "central retinal vein occlusion clomiphene citrate thrombosis vascular occlusion",
    "argatroban cardiac transplant heparin-induced thrombocytopenia anticoagulation",
    "massive transfusion intraoperative bleeding coagulopathy surgical complication",
    "myocardial infarction isosorbide dinitrate vasodilator nitrate therapy",
    "neuroleptic akathisia fluoxetine psychomotor restlessness SSRI antipsychotic",
    "type II diabetes mellitus glyburide acute hepatitis-like syndrome",
    "prazosin verapamil urodynamic studies urinary dysfunction cardiovascular medication",
    "allergic reaction angioneurotic edema 5-fluorouracil chemotherapy anaphylaxis",
    "mild neutropenia retrospective cohort adverse drug effect study",
    "cisplatin etoposide gastric adenocarcinoma combination chemotherapy inoperable",
    "ampicillin gentamicin septicemia suspected infection antibiotic therapy",
    "cardiac arrest hyperkalaemia suxamethonium anesthetic complication electrolyte imbalance",
    "vincristine intrathecal injection fatal lymphoblastic leucemia chemotherapy error",
    "coronary artery disease markedly prolonged QT interval torsades de pointes ketoconazole",
    "schizophrenia myocarditis clozapine atypical antipsychotic cardiac inflammation",
    "idiopathic epilepsy generalized seizures LEV valproate seizure frequency increase",
    "interferon alpha ribavirin hepatitis C diplopia ophthalmologic ptosis",
    "hemolytic uremic syndrome oral contraceptives thrombotic microangiopathy",
    "carbamazepine neurotoxicity verapamil calcium channel blocker drug interaction",
    "cerebral infarction PPA phenylpropanolamine stroke risk",
    "WPW syndrome dilated cardiomyopathy atrioventricular reentrant tachycardia iatrogenic",
    "pethidine pain control patient-controlled analgesia postoperative management",
    "Stevens-Johnson syndrome ibuprofen vanishing bile duct pediatric adverse reaction",
    "euphoria choreoathetoid movements methadone inpatient drug abuse heroin cocaine",
    "cyclosporine-induced TMA renal transplant plasmapheresis coagulation disorder",
    "hypertensive solitary kidney chronic renal insufficiency losartan transient anuria",
    "cocaine priapism emergency department frequent user repeated episodes",
    "simvastatin ezetimibe fulminant hepatic failure liver transplantation drug conversion",
    "lopinavir ritonavir complete heart block dilated cardiomyopathy protease inhibitor therapy",
    "diethylstilbestrol adenocarcinoma angiosarcoma liver cancer long-term hormone therapy",
    "sensorineural hearing loss drug-induced ototoxicity irreversible auditory impairment",
    "nephrotic syndrome focal segmental glomerulosclerosis steroid therapy responsive",
    "propylthiouracil hyperthyroidism pericarditis fever glomerulonephritis autoimmune reaction",
    "morphine unrelieved pain cancer patients randomized crossover study",
    "Graves disease febrile illness pericarditis biopsy confirmed",
    "ketamine drowsiness CNS depression sedation clinical trial",
    "MMSE cognitive impairment time interval treatment comparison",
    "testosterone low sexual desire erectile dysfunction screening prolactin",
    "propofol injection pain anesthesia induction ambulatory surgery",
    "lidocaine thiopentone analgesic premedication anesthesia protocols",
    "glycopyrrolate atropine anticholinergic comparison lower dose range",
    "acetylsalicylate caffeine acetaminophen analgesic combination caffeine toxicity",
    "unstable angina flestolol beta-blocker chest pain control",
    "intractable tinnitus therapeutic trial complete abolition compensated decompensated",
    "SSRI selective serotonin reuptake inhibitor sexual dysfunction erectile dysfunction",
    "bupropion sexual function IIEF score antidepressant improvement",
    "drug-induced thrombocytopenia bleeding manifestations treatment discontinuation",
    "hepatitis autoimmune drug rechallenge causality assessment mechanism",
    "drug interaction carbamazepine verapamil neurotoxicity combined therapy",
    "postoperative pain management analgesic efficacy patient satisfaction outcomes",
    "chemotherapy-induced cardiomyopathy left ventricular dysfunction cardiac monitoring",
    "antithymocyte globulin transplant rejection bone marrow suppression immunosuppression",
    "electrolyte imbalance hyperkalemia cardiac arrhythmia emergency treatment",
    "drug-induced liver injury hepatotoxicity enzyme elevation biopsy findings",
    "medication error intrathecal injection fatal outcome adverse event reporting",
    "angiotensin receptor blocker renal hemodynamics blood pressure control",
    "corticosteroid therapy immunosuppression infection risk opportunistic pathogens",
    "anticonvulsant hypersensitivity syndrome DRESS fever rash lymphadenopathy",
    "opioid receptor agonist methadone addiction treatment withdrawal management",
    "microangiopathic hemolytic anemia schizocytes platelet consumption hemolysis",
    "drug-induced QT prolongation torsades de pointes cardiac arrhythmia antiarrhythmic",
    "intracranial pressure monitoring neurological deterioration traumatic head injury",
    "antibiotic-induced ototoxicity aminoglycoside hearing loss vestibular dysfunction",
    "anesthetic induction propofol sedation airway management emergence",
    "chemotherapy regimen platinum-based etoposide fluorouracil combination cancer",
]

# Ground truth mapping - each query maps to document indices containing relevant medical concepts
# Building based on actual document content matching
def build_ground_truth(docs, queries):
    """
    Build ground truth by finding relevant documents for each query.
    A document is relevant if it contains significant overlap with query terms.
    """
    ground_truth = {}
    
    for qid, query in enumerate(queries):
        query_terms = set(query.lower().split())
        relevant_docs = []
        
        for doc_id, doc in enumerate(docs):
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            
            # Consider relevant if there's sufficient term overlap (at least 2-3 terms)
            if overlap >= min(3, len(query_terms) // 2 + 1):
                relevant_docs.append(doc_id)
        
        # If no relevant docs found, at least add the closest one by position
        if not relevant_docs:
            relevant_docs = [qid % len(docs)]
        
        ground_truth[qid] = relevant_docs
    
    return ground_truth


# Ground truth mapping - built from actual document-query similarities
ground_truth = build_ground_truth(clean_docs, queries)


############################
# 3. METRICS
############################

def recall_at_k(ranked_indices, relevant_indices, k=3):
    """Check if any relevant document is in top-k results"""
    relevant_set = set(relevant_indices)
    retrieved_set = set(ranked_indices[:k])
    if not relevant_set:
        return 1.0  # If no relevant docs, consider it a match
    return int(len(relevant_set & retrieved_set) > 0)

def mrr_at_k(ranked_indices, relevant_indices, k=3):
    """Mean Reciprocal Rank - position of first relevant document"""
    relevant_set = set(relevant_indices)
    for rank, idx in enumerate(ranked_indices[:k], start=1):
        if idx in relevant_set:
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

        relevant_docs = ground_truth[qid]

        recalls.append(recall_at_k(ranked_indices, relevant_docs, k=3))
        mrrs.append(mrr_at_k(ranked_indices, relevant_docs, k=3))

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
        relevant_docs = ground_truth[qid]

        recalls.append(recall_at_k(ranked_indices, relevant_docs, k=3))
        mrrs.append(mrr_at_k(ranked_indices, relevant_docs, k=3))

    return np.mean(recalls), np.mean(mrrs)


############################
# 6. HYBRID RETRIEVAL (BM25 + Dense + Cross-Encoder Reranking)
############################

def evaluate_hybrid(docs, queries, ground_truth):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize BM25
    tokenized_corpus = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    # Initialize dense retriever
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

    doc_embeddings = bi_encoder.encode(docs, convert_to_numpy=True)
    query_embeddings = bi_encoder.encode(queries, convert_to_numpy=True)

    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # Normalize for cosine similarity
    faiss.normalize_L2(doc_embeddings)
    faiss.normalize_L2(query_embeddings)
    index.add(doc_embeddings)

    recalls = []
    mrrs = []

    for qid, query in enumerate(queries):
        # Get BM25 results (top 10)
        bm25_scores = bm25.get_scores(query.split())
        bm25_indices = np.argsort(bm25_scores)[::-1][:10]

        # Get Dense results (top 10)
        D, I = index.search(query_embeddings[qid:qid+1], 10)
        dense_indices = I[0]

        # Combine and deduplicate results
        combined_indices = list(dict.fromkeys(list(bm25_indices) + list(dense_indices)))[:20]

        # Rerank using cross-encoder
        query_doc_pairs = [[query, docs[idx]] for idx in combined_indices]
        cross_scores = cross_encoder.predict(query_doc_pairs)

        # Get final ranked indices
        ranked_positions = np.argsort(cross_scores)[::-1]
        ranked_indices = [combined_indices[pos] for pos in ranked_positions]

        relevant_docs = ground_truth[qid]

        recalls.append(recall_at_k(ranked_indices, relevant_docs, k=3))
        mrrs.append(mrr_at_k(ranked_indices, relevant_docs, k=3))

    return np.mean(recalls), np.mean(mrrs)


############################
# 7. RUN EXPERIMENT
############################

# Store results for visualization
results = {
    "BM25": {},
    "Dense": {},
    "Hybrid": {}
}

print("=== BM25 ===")
clean_recall, clean_mrr = evaluate_bm25(clean_docs, queries, ground_truth)
phi_recall, phi_mrr = evaluate_bm25(phi_docs, queries, ground_truth)

results["BM25"]["clean_recall"] = clean_recall
results["BM25"]["clean_mrr"] = clean_mrr
results["BM25"]["phi_recall"] = phi_recall
results["BM25"]["phi_mrr"] = phi_mrr
results["BM25"]["recall_drop"] = clean_recall - phi_recall
results["BM25"]["mrr_drop"] = clean_mrr - phi_mrr

print(f"Clean   - Recall@3: {clean_recall:.3f}, MRR@3: {clean_mrr:.3f}")
print(f"PHI     - Recall@3: {phi_recall:.3f}, MRR@3: {phi_mrr:.3f}")
print(f"Drop    - Recall Δ: {clean_recall - phi_recall:.3f}, MRR Δ: {clean_mrr - phi_mrr:.3f}")

print("\n=== Dense (FAISS) ===")
clean_recall, clean_mrr = evaluate_dense(clean_docs, queries, ground_truth)
phi_recall, phi_mrr = evaluate_dense(phi_docs, queries, ground_truth)

results["Dense"]["clean_recall"] = clean_recall
results["Dense"]["clean_mrr"] = clean_mrr
results["Dense"]["phi_recall"] = phi_recall
results["Dense"]["phi_mrr"] = phi_mrr
results["Dense"]["recall_drop"] = clean_recall - phi_recall
results["Dense"]["mrr_drop"] = clean_mrr - phi_mrr

print(f"Clean   - Recall@3: {clean_recall:.3f}, MRR@3: {clean_mrr:.3f}")
print(f"PHI     - Recall@3: {phi_recall:.3f}, MRR@3: {phi_mrr:.3f}")
print(f"Drop    - Recall Δ: {clean_recall - phi_recall:.3f}, MRR Δ: {clean_mrr - phi_mrr:.3f}")

print("\n=== Hybrid (BM25 + Dense + Cross-Encoder) ===")
clean_recall, clean_mrr = evaluate_hybrid(clean_docs, queries, ground_truth)
phi_recall, phi_mrr = evaluate_hybrid(phi_docs, queries, ground_truth)

results["Hybrid"]["clean_recall"] = clean_recall
results["Hybrid"]["clean_mrr"] = clean_mrr
results["Hybrid"]["phi_recall"] = phi_recall
results["Hybrid"]["phi_mrr"] = phi_mrr
results["Hybrid"]["recall_drop"] = clean_recall - phi_recall
results["Hybrid"]["mrr_drop"] = clean_mrr - phi_mrr

print(f"Clean   - Recall@3: {clean_recall:.3f}, MRR@3: {clean_mrr:.3f}")
print(f"PHI     - Recall@3: {phi_recall:.3f}, MRR@3: {phi_mrr:.3f}")
print(f"Drop    - Recall Δ: {clean_recall - phi_recall:.3f}, MRR Δ: {clean_mrr - phi_mrr:.3f}")


############################
# RESULTS TABLE
############################

print("\n" + "="*100)
print("COMPREHENSIVE RESULTS TABLE - Retrieval Sensitivity Analysis")
print("="*100)

# Create table data
table_data = []
headers = ["Method", "Metric", "Clean", "PHI-Augmented", "Sensitivity Drop"]

for method in ["BM25", "Dense", "Hybrid"]:
    recall_clean = results[method]["clean_recall"]
    recall_phi = results[method]["phi_recall"]
    recall_drop = results[method]["recall_drop"]
    
    mrr_clean = results[method]["clean_mrr"]
    mrr_phi = results[method]["phi_mrr"]
    mrr_drop = results[method]["mrr_drop"]
    
    table_data.append([method, "Recall@3", f"{recall_clean:.4f}", f"{recall_phi:.4f}", f"{recall_drop:.6f}"])
    table_data.append(["", "MRR@3", f"{mrr_clean:.4f}", f"{mrr_phi:.4f}", f"{mrr_drop:.6f}"])

# Print header
print(f"\n{'Method':<15} {'Metric':<12} {'Clean':<15} {'PHI-Aug':<15} {'Drop (Δ)':<15}")
print("-" * 100)

# Print rows
for row in table_data:
    method, metric, clean, phi, drop = row
    print(f"{method:<15} {metric:<12} {clean:<15} {phi:<15} {drop:<15}")

print("="*100)


############################
# 8. VISUALIZATION
############################

methods = list(results.keys())
colors = ["#2E86AB", "#A23B72", "#F18F01"]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Retrieval Sensitivity Analysis: Clean vs PHI-Augmented Documents", fontsize=18, fontweight='bold', y=0.995)

# 1. Recall@3 Comparison
ax = axes[0, 0]
clean_recalls = [results[m]["clean_recall"] for m in methods]
phi_recalls = [results[m]["phi_recall"] for m in methods]
x = np.arange(len(methods))
width = 0.35
bars1 = ax.bar(x - width/2, clean_recalls, width, label="Clean", color=colors, alpha=0.8)
bars2 = ax.bar(x + width/2, phi_recalls, width, label="PHI-Augmented", color=colors, alpha=0.4)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

min_recall = min(min(clean_recalls), min(phi_recalls))
ax.set_ylabel("Recall@3", fontweight='bold', fontsize=12)
ax.set_title("Recall@3 Comparison", fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(fontsize=11)
ax.set_ylim([max(0, min_recall - 0.05), 1.0])  # Zoom in
ax.grid(axis='y', alpha=0.3)

# 2. MRR@3 Comparison
ax = axes[0, 1]
clean_mrrs = [results[m]["clean_mrr"] for m in methods]
phi_mrrs = [results[m]["phi_mrr"] for m in methods]
bars1 = ax.bar(x - width/2, clean_mrrs, width, label="Clean", color=colors, alpha=0.8)
bars2 = ax.bar(x + width/2, phi_mrrs, width, label="PHI-Augmented", color=colors, alpha=0.4)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

min_mrr = min(min(clean_mrrs), min(phi_mrrs))
ax.set_ylabel("MRR@3", fontweight='bold', fontsize=12)
ax.set_title("MRR@3 Comparison", fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(fontsize=11)
ax.set_ylim([max(0, min_mrr - 0.05), 1.0])  # Zoom in
ax.grid(axis='y', alpha=0.3)

# 3. Performance Drop (Recall)
ax = axes[1, 0]
recall_drops = [results[m]["recall_drop"] for m in methods]
bars = ax.bar(methods, recall_drops, color=colors, alpha=0.8, width=0.6)
ax.set_ylabel("Recall@3 Drop (Δ)", fontweight='bold', fontsize=12)
ax.set_title("Recall@3 Drop Due to PHI Augmentation", fontweight='bold', fontsize=13)

# Smart y-limit for drop plot
max_drop = max(recall_drops)
min_drop = min(recall_drops)
y_margin = (max_drop - min_drop) * 0.2 if (max_drop - min_drop) > 0 else max_drop * 0.1
ax.set_ylim([max(0, min_drop - y_margin), max_drop + y_margin])

ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 4. Performance Drop (MRR)
ax = axes[1, 1]
mrr_drops = [results[m]["mrr_drop"] for m in methods]
bars = ax.bar(methods, mrr_drops, color=colors, alpha=0.8, width=0.6)
ax.set_ylabel("MRR@3 Drop (Δ)", fontweight='bold', fontsize=12)
ax.set_title("MRR@3 Drop Due to PHI Augmentation", fontweight='bold', fontsize=13)

# Smart y-limit for drop plot
max_drop = max(mrr_drops)
min_drop = min(mrr_drops)
y_margin = (max_drop - min_drop) * 0.2 if (max_drop - min_drop) > 0 else max_drop * 0.1
ax.set_ylim([max(0, min_drop - y_margin), max_drop + y_margin])

ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig("retrieval_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Plot saved as 'retrieval_sensitivity_analysis.png'")
print("="*70)
print("\nVisualization Details:")
print("  • Top-left: Recall@3 scores comparison")
print("  • Top-right: MRR@3 scores comparison")
print("  • Bottom-left: Recall sensitivity drop")
print("  • Bottom-right: MRR sensitivity drop")
print("\nAll values are displayed on the bars for precise comparison.")
print("="*70)
plt.show()