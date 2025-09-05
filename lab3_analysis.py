import os, re, math
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- A. LOAD & NORMALIZE COLUMNS ----------
# Change this to your actual CSV path
CSV_PATH = "/home/set-iitgn-vm/Desktop/stt_lab/part_c_bugfix_commits.csv"  # or part_c_bugfix_commits.csv

df = pd.read_csv(CSV_PATH)

# Map possible column name variants → standard names we’ll use
col_map_candidates = {
    'hash': ['hash','commit','commit_hash','sha'],
    'message': ['message','commit_message'],
    'filename': ['filename','file','path'],
    'before': ['source code (before)','source_before','code_before','old_code','before'],
    'current': ['source code (current)','source_current','code_current','new_code','current','after'],
    'diff': ['diff'],
    'llm_fix_type': ['llm inference (fix type)','fix_type','llm_fix_type'],
    'rectified_message': ['rectified message','rectified_message']
}

def find_col(df_cols, options):
    s = set(c.lower().strip() for c in df_cols)
    for opt in options:
        if opt in s:
            # return the original exact column name
            for original in df_cols:
                if original.lower().strip() == opt:
                    return original
    return None

std_cols = {}
for std, options in col_map_candidates.items():
    c = find_col(df.columns, options)
    std_cols[std] = c

required = ['hash','filename','before','current']
missing = [k for k in required if std_cols[k] is None]
if missing:
    raise ValueError(f"Missing required columns in CSV for: {missing}\nFound columns: {list(df.columns)}")

# Keep a clean subset + rename to standard names
keep = [v for v in std_cols.values() if v is not None]
work = df[keep].rename(columns={
    std_cols['hash']: 'hash',
    std_cols['filename']: 'filename',
    std_cols['before']: 'before',
    std_cols['current']: 'after',
    **({std_cols['message']:'message'} if std_cols['message'] else {}),
    **({std_cols['diff']:'diff'} if std_cols['diff'] else {}),
    **({std_cols['llm_fix_type']:'llm_fix_type'} if std_cols['llm_fix_type'] else {}),
    **({std_cols['rectified_message']:'rectified_message'} if std_cols['rectified_message'] else {}),
}).copy()

# ---------- B. BASELINE DESCRIPTIVE STATS ----------
def file_ext(path):
    base = os.path.basename(str(path))
    _, ext = os.path.splitext(base)
    return ext.lower()

work['extension'] = work['filename'].apply(file_ext)

total_commits = work['hash'].nunique()
total_files = len(work)
avg_files_per_commit = work.groupby('hash')['filename'].nunique().mean()

fix_dist = (work['llm_fix_type'].value_counts(dropna=True, normalize=True)
            if 'llm_fix_type' in work.columns else pd.Series(dtype=float))
top_extensions = work['extension'].value_counts().head(10)

print("=== (b) Baseline descriptive statistics ===")
print(f"Total commits: {total_commits}")
print(f"Total files: {total_files}")
print(f"Average # modified files / commit: {avg_files_per_commit:.2f}")
if not fix_dist.empty:
    print("\nDistribution of LLM fix types:")
    print((fix_dist*100).round(2).astype(str) + '%')
print("\nMost frequent file extensions:")
print(top_extensions)

# ---------- C. STRUCTURAL METRICS WITH RADON ----------
from radon.metrics import mi_visit
from radon.raw import analyze as raw_analyze
from radon.complexity import cc_visit

def safe_mi(code: str):
    try:
        return float(mi_visit(code or ""))
    except Exception:
        return np.nan

def safe_loc(code: str):
    try:
        return int(raw_analyze(code or "").loc)
    except Exception:
        return np.nan

def safe_cc_mean(code: str):
    try:
        blocks = cc_visit(code or "")
        if not blocks:
            return 0.0
        return float(np.mean([b.complexity for b in blocks]))
    except Exception:
        return np.nan

for side, col in [('Before','before'),('After','after')]:
    work[f'MI_{side}']  = work[col].apply(safe_mi)
    work[f'LOC_{side}'] = work[col].apply(safe_loc)
    work[f'CC_{side}']  = work[col].apply(safe_cc_mean)

work['MI_Change']  = work['MI_After']  - work['MI_Before']
work['LOC_Change'] = work['LOC_After'] - work['LOC_Before']
work['CC_Change']  = work['CC_After']  - work['CC_Before']

# ---------- D. CHANGE MAGNITUDE METRICS ----------
# D1) Semantic similarity using CodeBERT
from transformers import AutoTokenizer, AutoModel
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
model.eval()

def embed_texts(texts, batch_size=8, max_len=512):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        with torch.no_grad():
            out = model(**{k:v.to(device) for k,v in tokens.items()})
            # CLS pooling
            cls = out.last_hidden_state[:,0,:]
            embs.append(cls.cpu())
    return torch.cat(embs, dim=0)

# Compute embeddings for before/after, then cosine similarity
before_emb = embed_texts(work['before'].fillna("").tolist())
after_emb  = embed_texts(work['after'].fillna("").tolist())

# cosine similarity row-wise
before_norm = torch.nn.functional.normalize(before_emb, dim=1)
after_norm  = torch.nn.functional.normalize(after_emb, dim=1)
cos = (before_norm * after_norm).sum(dim=1).numpy()
work['Semantic_Similarity'] = cos

# D2) Token similarity using BLEU (treat code tokens as space-split; BLEU expects tokens)
from sacrebleu.metrics import BLEU
bleu_metric = BLEU(tokenize='none')  # do not re-tokenize; we pass already-split tokens

def code_tokenize(s: str):
    # simple code-aware split: words or single non-space symbols
    return re.findall(r"\w+|[^\s\w]", s or "", re.UNICODE)

def sentence_bleu(hyp, ref):
    # sacrebleu's sentence BLEU via .corpus_score on single pair
    result = bleu_metric.corpus_score(
        [" ".join(hyp)],
        [[" ".join(ref)]]
    )
    # sacrebleu returns corpus BLEU in [0,100]; we scale to [0,1]
    return result.score / 100.0

token_bleus = []
for b, a in tqdm(zip(work['before'].fillna(""), work['after'].fillna("")),
                 total=len(work), desc="BLEU"):
    token_bleus.append(sentence_bleu(code_tokenize(a), code_tokenize(b)))
work['Token_Similarity'] = token_bleus

# ---------- E. CLASSIFICATION & AGREEMENT ----------
# You can adjust these thresholds if your TA suggests different values
SEM_MINOR_TH = 0.80
TOK_MINOR_TH = 0.75

work['Semantic_class'] = np.where(work['Semantic_Similarity'] >= SEM_MINOR_TH, 'Minor', 'Major')
work['Token_class']    = np.where(work['Token_Similarity']    >= TOK_MINOR_TH, 'Minor', 'Major')
work['Classes_Agree']  = np.where(work['Semantic_class'] == work['Token_class'], 'YES', 'NO')

# (Optional) If you must classify at COMMIT level (not file level),
# aggregate by hash (mean similarity → class). Uncomment next block:
# agg = work.groupby('hash').agg(
#     Semantic_Similarity=('Semantic_Similarity','mean'),
#     Token_Similarity=('Token_Similarity','mean'),
#     files=('filename','nunique')
# ).reset_index()
# agg['Semantic_class'] = np.where(agg['Semantic_Similarity'] >= SEM_MINOR_TH, 'Minor','Major')
# agg['Token_class']    = np.where(agg['Token_Similarity']    >= TOK_MINOR_TH, 'Minor','Major')
# agg['Classes_Agree']  = np.where(agg['Semantic_class']==agg['Token_class'], 'YES','NO')
# agg.to_csv("lab3_commit_level_classification.csv", index=False)

# ---------- F. SAVE FINAL TABLE ----------
OUT_PATH = "lab3_final_file_level.csv"
work.to_csv(OUT_PATH, index=False)
print(f"\nSaved final table → {OUT_PATH}")
