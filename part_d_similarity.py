import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sacrebleu.metrics import BLEU
import re
from tqdm import tqdm
from tabulate import tabulate  # <-- NEW
import os

# Create 'graphs' folder automatically if it doesn't exist
os.makedirs("graphs", exist_ok=True)

# Load dataset
CSV_PATH = "/home/set-iitgn-vm/Desktop/stt_lab/part_c_structural_metrics.csv"
df = pd.read_csv(CSV_PATH)

# ---------- D1. Semantic Similarity using CodeBERT ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
model.eval()

def embed_texts(texts, batch_size=8, max_len=512):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Code"):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in tokens.items()})
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embeddings.cpu())
    return torch.cat(embeddings, dim=0)

# Fill NA to avoid errors
df['Source Code (before)'] = df['Source Code (before)'].fillna("")
df['Source Code (current)'] = df['Source Code (current)'].fillna("")

# Compute embeddings
before_emb = embed_texts(df['Source Code (before)'].tolist())
after_emb = embed_texts(df['Source Code (current)'].tolist())

# Cosine similarity
before_norm = torch.nn.functional.normalize(before_emb, dim=1)
after_norm = torch.nn.functional.normalize(after_emb, dim=1)
cosine_similarity = (before_norm * after_norm).sum(dim=1).numpy()
df['Semantic_Similarity'] = cosine_similarity

print("\nSemantic Similarity Computed!")

# ---------- D2. Token Similarity using BLEU ----------
bleu = BLEU(tokenize='none')

def code_tokenize(code):
    return re.findall(r"\w+|[^\s\w]", code or "", re.UNICODE)

def calculate_bleu(before_code, after_code):
    hyp = " ".join(code_tokenize(after_code))
    ref = [" ".join(code_tokenize(before_code))]
    score = bleu.corpus_score([hyp], [ref])
    return score.score / 100.0  # scale to 0-1

token_scores = []
for before, after in tqdm(zip(df['Source Code (before)'], df['Source Code (current)']),
                         total=len(df), desc="Calculating BLEU"):
    token_scores.append(calculate_bleu(before, after))

df['Token_Similarity'] = token_scores

# Save updated CSV
df.to_csv("part_d_change_magnitude.csv", index=False)
print("\nPart D Completed! Results saved to part_d_change_magnitude.csv")

# ---------- PRINT in Terminal ----------
print("\n=== Part D Output (First 10 rows) ===")
print(tabulate(df[['Hash', 'Filename', 'MI_Change', 'CC_Change', 'LOC_Change', 'Semantic_Similarity', 'Token_Similarity']].head(10),
               headers='keys', tablefmt='fancy_grid', showindex=False))

# ---------- Plot Similarity Distributions ----------
plt.figure(figsize=(8, 5))
sns.histplot(df['Semantic_Similarity'], bins=20, kde=True, color='blue')
plt.title("Semantic Similarity Distribution")
plt.xlabel("Semantic Similarity")
plt.ylabel("Frequency")
plt.savefig("graphs/semantic_similarity_distribution.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['Token_Similarity'], bins=20, kde=True, color='green')
plt.title("Token Similarity Distribution")
plt.xlabel("Token Similarity")
plt.ylabel("Frequency")
plt.savefig("graphs/token_similarity_distribution.png")
plt.show()
