import pandas as pd
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Ensure graphs folder exists ----------
os.makedirs("graphs", exist_ok=True)

# ---------- B. BASELINE DESCRIPTIVE STATISTICS ----------
CSV_PATH = "/home/set-iitgn-vm/Desktop/stt_lab/part_d_diff_analysis.csv"

df = pd.read_csv(CSV_PATH)

# Helper to get file extension
def file_ext(path):
    base = os.path.basename(str(path))
    _, ext = os.path.splitext(base)
    return ext.lower()

df['extension'] = df['Filename'].apply(file_ext)

# 1. Total number of commits and files
total_commits = df['Hash'].nunique()
total_files = len(df)

# 2. Average number of modified files per commit
avg_files_per_commit = df.groupby('Hash')['Filename'].nunique().mean()

# 3. Distribution of fix types
if 'LLM Inference (fix type)' in df.columns:
    fix_dist = df['LLM Inference (fix type)'].value_counts(normalize=True) * 100
else:
    fix_dist = None

# 4. Most frequent file extensions
top_extensions = df['extension'].value_counts().head(10)

# ---- Print Results ----
print("=== Part B: Baseline Descriptive Statistics ===\n")
summary_data = [
    ["Total commits", total_commits],
    ["Total files", total_files],
    ["Average # of modified files per commit", f"{avg_files_per_commit:.2f}"],
]
print(tabulate(summary_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))

if fix_dist is not None:
    print("\nDistribution of Fix Types (LLM Inference):")
    print(tabulate(fix_dist.reset_index().values, headers=["Fix Type", "Percentage"], tablefmt="fancy_grid"))
else:
    print("\nNo 'LLM Inference (fix type)' column found in CSV.\n")

print("\nMost Frequently Modified File Extensions:")
print(tabulate(top_extensions.reset_index().values, headers=["Extension", "Count"], tablefmt="fancy_grid"))

# ---------- GRAPHS ----------

# 1. Pie Chart: Distribution of Fix Types
if fix_dist is not None:
    plt.figure(figsize=(6, 6))
    plt.pie(fix_dist, labels=fix_dist.index, autopct='%1.1f%%', startangle=140)
    plt.title("Distribution of Fix Types (LLM Inference)")
    plt.savefig("graphs/fix_type_distribution_pie.png")
    plt.show()

# 2. Bar Chart: Top 10 Most Frequent File Extensions
plt.figure(figsize=(8, 6))
sns.barplot(x=top_extensions.values, y=top_extensions.index, palette="Blues_r")
plt.title("Top 10 Most Frequently Modified File Extensions")
plt.xlabel("Count")
plt.ylabel("File Extension")
plt.savefig("graphs/top_file_extensions_bar.png")
plt.show()
