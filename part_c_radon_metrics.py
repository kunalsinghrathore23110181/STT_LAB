import pandas as pd
import numpy as np
from radon.metrics import mi_visit
from radon.raw import analyze as raw_analyze
from radon.complexity import cc_visit

# ---------- C. STRUCTURAL METRICS ----------
CSV_PATH = "/home/set-iitgn-vm/Desktop/stt_lab/part_d_diff_analysis.csv"  # Replace with your dataset CSV
df = pd.read_csv(CSV_PATH)

# --- Helper functions for safe metric extraction ---
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

# Calculate metrics for before and after code
df['MI_Before']  = df['Source Code (before)'].apply(safe_mi)
df['MI_After']   = df['Source Code (current)'].apply(safe_mi)

df['LOC_Before'] = df['Source Code (before)'].apply(safe_loc)
df['LOC_After']  = df['Source Code (current)'].apply(safe_loc)

df['CC_Before']  = df['Source Code (before)'].apply(safe_cc_mean)
df['CC_After']   = df['Source Code (current)'].apply(safe_cc_mean)

# Compute changes
df['MI_Change']  = df['MI_After']  - df['MI_Before']
df['LOC_Change'] = df['LOC_After'] - df['LOC_Before']
df['CC_Change']  = df['CC_After']  - df['CC_Before']

# Save the updated CSV
df.to_csv("part_c_structural_metrics.csv", index=False)

print("=== Part C Completed ===")
print("Metrics saved to part_c_structural_metrics.csv")
print(df[['Filename', 'MI_Change', 'LOC_Change', 'CC_Change']].head())
