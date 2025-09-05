import os, re, math
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- A. LOAD & NORMALIZE COLUMNS ----------
CSV_PATH = "/home/set-iitgn-vm/Desktop/stt_lab/part_d_diff_analysis.csv"  # <-- Replace with actual CSV file path

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

print("Loaded and normalized CSV successfully!")
print(work.head())
