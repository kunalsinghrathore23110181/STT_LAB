import pandas as pd
from tabulate import tabulate  # <-- NEW

# Load final data
df = pd.read_csv("/home/set-iitgn-vm/Desktop/stt_lab/part_e_classification.csv")

# Keep only final required columns
final_cols = [
    'Hash', 'Filename',
    'MI_Change', 'CC_Change', 'LOC_Change',
    'Semantic_Similarity', 'Token_Similarity',
    'Semantic_class', 'Token_class', 'Classes_Agree'
]
final_df = df[final_cols]

# Save final output
final_df.to_csv("lab3_final_file_level.csv", index=False)
print("Part F Completed! Final table saved to lab3_final_file_level.csv")

# ---------- PRINT in Terminal ----------
print("\n=== Part F Final Output (First 10 rows) ===")
print(tabulate(final_df.head(10), headers='keys', tablefmt='fancy_grid', showindex=False))
