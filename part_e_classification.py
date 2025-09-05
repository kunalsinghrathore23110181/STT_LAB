import pandas as pd
import numpy as np
from tabulate import tabulate  # <-- NEW
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
CSV_PATH = "/home/set-iitgn-vm/Desktop/stt_lab/part_d_change_magnitude.csv"
df = pd.read_csv(CSV_PATH)

# Thresholds
SEM_THRESHOLD = 0.80
TOK_THRESHOLD = 0.75

# Classification
df['Semantic_class'] = np.where(df['Semantic_Similarity'] >= SEM_THRESHOLD, 'Minor', 'Major')
df['Token_class'] = np.where(df['Token_Similarity'] >= TOK_THRESHOLD, 'Minor', 'Major')

# Agreement
df['Classes_Agree'] = np.where(df['Semantic_class'] == df['Token_class'], 'YES', 'NO')

# Save to CSV
df.to_csv("part_e_classification.csv", index=False)
print("Part E Completed! Classification saved to part_e_classification.csv")

# ---------- PRINT in Terminal ----------
print("\n=== Part E Output (First 10 rows) ===")
print(tabulate(df[['Hash', 'Semantic_Similarity', 'Token_Similarity', 'Semantic_class', 'Token_class', 'Classes_Agree']].head(10),
               headers='keys', tablefmt='fancy_grid', showindex=False))

# ---------- Plot Classification Counts ----------
plt.figure(figsize=(6, 6))
df['Semantic_class'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title("Semantic Classification (Major vs Minor)")
plt.ylabel("")
plt.savefig("graphs/semantic_class_pie.png")
plt.show()

plt.figure(figsize=(6, 6))
df['Token_class'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#99ff99','#ffcc99'])
plt.title("Token Classification (Major vs Minor)")
plt.ylabel("")
plt.savefig("graphs/token_class_pie.png")
plt.show()
