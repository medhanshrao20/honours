import pandas as pd
import os

filepath = "Features/features_enhanced.parquet"
if not os.path.exists(filepath):
    # Try alternate path if not in current dir
    filepath = os.path.join("c:/Users/himas/ARCANA/Honors Phase 2", filepath)

try:
    df = pd.read_parquet(filepath)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nBasic column checks:")
    print("subject unique:", df['subject'].nunique())
    print("conditions:", df['condition'].value_counts())
    print("modalities:", df['modality'].value_counts() if 'modality' in df.columns else "N/A")

    print("\nSubjects × modalities table:")
    if 'modality' in df.columns:
        print(pd.crosstab(df['subject'], df['modality']))
    else:
        print("Modality column not found")
except Exception as e:
    print(f"Error: {e}")
