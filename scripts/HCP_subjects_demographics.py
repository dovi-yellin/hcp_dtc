from os.path import join
import numpy as np
import pandas as pd

# ===============================
# 1️⃣  Provide inputs here
# ===============================

path_str = 'data'
csv_filename = "HCP_YA_subjects_2026_02_25_09_55_07.csv"  # <-- relative path to your demographics CSV

my_subjects = [
    100610, 102311, 102816, 104416, 105923, 108323, 109123, 111514, 114823, 115017, 115825, 116726, 118225, 125525, 126426, 128935, 130114, 130518, 131217, 131722, 132118, 134627, 134829, 135124, 137128, 140117, 144226, 145834, 146129, 146432, 146735, 146937, 148133, 150423, 155938, 156334, 157336, 158035, 158136, 159239
]

# ===============================
# 2️⃣  Load full dataset
# ===============================

df = pd.read_csv(join(path_str, csv_filename))

# Ensure Subject column is numeric
df["Subject"] = pd.to_numeric(df["Subject"], errors="coerce")

# ===============================
# 3️⃣  Extract subset
# ===============================

subset = df[df["Subject"].isin(my_subjects)][["Subject", "Gender", "Age"]]
subset = subset.sort_values("Subject")

print("\nMatched subjects:", len(subset), "out of", len(my_subjects))

# ===============================
# 4️⃣  Gender stats
# ===============================

gender_counts = subset["Gender"].value_counts()
print("\nGender counts:")
print(gender_counts)

# ===============================
# 5️⃣  Age stats
# ===============================

# Try converting age to numeric (works if age is continuous)
age_numeric = pd.to_numeric(subset["Age"], errors="coerce")

if age_numeric.notna().sum() > 0:
    print("\nAge statistics:")
    print("Mean age :", round(age_numeric.mean(), 2))
    print("Std age  :", round(age_numeric.std(), 2))
    print("Min age  :", age_numeric.min())
    print("Max age  :", age_numeric.max())
else:
    print("\nAge appears to be binned (e.g., '22-25').")
    print("Age distribution:")
    print(subset["Age"].value_counts())

# ===============================
# 6️⃣  Optional: save subset
# ===============================

subset.to_csv("hcp_40_subjects_demographics.csv", index=False)
print("\nSaved subset to: hcp_40_subjects_demographics.csv")