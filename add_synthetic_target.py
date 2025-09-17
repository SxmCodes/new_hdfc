import pandas as pd
import numpy as np
from pathlib import Path

SRC = Path(r"d:\Mdel Trained\test_Y3wMUE5_7gLdaTN.csv")
OUT = Path(r"d:\Mdel Trained\test_Y3wMUE5_7gLdaTN_labeled.csv")

df = pd.read_csv(SRC)

# basic numeric coercion
for c in ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Dependents clean
if "Dependents" in df.columns:
    df["Dependents"] = df["Dependents"].replace("3+", 3)
    df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce").fillna(0).astype(int)

# feature for heuristic
df["ApplicantIncome"] = df["ApplicantIncome"].fillna(0)
df["CoapplicantIncome"] = df["CoapplicantIncome"].fillna(0)
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

# fill LoanAmount/credit history missing for rule
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Credit_History"] = df["Credit_History"].fillna(0)

# Heuristic label:
# Approve if credit history present AND loan amount is not large relative to income
# ratio threshold can be tuned; this is only for demo purposes.
ratio = df["LoanAmount"] / (df["TotalIncome"].replace(0, np.nan))
df["Loan_Status"] = np.where((df["Credit_History"] == 1) & (ratio < 0.4), "Y", "N")
df["Loan_Status"] = df["Loan_Status"].fillna("N")

# show simple stats
print("Generated labels counts:\n", df["Loan_Status"].value_counts())

# save new file (does not overwrite original)
df.to_csv(OUT, index=False)
print("Saved labeled file to:", OUT)