import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

SRC = Path("test_Y3wMUE5_7gLdaTN.csv")
CLEAN_OUT = Path("test_cleaned_with_features.csv")
PIPE_OUT = Path("loan_model_pipeline.joblib")
LABELED_OUT = Path("test_Y3wMUE5_7gLdaTN_labeled.csv")
TARGET_LABEL = "Loan_Status"  # Target column for loan approval status (auto-generated if missing)

def generate_demo_target(dataframe):
    """
    Create a synthetic target column for loan approval using simple heuristics.
    Numeric columns are coerced, dependents are standardized, and a rule-based target is generated.
    """
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")
    # Standardize dependents
    if "Dependents" in dataframe.columns:
        dataframe["Dependents"] = dataframe["Dependents"].replace("3+", 3)
        dataframe["Dependents"] = pd.to_numeric(dataframe["Dependents"], errors="coerce").fillna(0).astype(int)

    dataframe["ApplicantIncome"] = dataframe.get("ApplicantIncome", 0).fillna(0)
    dataframe["CoapplicantIncome"] = dataframe.get("CoapplicantIncome", 0).fillna(0)
    dataframe["TotalIncome"] = dataframe["ApplicantIncome"] + dataframe["CoapplicantIncome"]

    # Fill missing values for rule-based target
    if "LoanAmount" in dataframe.columns:
        dataframe["LoanAmount"] = dataframe["LoanAmount"].fillna(dataframe["LoanAmount"].median())
    if "Credit_History" in dataframe.columns:
        dataframe["Credit_History"] = dataframe["Credit_History"].fillna(0)

    # Calculate loan-to-income ratio
    denominator = dataframe["TotalIncome"].replace(0, np.nan)
    loan_ratio = dataframe["LoanAmount"] / denominator
    # Simple approval rule
    dataframe[TARGET_LABEL] = np.where((dataframe.get("Credit_History", 0) == 1) & (loan_ratio < 0.4), "Y", "N")
    dataframe[TARGET_LABEL] = dataframe[TARGET_LABEL].fillna("N")
    return dataframe

def load_and_label_data(filepath):
    """
    Load data from CSV and ensure target column exists. If missing, generate it and save a labeled copy.
    """
    df = pd.read_csv(filepath)
    if TARGET_LABEL not in df.columns:
        print(f"Target column '{TARGET_LABEL}' not found in {filepath}. Generating demo target.")
        df = generate_demo_target(df)
        try:
            df.to_csv(LABELED_OUT, index=False)
            print(f"Demo labels saved to: {LABELED_OUT}")
            print("Label distribution:\n", df[TARGET_LABEL].value_counts())
        except Exception as exc:
            print("Warning: could not save labeled file:", exc)
    return df

def remove_duplicates_and_coerce(df):
    """
    Remove duplicate rows and coerce numeric columns. Standardize dependents column.
    """
    df = df.drop_duplicates()
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", 3)
        df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce")
    return df

def engineer_features(df):
    """
    Create new features for modeling: total income, EMI, log transforms, and fill missing values.
    """
    df["ApplicantIncome"] = df["ApplicantIncome"].fillna(0)
    df["CoapplicantIncome"] = df["CoapplicantIncome"].fillna(0)
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    # Calculate EMI safely
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].replace(0, np.nan)
    df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"]
    df["EMI"] = df["EMI"].replace([np.inf, -np.inf], np.nan)
    # Log transforms for skewed features
    df["LogTotalIncome"] = np.log1p(df["TotalIncome"].clip(lower=0))
    df["LogLoanAmount"] = np.log1p(df["LoanAmount"].clip(lower=0))
    # Fill missing values for robustness
    for col in ["LoanAmount", "Loan_Amount_Term", "Credit_History"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].fillna(0).astype(int)
    return df

def get_onehot_encoder():
    """
    Return a OneHotEncoder instance, handling sklearn version differences.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore")

def train_and_save_model(df):
    """
    Build preprocessing pipeline, train RandomForest model, evaluate, and save outputs.
    """
    if TARGET_LABEL not in df.columns:
        print(f"ERROR: target column '{TARGET_LABEL}' missing after preprocessing.")
        sys.exit(1)

    features = df.drop(columns=[TARGET_LABEL, "Loan_ID"] if "Loan_ID" in df.columns else [TARGET_LABEL])
    target = df[TARGET_LABEL].map({"Y": 1, "N": 0}).fillna(df[TARGET_LABEL]).astype(int)

    numeric_features = [c for c in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "TotalIncome", "EMI", "LogTotalIncome", "LogLoanAmount"] if c in features.columns]
    categorical_features = [c for c in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"] if c in features.columns]
    if "Dependents" in features.columns and "Dependents" not in numeric_features:
        numeric_features.append("Dependents")

    numeric_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", get_onehot_encoder())
    ])
    preprocessing = ColumnTransformer([
        ("num", numeric_transform, numeric_features),
        ("cat", categorical_transform, categorical_features)
    ], remainder="drop")

    pipeline = Pipeline([
        ("pre", preprocessing),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))
    ])

    try:
        print("Target distribution:\n", target.value_counts(normalize=True).round(3))
    except Exception:
        pass

    X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(pipeline, PIPE_OUT)
    df.to_csv(CLEAN_OUT, index=False)
    print(f"Saved pipeline to {PIPE_OUT}")
    print(f"Saved cleaned data to {CLEAN_OUT}")

def main():
    """
    Main entry point: load, clean, engineer features, train, and save model.
    """
    df = load_and_label_data(SRC)
    df = remove_duplicates_and_coerce(df)
    df = engineer_features(df)
    train_and_save_model(df)

if __name__ == "__main__":
    main()