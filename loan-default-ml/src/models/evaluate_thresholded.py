import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix


def load_data(path):
    df = pd.read_csv(path)
    if 'Loan_Status' not in df.columns:
        raise ValueError('Expected column Loan_Status in data')
    y = df['Loan_Status']
    if y.dtype == object:
        y = y.map({'Y': 1, 'N': 0}).fillna(y)
    try:
        y = y.astype(int)
    except Exception:
        pass
    X = df.drop(columns=['Loan_Status'])
    return X, y


def evaluate(wrapper, X, y, seed=999, test_size=0.2):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, test_idx = next(sss.split(X, y))
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    pipeline = wrapper['pipeline']
    threshold = float(wrapper.get('threshold', 0.5))

    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    res = {
        'threshold': threshold,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'recall_0': float(recall_score(y_test, y_pred, pos_label=0)),
        'precision_0': float(precision_score(y_test, y_pred, pos_label=0, zero_division=0)),
        'f1_0': float(f1_score(y_test, y_pred, pos_label=0, zero_division=0)),
        'recall_1': float(recall_score(y_test, y_pred, pos_label=1)),
        'precision_1': float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        'f1_1': float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'n_test': int(len(y_test)),
    }
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    wrapper = joblib.load(model_path)
    X, y = load_data(args.data)

    res = evaluate(wrapper, X, y)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2)

    print(f"Evaluation saved to {out_path}")
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
