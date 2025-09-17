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
    # robust mapping
    if y.dtype == object:
        y = y.map({'Y': 1, 'N': 0}).fillna(y)
    try:
        y = y.astype(int)
    except Exception:
        pass
    X = df.drop(columns=['Loan_Status'])
    return X, y


def find_best_threshold(model, X, y, seed=42, test_size=0.2):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(sss.split(X, y))
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # predict probabilities for class 1
    proba = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.0, 1.0, 101)
    records = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        rec0 = recall_score(y_val, y_pred, pos_label=0)
        prec0 = precision_score(y_val, y_pred, pos_label=0, zero_division=0)
        f10 = f1_score(y_val, y_pred, pos_label=0, zero_division=0)
        acc = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred).tolist()
        records.append({
            'threshold': float(t),
            'recall_0': float(rec0),
            'precision_0': float(prec0),
            'f1_0': float(f10),
            'accuracy': float(acc),
            'confusion_matrix': cm,
        })

    # choose threshold maximizing recall_0, break ties with f1_0
    best = max(records, key=lambda r: (r['recall_0'], r['f1_0']))
    return best, records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='path to saved pipeline joblib')
    p.add_argument('--data', required=True, help='path to cleaned CSV with Loan_Status')
    p.add_argument('--out', required=True, help='output path for thresholded model (.joblib)')
    p.add_argument('--metrics', required=False, help='output path for metrics JSON', default=None)
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f'model not found: {model_path}')

    model = joblib.load(model_path)

    X, y = load_data(args.data)

    best, records = find_best_threshold(model, X, y)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save a small wrapper dict with pipeline and chosen threshold
    wrapped = {'pipeline': model, 'threshold': best['threshold']}
    joblib.dump(wrapped, out_path)

    metrics_out = args.metrics or str(out_path.with_suffix('.metrics.json'))
    with open(metrics_out, 'w') as f:
        json.dump({'best': best, 'records': records}, f, indent=2)

    print(f"Saved thresholded model to {out_path}")
    print(f"Best threshold: {best['threshold']}")
    print(json.dumps(best, indent=2))


if __name__ == '__main__':
    main()
