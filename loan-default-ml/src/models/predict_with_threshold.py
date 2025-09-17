import argparse
from pathlib import Path

import joblib
import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    if 'Loan_Status' in df.columns:
        X = df.drop(columns=['Loan_Status'])
    else:
        X = df
    return df, X


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='path to wrapper joblib (pipeline + threshold)')
    p.add_argument('--data', required=True, help='path to CSV to predict')
    p.add_argument('--out', required=True, help='output CSV path')
    args = p.parse_args()

    wrapper = joblib.load(args.model)
    pipeline = wrapper.get('pipeline', wrapper)
    threshold = float(wrapper.get('threshold', 0.5))

    df_orig, X = load_data(args.data)

    proba = pipeline.predict_proba(X)[:, 1]
    pred_label = (proba >= threshold).astype(int)

    out_df = df_orig.copy()
    out_df['pred_proba_1'] = proba
    out_df['pred_label_threshold'] = pred_label
    out_df['pred_label_threshold_str'] = out_df['pred_label_threshold'].map({1: 'Y', 0: 'N'})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    counts = out_df['pred_label_threshold'].value_counts().to_dict()
    print(f"Saved predictions to {out_path}")
    print("Prediction counts:", counts)


if __name__ == '__main__':
    main()
