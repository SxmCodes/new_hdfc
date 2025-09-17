import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_predictions(path):
    df = pd.read_csv(path)
    return df


def safe_map_status(s):
    if pd.isna(s):
        return s
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip()
    if s in ('Y', 'y', '1'):
        return 1
    if s in ('N', 'n', '0'):
        return 0
    return s


def main(pred_csv):
    p = Path(pred_csv)
    if not p.exists():
        print(f'File not found: {p}')
        return 2

    df = load_predictions(p)

    print('\nTop 10 rows:')
    print(df.head(10).to_string(index=False))

    # Basic counts
    print('\nColumns present:', ', '.join(df.columns))

    if 'pred_proba_1' in df.columns:
        print('\nPredicted probability (class 1) summary:')
        print(df['pred_proba_1'].describe().to_string())
        print('\nProbability quantiles:')
        print(df['pred_proba_1'].quantile([0, .01, .05, .25, .5, .75, .95, .99, 1.0]).to_string())

    if 'pred_label_threshold' in df.columns:
        print('\nPrediction counts (thresholded):')
        print(df['pred_label_threshold'].value_counts(dropna=False).to_string())

    # If true label present, compute metrics
    if 'Loan_Status' in df.columns:
        y_true = df['Loan_Status'].map({'Y': 1, 'N': 0}).fillna(df['Loan_Status'])
        y_true = y_true.apply(safe_map_status)
        if 'pred_label_threshold' in df.columns:
            y_pred = df['pred_label_threshold'].apply(safe_map_status)
        elif 'pred_proba_1' in df.columns:
            y_pred = (df['pred_proba_1'] >= 0.5).astype(int)
        else:
            print('\nNo prediction column found to evaluate against Loan_Status')
            return 0

        # drop rows where y_true is not 0/1
        mask = y_true.isin([0,1])
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        print('\nEvaluation vs true Loan_Status:')
        print('n_eval =', len(y_true))
        print('Accuracy:', accuracy_score(y_true, y_pred))
        print('\nClassification report:')
        print(classification_report(y_true, y_pred, digits=3))
        print('Confusion matrix:')
        print(confusion_matrix(y_true, y_pred))

        if 'pred_proba_1' in df.columns:
            print('\nMean predicted probability by true class:')
            print(df[mask].groupby(y_true)['pred_proba_1'].mean().to_string())

    else:
        print('\nNo true label `Loan_Status` in CSV — only prediction summaries shown.')

    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: inspect_predictions.py <predictions_csv>')
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
