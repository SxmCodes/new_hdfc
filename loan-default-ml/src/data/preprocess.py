import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

def find_raw(default_names=None):
	root = Path(__file__).resolve().parents[2]
	candidates = []
	if default_names:
		candidates.extend(default_names)
	# common locations
	candidates.append(root / 'data' / 'raw' / 'test_Y3wMUE5_7gLdaTN.csv')
	candidates.append(Path('d:/Mdel Trained/test_Y3wMUE5_7gLdaTN.csv'))
	for p in candidates:
		p = Path(p)
		if p.exists():
			return p
	raise FileNotFoundError('raw CSV not found in expected locations; provide --input')

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
	# Drop exact duplicates
	df = df.drop_duplicates().copy()

	# Standardize column names
	df.columns = [c.strip() for c in df.columns]

	# Numeric coercion
	for c in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]:
		if c in df.columns:
			df[c] = pd.to_numeric(df[c], errors='coerce')

	# Dependents: '3+' -> 3
	if 'Dependents' in df.columns:
		df['Dependents'] = df['Dependents'].replace('3+', 3)
		df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')

	# Impute simple missing values
	# categorical: fill with mode
	cat_cols = [c for c in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'] if c in df.columns]
	for c in cat_cols:
		if df[c].isna().any():
			df[c] = df[c].fillna(df[c].mode().iloc[0])

	# numeric: median
	num_cols = [c for c in ['LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome', 'CoapplicantIncome', 'Credit_History'] if c in df.columns]
	for c in num_cols:
		if df[c].isna().any():
			df[c] = df[c].fillna(df[c].median())

	# Feature engineering
	df['ApplicantIncome'] = df.get('ApplicantIncome', 0).fillna(0)
	df['CoapplicantIncome'] = df.get('CoapplicantIncome', 0).fillna(0)
	df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
	# EMI: avoid zero division
	df['Loan_Amount_Term'] = df.get('Loan_Amount_Term').replace(0, np.nan)
	df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
	df['EMI'] = df['EMI'].replace([np.inf, -np.inf], np.nan).fillna(df['EMI'].median())
	df['LogTotalIncome'] = np.log1p(df['TotalIncome'].clip(lower=0))
	df['LogLoanAmount'] = np.log1p(df['LoanAmount'].clip(lower=0))
	df['DebtIncomeRatio'] = df['LoanAmount'] / df['TotalIncome'].replace(0, np.nan)
	df['DebtIncomeRatio'] = df['DebtIncomeRatio'].replace([np.inf, -np.inf], np.nan).fillna(df['DebtIncomeRatio'].median())

	# Dependents fill
	if 'Dependents' in df.columns:
		df['Dependents'] = df['Dependents'].fillna(0).astype(int)

	return df

def main():
	parser = argparse.ArgumentParser(description='Preprocess raw loan CSV and save cleaned file')
	parser.add_argument('--input', '-i', default=None, help='Raw CSV path')
	parser.add_argument('--output', '-o', default=None, help='Output cleaned CSV path')
	args = parser.parse_args()

	raw = Path(args.input) if args.input else find_raw()
	out_default = Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'test_cleaned_with_features.csv'
	out_root = Path('d:/Mdel Trained/test_cleaned_with_features.csv')
	out = Path(args.output) if args.output else out_default

	print('Loading raw data from', raw)
	df = pd.read_csv(raw)
	df = clean_and_engineer(df)

	out.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(out, index=False)
	print('Saved cleaned data to', out)

	# also save copy to root path for backwards compatibility
	try:
		out_root.parent.mkdir(parents=True, exist_ok=True)
		df.to_csv(out_root, index=False)
		print('Also saved cleaned copy to', out_root)
	except Exception:
		pass

	# save a small metadata file
	meta = {
		'rows': int(df.shape[0]),
		'columns': df.columns.tolist()
	}
	joblib.dump(meta, out.with_suffix('.meta.joblib'))

if __name__ == '__main__':
	main()