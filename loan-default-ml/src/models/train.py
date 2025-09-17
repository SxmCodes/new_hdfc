import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import make_scorer, recall_score, f1_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def _make_ohe():
	try:
		return OneHotEncoder(handle_unknown='ignore', sparse=False)
	except TypeError:
		try:
			return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
		except TypeError:
			return OneHotEncoder(handle_unknown='ignore')

def load_clean(path):
	df = pd.read_csv(path)
	if 'Loan_Status' not in df.columns:
		raise SystemExit('ERROR: cleaned file must include Loan_Status column')
	return df

def build_pipeline(num_cols, cat_cols):
	num_pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
	cat_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('ohe', _make_ohe())])
	pre = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], remainder='drop')
	clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, tree_method='hist')
	pipe = ImbPipeline([('pre', pre), ('smote', SMOTE(random_state=42)), ('clf', clf)])
	return pipe

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', '-d', default=str(Path('d:/Mdel Trained/test_cleaned_with_features.csv')))
	parser.add_argument('--out', '-o', default=str(Path('d:/Mdel Trained/loan-default-ml/models/best_model.joblib')))
	parser.add_argument('--niter', type=int, default=40)
	parser.add_argument('--scoring', type=str, default='roc_auc', help='Scoring metric for RandomizedSearchCV')
	parser.add_argument('--drop-credit-history', action='store_true', help='Drop Credit_History from features to avoid leakage')
	args = parser.parse_args()

	df = load_clean(args.data)
	# map target
	y = df['Loan_Status'].map({'Y':1, 'N':0}).fillna(df['Loan_Status']).astype(int)
	X = df.drop(columns=['Loan_Status', 'Loan_ID'], errors='ignore')
	if args.drop_credit_history and 'Credit_History' in X.columns:
		X = X.drop(columns=['Credit_History'])

	# split numeric/categorical
	num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
	cat_cols = [c for c in X.columns if c not in num_cols]

	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

	pipeline = build_pipeline(num_cols, cat_cols)

	param_dist = {
		'clf__n_estimators': [100, 200, 400],
		'clf__max_depth': [3, 6, 9],
		'clf__learning_rate': [0.01, 0.03, 0.1],
		'clf__subsample': [0.6, 0.8, 1.0],
		'clf__colsample_bytree': [0.6, 0.8, 1.0],
		# tune class imbalance handling and SMOTE sampling
		'clf__scale_pos_weight': [1, 2, 4, 8],
		'smote__sampling_strategy': [0.4, 0.6, 0.8, 1.0]
	}

	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	# support special scoring aliases
	scoring = args.scoring
	if scoring == 'recall_0':
		scoring = make_scorer(recall_score, pos_label=0)
	elif scoring == 'f1_0':
		scoring = make_scorer(f1_score, pos_label=0)

	rs = RandomizedSearchCV(pipeline, param_dist, n_iter=args.niter, cv=cv, scoring=scoring, n_jobs=-1, verbose=2, random_state=42)
	rs.fit(X_train, y_train)

	print('Best CV score:', rs.best_score_)
	print('Best params:', rs.best_params_)

	best = rs.best_estimator_
	y_pred = best.predict(X_test)
	print('Test accuracy:', accuracy_score(y_test, y_pred))
	try:
		print('ROC AUC:', roc_auc_score(y_test, best.predict_proba(X_test)[:,1]))
	except Exception:
		pass
	print(classification_report(y_test, y_pred))
	print(confusion_matrix(y_test, y_pred))

	Path(args.out).parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(best, args.out)
	print('Saved model to', args.out)

if __name__ == '__main__':
	main()