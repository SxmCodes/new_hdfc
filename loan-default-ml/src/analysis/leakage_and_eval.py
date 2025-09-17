import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif


DATA_PATH = Path(r"d:\Mdel Trained\test_cleaned_with_features.csv")
print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Rows, cols:", df.shape)
print("Columns:\n", df.dtypes)

# Basic target checks
TARGET = 'Loan_Status'
if TARGET not in df.columns:
    raise SystemExit(f"Target column {TARGET} not found")

print('\nTarget distribution:')
print(df[TARGET].value_counts(dropna=False))

# Convert Y/N to 1/0
y = df[TARGET].map({'Y':1,'N':0})
if y.isnull().any():
    print('Warning: target has unexpected values: ', df[TARGET].unique())

# Duplicates
dups = df.duplicated().sum()
print(f"\nFull-row duplicates: {dups}")

# Check Loan_ID uniqueness
if 'Loan_ID' in df.columns:
    n_unique = df['Loan_ID'].nunique()
    print('Loan_ID unique count:', n_unique)
    if n_unique != len(df):
        print('Warning: Loan_ID not unique -> potential duplicated rows')

# Nulls
print('\nMissing values per column:')
print(df.isnull().sum())

# Candidate features
feature_cols = [c for c in df.columns if c not in ['Loan_ID', TARGET]]
print('\nFeature cols:', feature_cols)

# For categorical columns, check if any category maps perfectly to one class
cat_cols = df[feature_cols].select_dtypes(include=['object','category']).columns.tolist()
num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print('\nCategorical cols:', cat_cols)
print('Numeric cols:', num_cols)

print('\nCategories that map perfectly to a single target class:')
perfect_cats = []
for c in cat_cols:
    groups = df.groupby(c)[TARGET].nunique()
    perfect = groups[groups == 1]
    if not perfect.empty:
        print(f" - {c}: {len(perfect)} category values with single-class mapping")
        perfect_cats.append((c, perfect.index.tolist()[:5]))

# For numeric features, fit a decision stump per feature to see if any single feature perfectly separates target
print('\nDecision-stump accuracy per numeric feature:')
perfect_nums = []
for c in num_cols:
    Xc = df[[c]].fillna(df[c].median())
    clf = DecisionTreeClassifier(max_depth=1, random_state=0)
    try:
        clf.fit(Xc, y)
        acc = clf.score(Xc, y)
        if acc == 1.0:
            perfect_nums.append(c)
        print(f" - {c}: {acc:.3f}")
    except Exception as e:
        print(f" - {c}: error {e}")

if perfect_cats or perfect_nums:
    print('\nPotential perfect predictors found. This indicates leakage or trivial mappings.')
else:
    print('\nNo single-feature perfect predictors detected.')

# Mutual information (numeric/categorical mixed)
print('\nTop mutual information scores:')
X = df[feature_cols].copy()
# Simple preprocessing for mutual info: numeric fillna, categorical -> codes
for c in X.columns:
    if X[c].dtype == 'object':
        X[c] = X[c].fillna('NA').astype('category').cat.codes
    else:
        X[c] = X[c].fillna(X[c].median())
mi = mutual_info_classif(X, y.fillna(0), random_state=0)
mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print(mi_scores.head(10))

# Quick strict holdout evaluation
print('\nHoldout evaluation (stratified 80/20)')
X_full = df[feature_cols].copy()
y_full = y.fillna(0)
# Build preprocessing
num_features = [c for c in X_full.columns if X_full[c].dtype != 'object']
cat_features = [c for c in X_full.columns if X_full[c].dtype == 'object']
num_pipeline = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
cat_pipeline = Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])
pre = ColumnTransformer([('num', num_pipeline, num_features), ('cat', cat_pipeline, cat_features)])

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)
print('Train/test sizes:', X_train.shape, X_test.shape)

pipe = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=1000))])
# cross-validated score on whole dataset
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
cv_scores = cross_val_score(pipe, X_full, y_full, cv=cv, scoring='accuracy')
print('CV accuracy (LogReg):', cv_scores, 'mean:', cv_scores.mean())

# Fit on train and eval on test
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('Holdout accuracy (LogReg):', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

print('\nDone diagnostics.')
