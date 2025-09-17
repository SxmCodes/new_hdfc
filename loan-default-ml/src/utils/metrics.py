import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv('d:\\Mdel Trained\\test_cleaned_with_features.csv')

# Preprocess the data
# Handle missing values (example: fill with mode for categorical, mean for numerical)
data.fillna(data.mode().iloc[0], inplace=True)

# Encode categorical variables
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# One-hot encoding for categorical features
X = pd.get_dummies(X, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))