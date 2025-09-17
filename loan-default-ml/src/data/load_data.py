import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the data
data = pd.read_csv('d:\\Mdel Trained\\test_cleaned_with_features.csv')

# Step 2: Preprocess the data
# Handle missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(data[['Gender', 'Married', 'Education', 'Property_Area']]).toarray()

# Create a DataFrame for the encoded features
encoded_df = pd.DataFrame(encoded_features)

# Combine with the original DataFrame (excluding the original categorical columns)
data = pd.concat([data.drop(['Gender', 'Married', 'Education', 'Property_Area'], axis=1), encoded_df], axis=1)

# Step 3: Define features and target variable
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status'].map({'Y': 1, 'N': 0})  # Convert target to binary

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Save the model
import joblib
joblib.dump(model, 'loan_model.pkl')