import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('d:\\Mdel Trained\\test_cleaned_with_features.csv')

# Preprocess the data
# Handle missing values (example: fill with mode for categorical, mean for numerical)
data.fillna(data.mode().iloc[0], inplace=True)

# Encode categorical variables
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(data[['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']]).toarray()

# Prepare features and target variable
X = pd.concat([data.drop(columns=['Loan_Status']), pd.DataFrame(encoded_features)], axis=1)
y = data['Loan_Status'].map({'Y': 1, 'N': 0})  # Convert target to binary

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
import joblib
joblib.dump(model, 'loan_model.pkl')