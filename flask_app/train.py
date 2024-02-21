# Import necessary libraries
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import pickle

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model using pickle
with open("logistic_regression_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("logistic_regression_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

y_pred = loaded_model.predict(X_test)
print(X_test)
print(y_pred)
print(y_test)

# # Optional: You can also save the scaler if needed
# with open("scaler.pkl", "wb") as scaler_file:
#     pickle.dump(scaler, scaler_file)

# # Make predictions on the test set
# y_pred = model.predict(X_test)
