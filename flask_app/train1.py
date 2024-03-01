# Step 1: Dataset Selection
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

# Load the Boston Housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Step 2: Feature Engineering and Preprocessing
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Model Selection
# Initialize the model
model = GradientBoostingRegressor()

# Step 4: Model Training and Evaluation
# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

# Step 5: Deployment (Saving and Loading the Model)
# Save the trained model using pickle
with open("housing_price_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Example of loading the model
with open("housing_price_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

# Making a prediction with the loaded model (example)
sample_data = X_test_scaled[0].reshape(1, -1)  # Reshape for a single sample
predicted_price = loaded_model.predict(sample_data)
print(f"Predicted housing price for the sample: {predicted_price[0]}")
