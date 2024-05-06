import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the Breast Cancer Wisconsin dataset
breastCancer = load_breast_cancer()
X = breastCancer.data
y = breastCancer.target
feature_names = breastCancer.feature_names

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit a logistic regression model
model = LogisticRegression()
model.fit(X_scaled, y)

# Extract coefficients
coefficients = model.coef_[0]

# Calculate odds ratios (exponential of coefficients)
odds_ratios = np.exp(coefficients)

# Create a DataFrame to display results
results_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Odds Ratio': odds_ratios
})

# Sort features by absolute coefficient magnitude
results_df['Absolute Coefficient'] = np.abs(results_df['Coefficient'])
results_df.sort_values(by='Absolute Coefficient', ascending=False, inplace=True)
results_df.reset_index(drop=True, inplace=True)

# Display the top influential features
top_features = results_df.head(10)  # Select top 10 features
print("Top Influential Features:")
print(top_features)
