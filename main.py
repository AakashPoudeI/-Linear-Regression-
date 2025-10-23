"""
Linear Regression - Housing Price Prediction
Dataset: https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Housing Price Prediction - Linear Regression")
print("="*60)

# Load the Dataset
print("\n1. Loading Dataset")


# Load the data 
df = pd.read_csv('Housing.csv')

print(f"\nDataset loaded successfully!")
print(f"\nShape: {df.shape}")
print(f"\nRows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\nFirst few rows:")
print(df.head(10))

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# Data Preprocessing
print("\n2. Data Preprocessing")


# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check for categorical columns
print("\nChecking data info:")
print(df.info())

# Convert categorical variables to numeric (if any)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if categorical_cols:
    print(f"\nCategorical columns found: {categorical_cols}")
    print("Converting to numeric (yes=1, no=0)...")
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].map({'yes': 1, 'no': 0})
            print(f"  Converted {col}")

print("\nData after conversion:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())

# Target variable
target = 'price'
print(f"\nTarget variable: {target}")

# Exploratory Visualization
print("\n3. Data Visualization")

# Distribution of target variable (price)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df[target], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices')

plt.subplot(1, 2, 2)
plt.boxplot(df[target])
plt.ylabel('Price')
plt.title('House Price Boxplot')

plt.tight_layout()
plt.savefig('price_distribution.png')
print("Saved: price_distribution.png")
plt.show()

# Selecting numeric features 
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove(target)

print(f"\nNumeric features: {numeric_features}")

# correlation with price
correlations = df[numeric_features + [target]].corr()[target].sort_values(ascending=False)
print("\nCorrelation with price:")
print(correlations)

# Top features based on correlation
top_features = correlations.abs().sort_values(ascending=False)[1:7].index.tolist()
print(f"\nTop features selected: {top_features}")

# Feature relationships with price
n_features = len(top_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

plt.figure(figsize=(15, 5 * n_rows))

for i, feature in enumerate(top_features, 1):
    plt.subplot(n_rows, n_cols, i)
    plt.scatter(df[feature], df[target], alpha=0.5, color='blue')
    plt.xlabel(feature)
    plt.ylabel('Price')
    plt.title(f'Price vs {feature}')

plt.tight_layout()
plt.savefig('feature_relationships.png')
print("Saved: feature_relationships.png")
plt.show()

# Simple Linear Regression
print("\n4. Simple Linear Regression")
print("-"*60)

# Use the feature with highest correlation
simple_feature = top_features[0]
print(f"Using feature: {simple_feature}")

X_simple = df[[simple_feature]]
y = df[target]

# Split data
X_train_simple, X_test_simple, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train_simple)}")
print(f"Testing samples: {len(X_test_simple)}")

# Train model
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)

print("\nModel trained!")
print(f"Coefficient: {model_simple.coef_[0]:.2f}")
print(f"Intercept: {model_simple.intercept_:.2f}")
print(f"\nEquation: Price = {model_simple.coef_[0]:.2f} × {simple_feature} + {model_simple.intercept_:.2f}")

# Predictions
y_pred_simple = model_simple.predict(X_test_simple)

# Evaluation
mae_simple = mean_absolute_error(y_test, y_pred_simple)
mse_simple = mean_squared_error(y_test, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)
r2_simple = r2_score(y_test, y_pred_simple)

print("\nModel Performance:")
print(f"MAE: {mae_simple:,.2f}")
print(f"MSE: {mse_simple:,.2f}")
print(f"RMSE: {rmse_simple:,.2f}")
print(f"R² Score: {r2_simple:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_test_simple, y_test, color='blue', alpha=0.5, label='Actual Price')
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2, label='Regression Line')
plt.xlabel(simple_feature)
plt.ylabel('Price')
plt.title(f'Simple Linear Regression: Price vs {simple_feature}')
plt.legend()
plt.savefig('simple_regression.png')
print("\nSaved: simple_regression.png")
plt.show()

# Multiple Linear Regression
print("\n5. Multiple Linear Regression")
print("-"*60)

# Use top features
X_multiple = df[top_features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_multiple, y, test_size=0.2, random_state=42
)

print(f"Features used: {top_features}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train model
model_multiple = LinearRegression()
model_multiple.fit(X_train, y_train)

print("\nModel trained!")
print("\nCoefficients:")
for feature, coef in zip(top_features, model_multiple.coef_):
    print(f"  {feature}: {coef:.2f}")
print(f"Intercept: {model_multiple.intercept_:.2f}")

# Predictions
y_pred_multiple = model_multiple.predict(X_test)

# Evaluation
mae_multiple = mean_absolute_error(y_test, y_pred_multiple)
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
rmse_multiple = np.sqrt(mse_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)

print("\nModel Performance:")
print(f"MAE: {mae_multiple:,.2f}")
print(f"MSE: {mse_multiple:,.2f}")
print(f"RMSE: {rmse_multiple:,.2f}")
print(f"R² Score: {r2_multiple:.4f}")

# Actual vs Predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_multiple, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         color='red', linewidth=2, linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.legend()
plt.savefig('multiple_regression.png')
print("\nSaved: multiple_regression.png")
plt.show()

# Residual plot
residuals = y_test - y_pred_multiple

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_multiple, residuals, color='purple', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot - Multiple Regression')
plt.savefig('residuals.png')
print("Saved: residuals.png")
plt.show()

# Model Comparison
print("\n6. Model Comparison")

comparison = pd.DataFrame({
    'Model': ['Simple (1 feature)', 'Multiple (All features)'],
    'R² Score': [r2_simple, r2_multiple],
    'RMSE': [rmse_simple, rmse_multiple],
    'MAE': [mae_simple, mae_multiple]
})

print(comparison.to_string(index=False))

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models = ['Simple', 'Multiple']
r2_scores = [r2_simple, r2_multiple]
rmse_scores = [rmse_simple, rmse_multiple]

axes[0].bar(models, r2_scores, color=['blue', 'green'])
axes[0].set_ylabel('R² Score')
axes[0].set_title('R² Score Comparison')
axes[0].set_ylim([0, 1])

axes[1].bar(models, rmse_scores, color=['blue', 'green'])
axes[1].set_ylabel('RMSE')
axes[1].set_title('RMSE Comparison')

plt.tight_layout()
plt.savefig('model_comparison.png')
print("\nSaved: model_comparison.png")
plt.show()

if r2_multiple > r2_simple:
    improvement = ((r2_multiple - r2_simple) / r2_simple) * 100
    print(f"\nMultiple regression is better by {improvement:.2f}%")

# Feature Importance
print("\n7. Feature Importance Analysis")
print("-"*60)

# Absolute coefficients
coef_df = pd.DataFrame({
    'Feature': top_features,
    'Coefficient': model_multiple.coef_,
    'Abs_Coefficient': np.abs(model_multiple.coef_)
})

coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
print("\nFeatures sorted by importance:")
print(coef_df.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='teal')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Coefficients in Multiple Regression')
plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
plt.savefig('feature_importance.png')
print("\nSaved: feature_importance.png")
plt.show()

# Making Predictions
print("\n8. Making New Predictions")
print("-"*60)

# Use mean values as example
example_house = pd.DataFrame()
for feature in top_features:
    example_house[feature] = [df[feature].mean()]

predicted_price = model_multiple.predict(example_house)

print("\nExample House (using mean values):")
print(example_house.to_string(index=False))
print(f"\nPredicted Price: {predicted_price[0]:,.2f}")

# Key Insights
print("\n9. Key Insights")

print("\nCoefficient Interpretation:")
for feature, coef in zip(top_features, model_multiple.coef_):
    print(f"• {feature}: Each unit increase changes price by {coef:,.2f}")

print(f"\nModel explains {r2_multiple*100:.2f}% of price variation")

if r2_multiple > 0.7:
    print("This is a good model for prediction!")
elif r2_multiple > 0.5:
    print("This is a decent model, but could be improved")
else:
    print("Model needs improvement")

print("Linear Regression Analysis Complete!")
