
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# Load the dataset
file_path = 'Facebook Metrics of Cosmetic Brand.xlsx'
data = pd.read_excel(file_path)

# Data Preprocessing
data['like'].fillna(0, inplace=True)
data['share'].fillna(0, inplace=True)
data['Paid'].fillna(data['Paid'].mode()[0], inplace=True)
data_encoded = pd.get_dummies(data, columns=['Type', 'Category'], drop_first=True)

# EDA - Exploratory Data Analysis
# Visualizing distribution of likes, comments, shares, and total interactions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram for Likes
axes[0, 0].hist(data['like'].dropna(), bins=30, color='blue', edgecolor='black')
axes[0, 0].set_title('Distribution of Likes')

# Histogram for Comments
axes[0, 1].hist(data['comment'].dropna(), bins=30, color='green', edgecolor='black')
axes[0, 1].set_title('Distribution of Comments')

# Histogram for Shares
axes[1, 0].hist(data['share'].dropna(), bins=30, color='red', edgecolor='black')
axes[1, 0].set_title('Distribution of Shares')

# Histogram for Total Interactions
axes[1, 1].hist(data['Total Interactions'].dropna(), bins=30, color='purple', edgecolor='black')
axes[1, 1].set_title('Distribution of Total Interactions')

plt.tight_layout()
plt.show()

# Correlation Heatmap
corr = data_encoded.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Define target and features
X = data_encoded.drop(['Unnamed: 0', 'Total Interactions'], axis=1)
y = data_encoded['Total Interactions']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
random_forest = RandomForestRegressor(random_state=42)
random_forest.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred_forest = random_forest.predict(X_test)
mae_forest = mean_absolute_error(y_test, y_pred_forest)
mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)

print(f"Random Forest Model Evaluation:")
print(f"MAE: {mae_forest}")
print(f"MSE: {mse_forest}")
print(f"R-Squared: {r2_forest}")

# Feature Importance
feature_importances = random_forest.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest Model')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Save feature importance to CSV for reporting
feature_importance_df.to_csv('feature_importance.csv', index=False)
