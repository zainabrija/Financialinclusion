import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import streamlit as st

# Set the style
plt.style.use('seaborn-v0_8')  # Updated style name

# Sidebar for controls
st.sidebar.title("Controls")

# File uploader for manual data upload
st.sidebar.markdown(
    "You can also upload your own dataset. Make sure it matches the structure of the original FINDEXData.csv."
)
uploaded_file = st.sidebar.file_uploader(
    "Or upload your own FINDEXData.csv file",
    type=["csv"],
    help="Upload a CSV file with the same structure as FINDEXData.csv"
)
if uploaded_file is not None:
    data_path = "data/FINDEXData.csv"
    if not os.path.exists("data"):
        os.makedirs("data")
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("File uploaded and saved! The dashboard will now use your uploaded data.")

# Load the dataset
data_path = 'data/FINDEXData.csv'
df = pd.read_csv(data_path)

# Display basic info about the dataset
print("Dataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Let's see what indicators we have
print("\nUnique Indicators:")
print(df['Indicator Name'].unique())

# Create pivot table
pivoted_df = df.pivot_table(
    index=['Country Name', 'Country Code'],
    columns='Indicator Name',
    values='MRV'
).reset_index()

# Display the pivoted data
print("\nPivoted Data Structure:")
print(pivoted_df.head())

# Choose target indicator
target_indicator = 'Account, female (% age 15+)'  # Example indicator
if target_indicator not in pivoted_df.columns:
    print(f"\nError: {target_indicator} not found in the dataset")
    print("Available indicators:", pivoted_df.columns.tolist())
    exit()

# Prepare features and target
X = pivoted_df.drop(columns=['Country Name', 'Country Code', target_indicator])
y = pivoted_df[target_indicator]

# Handle missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Convert target to binary
y_binary = (y > y.median()).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create visualizations directory if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# 1. Distribution of Target Indicator
plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True, color='skyblue')
plt.title(f'Distribution of {target_indicator}', fontsize=14, pad=20)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('visualizations/distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', mask=mask)
plt.title('Correlation Heatmap of Financial Indicators', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('visualizations/correlation.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Feature Importance
plt.figure(figsize=(12, 8))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_[0]
}).sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance for Financial Inclusion Prediction', fontsize=14, pad=20)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Financial Indicator', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Top and Bottom Countries
plt.figure(figsize=(15, 10))

# Top 10 Countries
plt.subplot(2, 1, 1)
top_countries = pivoted_df.nlargest(10, target_indicator)
sns.barplot(x=target_indicator, y='Country Name', data=top_countries, palette='viridis')
plt.title('Top 10 Countries with Highest Financial Inclusion', fontsize=14, pad=20)
plt.xlabel('Financial Inclusion Score', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.grid(True, alpha=0.3)

# Bottom 10 Countries
plt.subplot(2, 1, 2)
bottom_countries = pivoted_df.nsmallest(10, target_indicator)
sns.barplot(x=target_indicator, y='Country Name', data=bottom_countries, palette='viridis')
plt.title('Bottom 10 Countries with Lowest Financial Inclusion', fontsize=14, pad=20)
plt.xlabel('Financial Inclusion Score', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/country_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix', fontsize=14, pad=20)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualizations have been saved in the 'visualizations' directory:")
print("1. distribution.png - Distribution of financial inclusion levels")
print("2. correlation.png - Correlation between different financial indicators")
print("3. feature_importance.png - Importance of each indicator in prediction")
print("4. country_comparison.png - Top and bottom countries comparison")
print("5. confusion_matrix.png - Model performance confusion matrix") 