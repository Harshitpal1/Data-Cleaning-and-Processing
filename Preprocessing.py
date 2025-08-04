# TASK 1
# DATA CLEANING AND PREPROCESSING

# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

print("--- Task 1: Data Cleaning & Preprocessing ---")

# 2. Load the Dataset
try:
    # The task suggests using the Titanic dataset.
    df = pd.read_csv('titanic.csv')
    print("\nDataset loaded successfully.")
except FileNotFoundError:
    print("\nError: 'titanic.csv' not found. Please download the dataset.")
    exit()

# STEP 1: EXPLORE BASIC INFO ---
print("\n--- Step 1: Initial Data Exploration ---")

# Display the first 5 rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Get a concise summary of the dataframe
print("\nDataset Info:")
df.info()

# Get the count of null values in each column
print("\nInitial Missing Values Count:")
print(df.isnull().sum())

# STEP 2: HANDLE MISSING VALUES ---
print("\n--- Step 2: Handling Missing Values ---")

# Fill 'Age' with the median value
df['Age'].fillna(df['Age'].median(), inplace=True)
print("\nMissing 'Age' values filled with median.")

# Fill 'Embarked' with the mode (most frequent value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
print("Missing 'Embarked' values filled with mode.")

# Drop the 'Cabin' column due to a high percentage of missing values
df.drop('Cabin', axis=1, inplace=True)
print("'Cabin' column dropped.")

# Verify that there are no more missing values
print("\nMissing Values Count After Handling:")
print(df.isnull().sum())


# STEP 3: ENCODE CATEGORICAL FEATURES ---
print("\n--- Step 3: Encoding Categorical Features ---")

# Convert 'Sex' and 'Embarked' into numerical format using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print("\n'Sex' and 'Embarked' columns converted to numerical using one-hot encoding.")

# Drop columns that are not useful for prediction
df.drop(['Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
print("Dropped non-essential columns: 'Name', 'PassengerId', 'Ticket'.")

print("\nDataset after encoding:")
print(df.head())


# STEP 4: NORMALIZE/STANDARDIZE NUMERICAL FEATURES ---
print("\n--- Step 4: Normalizing Numerical Features ---")

# Select numerical features for scaling
numerical_features = ['Age', 'Fare', 'Parch', 'SibSp']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the numerical features
df[numerical_features] = scaler.fit_transform(df[numerical_features])
print("\nNumerical features ('Age', 'Fare', 'Parch', 'SibSp') have been standardized.")

print("\nDataset after normalization:")
print(df.head())


# STEP 5: VISUALIZE AND REMOVE OUTLIERS ---
print("\n--- Step 5: Visualizing and Removing Outliers ---")

# Visualize outliers using boxplots before removal
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_features])
plt.title('Boxplots of Numerical Features Before Outlier Removal')
plt.ylabel('Values')
plt.savefig('outliers_before.png') # Saving the plot as an image
print("\nGenerated 'outliers_before.png' to show data distribution.")

# Remove outliers using the IQR method for 'Fare' and 'Age'
initial_rows = df.shape[0]
for feature in ['Age', 'Fare']:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR # Corrected the upper bound calculation

    # Filter out the outliers
    df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

print(f"\nRemoved {initial_rows - df.shape[0]} outlier rows based on 'Age' and 'Fare' IQR.")
print(f"Shape of the final preprocessed dataframe: {df.shape}")

# Visualize again after removal
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title('Boxplots of Age and Fare After Outlier Removal')
plt.ylabel('Standardized Values')
plt.savefig('outliers_after.png') # Saving the plot as an image
print("Generated 'outliers_after.png' to show data distribution after cleaning.")

print("\n--- Preprocessing Complete ---")
print("Final preprocessed data is ready.")
print(df.head())
