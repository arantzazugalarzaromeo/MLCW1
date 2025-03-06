import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib  

# Loading train dataset
train = pd.read_csv('CW1_train.csv')

sns.set_style("whitegrid")

# EDA:
# Head and Info of the Dataset:
print(f"Train Dataset Head:\n{train.head()}")
print(f"Train Dataset Info:\n{train.info()} ")

# Identify categorical and numerical columns
categorical_cols = train.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Remove target variable from numerical columns list 
if "outcome" in numerical_cols:
    numerical_cols.remove("outcome")

print("\n Categorical Columns:", categorical_cols)
print("\n Numerical Columns:", numerical_cols)

# Summary statistics for the numerical variables:
print("\n Summary Statistics:")
print(train.describe())

# Checking for missing values:
print("\n Missing Values (Train):")
print(train.isnull().sum())

# EDA Visualization:

# Visualizing the Target Variable
plt.figure(figsize=(8, 5))
sns.histplot(train["outcome"], bins=50, kde=True, color="red")
plt.xlabel("Outcome")
plt.ylabel("Frequency")
plt.title("Distribution of Outcome Variable")
plt.show()

# Histograms for Numerical Features
train[numerical_cols].hist(figsize=(15, 10), bins=30, edgecolor="green")
plt.suptitle("Histograms of Numerical Features")
plt.show()

# Boxplots for Categorical Features
fig, axes = plt.subplots(1, len(categorical_cols), figsize=(18, 5))

for i, var in enumerate(categorical_cols):
    sns.boxplot(x=train[var], y=train["outcome"], ax=axes[i])
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
    axes[i].set_title(f"Outcome vs {var}")

plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# Split dataset into 80% train and 20% validation
train_set, val_set = train_test_split(train, test_size=0.2, random_state=123)

# Encoding the categorical data into numerical data:
one_hot_encoder = OneHotEncoder(drop="first", sparse_output=False)  # Using `sparse_output` instead of `sparse`

encoded_train = one_hot_encoder.fit_transform(train_set[categorical_cols])
encoded_val = one_hot_encoder.transform(val_set[categorical_cols])

# Convert encoded data into DataFrames
encoded_train_df = pd.DataFrame(encoded_train, columns=one_hot_encoder.get_feature_names_out(categorical_cols))
encoded_val_df = pd.DataFrame(encoded_val, columns=one_hot_encoder.get_feature_names_out(categorical_cols))

train_encoded = train_set.drop(columns=categorical_cols).join(encoded_train_df)
val_encoded = val_set.drop(columns=categorical_cols).join(encoded_val_df)

# Save training and validation datasets
train_encoded.to_csv("CW1_train_encoded.csv", index=False)
val_encoded.to_csv("CW1_val_encoded.csv", index=False)

print(" Trainin saved CW1_train_encoded.csv")
print(" Validation saved CW1_val_encoded.csv")



# test dataset
test = pd.read_csv("CW1_test.csv")

one_hot_encoder = joblib.load("one_hot_encoder.pkl")

# Identify categorical columns
categorical_cols = test.select_dtypes(include=["object"]).columns.tolist()

encoded_test = one_hot_encoder.transform(test[categorical_cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=one_hot_encoder.get_feature_names_out(categorical_cols))

test_encoded = test.drop(columns=categorical_cols).join(encoded_test_df)

train_encoded = pd.read_csv("CW1_train_encoded.csv")
expected_features = [col for col in train_encoded.columns if col != "outcome"]

missing_cols = set(expected_features) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0  

test_encoded = test_encoded[expected_features]  

# Save the processed test dataset
test_encoded.to_csv("CW1_test_encoded.csv", index=False)
print(" Test set successfully saved as CW1_test_encoded.csv")