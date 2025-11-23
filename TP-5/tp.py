import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# Part 1: Load & Inspect Data
# ==========================

# Load dataset
df = pd.read_csv(r"sample dataset/traffic_sample.csv")

# Display first row
print("First row:")
print(df.head(1))

# Display dataset dimensions
print("Dataset dimensions:")
print(df.shape)


print("\n","-" * 40, "Part 2: Missing Values", "-" * 40)

# ==========================
# Part 2: Handling Missing Values
# ==========================

# Handling missing values
missing_overall = df.isnull().sum().sum()
print("\nTotal missing values in dataset:", missing_overall)

# Missing per column
missing_per_column = df.isnull().sum()
print("\nMissing values per column:")
print(missing_per_column)

# Decision rules
threshold = len(df) * 0.5  # 50%

for column in df.columns:
    if missing_per_column[column] > threshold:
        df = df.drop(columns=[column])
        print(f"\nColumn '{column}' deleted (too many missing values).")
    else:
        # Numeric → mean
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())
            print(f"\nColumn '{column}' imputed with mean.")
        # Categorical → mode
        else:
            df[column] = df[column].fillna(df[column].mode()[0])
            print(f"\nColumn '{column}' imputed with mode.")

# Final check
print("\nMissing values after cleaning:")
print(df.isnull().sum())


print("\n","-" * 40, "Part 2.1: Outlier Detection (IQR)", "-" * 40)

# ==========================
# Part 2.1: Outlier Detection using IQR
# ==========================

# Choose numeric columns only
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

outliers = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    outliers[col] = df[col][outlier_mask]

    print(f"\nColumn: {col}")
    print(f"IQR: {IQR}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Upper Bound: {upper_bound}")
    print(f"Outliers found: {outlier_mask.sum()}")

    # Plot boxplot
    plt.figure(figsize=(5, 3))
    plt.boxplot(df[col].dropna(), vert=False)
    plt.title(f"Boxplot of {col} (Outlier Detection - IQR)")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# Outlier Summary
print("\n===== Outliers Summary =====")
for col, values in outliers.items():
    print(f"{col}: {len(values)} outliers detected")
