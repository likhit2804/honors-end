import pandas as pd
import numpy as np

# Define the input and output filenames
input_file = 'data/omics/mirna.csv'
output_file = 'data/omics/mirna_cleaned_for_model.csv'

try:
    print(f"Loading '{input_file}'...")
    # Load the data.
    # Features (miRNA IDs) are in the first column (index_col=0)
    # Samples (TCGA IDs) are in the first row (header=0)
    df = pd.read_csv(input_file, index_col=0, header=0)
    
    print(f"Original shape (features, samples): {df.shape}")

    # 1. Transpose the data
    # We want samples as rows and features as columns for modeling.
    df_transposed = df.T
    print(f"Transposed shape (samples, features): {df_transposed.shape}")

    # 2. Clean Feature (Column) Names
    # The feature names (now column names) might be numbers or contain 'NaN'
    
    # 2a. Rename the 'NaN' column if it exists (from the file-read)
    if np.nan in df_transposed.columns:
        print("Found 'NaN' column name. Renaming to 'miRNA_Unknown'.")
        df_transposed = df_transposed.rename(columns={np.nan: 'miRNA_Unknown'})
        
    # 2b. Ensure all column names are strings and add a prefix
    # This prevents issues with models that can't handle numeric column names.
    df_transposed.columns = [f"miRNA_{col}" for col in df_transposed.columns]
    
    # 3. Handle Missing Values (Imputation)
    print("Handling missing values (NaNs) in expression data...")
    missing_before = df_transposed.isnull().sum().sum()
    print(f"Total missing values before imputation: {missing_before}")

    # All columns are numeric expression values.
    # We impute missing values with the median of each feature (column).
    for col in df_transposed.columns:
        median_val = df_transposed[col].median()
        df_transposed[col] = df_transposed[col].fillna(median_val)
        
    missing_after = df_transposed.isnull().sum().sum()
    print(f"Total missing values after imputation: {missing_after}")

    # 4. Final Check and Save
    df_transposed.to_csv(output_file)
    print(f"\n*** Successfully cleaned and saved data to '{output_file}' ***")
    
    print("\n--- Head of Cleaned Data ---")
    print(df_transposed.head())

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
    print("Please make sure it's in the same directory as this script.")
except Exception as e:
    print(f"An error occurred: {e}")