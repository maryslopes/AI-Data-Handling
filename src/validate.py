import pandas as pd

def validate_data():
    df = pd.read_csv("data/raw/Equipment-Failure.csv")

    # Check duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")

    # Check missing values
    missing = df.isnull().sum()
    print("Missing values per column:\n", missing)

    # Check target variable integrity
    if set(df['machine_failure'].unique()) <= {0,1}:
        print("Target variable is valid (binary).")
    else:
        print("Target variable has unexpected values!")

if __name__ == "__main__":
    validate_data()
