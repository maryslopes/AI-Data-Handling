import pandas as pd

def ingest_data():
    # Load dataset from Kaggle (local CSV or GCS)
    raw_path = "data/raw/Equipment-Failure.csv"
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

if __name__ == "__main__":
    ingest_data()
