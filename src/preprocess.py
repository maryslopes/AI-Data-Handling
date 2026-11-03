import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    df = pd.read_csv("data/raw/Equipment-Failure.csv")

    # Example: rename columns for BigQuery compatibility
    df.columns = [c.replace(" ", "_").lower() for c in df.columns]

    # Encode categorical variable
    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = ["air_temperature_", "process_temperature_k", 
                    "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save processed file
    df.to_csv("data/processed/Equipment-Failure-clean.csv", index=False)
    print("Processed dataset saved.")

if __name__ == "__main__":
    preprocess_data()
