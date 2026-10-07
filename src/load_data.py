"""Load the real IBM Telco Customer Churn dataset (7,043 customers).

Source: OpenML 'telco-customer-churn'. Cached to data/customers.csv so the
repo runs offline after the first run.
"""
import os
import pandas as pd

CSV_PATH = "data/customers.csv"


def load():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name="telco-customer-churn", version=1, as_frame=True)
    df = bunch.frame.copy()
    os.makedirs("data", exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} customers -> {CSV_PATH}")
    print(df.head())
