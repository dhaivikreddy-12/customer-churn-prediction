"""Single-customer churn prediction."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/customers.csv")
df = pd.get_dummies(df, columns=["contract_type"], drop_first=True)
X = df.drop(columns=["churn"])
y = df["churn"]

scaler = StandardScaler().fit(X)
model = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)


def predict_churn(tenure, monthly_charges, contract, usage_gb, tickets):
    row = pd.DataFrame([{
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "contract_type_one_year": 1 if contract == "one_year" else 0,
        "contract_type_two_year": 1 if contract == "two_year" else 0,
        "total_usage_gb": usage_gb,
        "support_tickets": tickets,
    }])
    row = row[X.columns]
    prob = model.predict_proba(scaler.transform(row))[0][1]
    return prob


if __name__ == "__main__":
    p = predict_churn(tenure=12, monthly_charges=70, contract="monthly",
                      usage_gb=40, tickets=3)
    print(f"Churn probability: {p:.1%}")
    print("High risk" if p > 0.5 else "Low risk")
