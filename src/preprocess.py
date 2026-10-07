"""Generate synthetic customer churn data."""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)

n = 800
tenure = rng.integers(1, 60, n)
contract = rng.choice(["monthly", "one_year", "two_year"], n, p=[0.55, 0.3, 0.15])
monthly_charges = np.clip(rng.normal(65, 25, n), 20, 120).round(2)
usage = np.clip(rng.normal(30, 15, n), 0, 100).round(1)
tickets = rng.poisson(1.5, n)

churn_prob = (
    0.20
    - 0.010 * tenure
    + 0.006 * monthly_charges
    - 0.22 * (contract == "two_year")
    - 0.10 * (contract == "one_year")
    + 0.09 * tickets
    - 0.004 * usage
    + rng.normal(0, 0.10, n)
)
churn_prob = np.clip(churn_prob, 0.02, 0.98)
churn = (rng.random(n) < churn_prob).astype(int)

df = pd.DataFrame({
    "tenure_months": tenure,
    "monthly_charges": monthly_charges,
    "contract_type": contract,
    "total_usage_gb": usage,
    "support_tickets": tickets,
    "churn": churn,
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/customers.csv", index=False)
print(f"Generated {len(df)} customers. Churn rate: {churn.mean():.1%}")
