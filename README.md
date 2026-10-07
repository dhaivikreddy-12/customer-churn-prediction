# 📉 Customer Churn Prediction

> *"Churn isn't just lost revenue — it's lost customers. Predicting who's about to leave lets a business act before it's too late."*

An intermediate classification project that predicts which customers are likely to cancel their subscription. It goes beyond accuracy into precision/recall and shows how class imbalance changes the game. This is the kind of problem every subscription-based company faces.

## What this project does

- Loads a realistic telco-style customer dataset (~800 customers).
- Does EDA to spot patterns between churned and loyal customers.
- Engineers a few extra features (tenure, usage, support tickets).
- Trains **Logistic Regression**, **Random Forest**, and **XGBoost**, then compares them.
- Evaluates with accuracy, precision/recall, and AUC — not just accuracy.
- Ships a simple predictor function for new customers.

## The dataset

Synthetic but realistic (`data/customers.csv`), with these key fields:

| Feature             | Description                        |
|---------------------|------------------------------------|
| `tenure_months`     | How long the customer has stayed   |
| `monthly_charges`   | What they pay each month           |
| `contract_type`     | Monthly / one year / two year      |
| `total_usage_gb`    | Data usage                         |
| `support_tickets`   | Number of tickets filed            |
| `churn`             | Target (1 = left, 0 = stayed)      |

## How to run it

```bash
pip install -r requirements.txt

# Generate data + full pipeline + model comparison
python churn.py

# Explore the data
python explore.py
```

## Project structure

```
customer-churn-prediction/
├── data/
│   └── customers.csv            # generated dataset
├── plots/
├── src/
│   ├── preprocess.py            # cleaning & feature engineering
│   ├── train_model.py           # trains & compares 3 models
│   └── predict.py               # single-customer prediction
├── churn.py                     # main script
├── requirements.txt
└── README.md
```

## What I learned

- That accuracy can be misleading with imbalanced classes — why precision/recall/AUC matter.
- How to compare three very different models fairly.
- Feature engineering beats adding more data sometimes.
- Realistic problems are messy — this project taught me to embrace that.

## Results

On the test set, the best model (usually **Random Forest or XGBoost**) reaches **~0.79 AUC** with balanced precision/recall. Logistic regression is close behind and far more interpretable — a real trade-off.

---

*Built with Python, pandas, scikit-learn, XGBoost, matplotlib. Made for learning, by a student, for students.*
