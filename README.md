# 📉 Customer Churn Prediction

> *"Churn isn't just lost revenue — it's a customer who already decided to leave. Predicting it early gives the business a chance to respond."*

An intermediate classification project on the **IBM Telco Customer Churn** dataset — 7,043 real subscribers, 20 features including contract type, tenure, and monthly charges. It's the same kind of problem every subscription company faces, and it made class imbalance click for me in a way no tutorial had.

## What this project does

- Loads the real Telco churn dataset (7,043 customers).
- Cleans a genuinely messy column — `TotalCharges` arrives as text with blanks.
- Builds a `ColumnTransformer` pipeline: median-impute + scale numerics, impute + one-hot categoricals.
- Trains Logistic Regression, Random Forest, and XGBoost, then compares them.
- Evaluates with accuracy, precision, recall, and AUC against a majority-class baseline.

## The dataset

[Telco Customer Churn](https://www.ibm.com/docs/en/cognos-analytics/11.1.0?topic=samples-telco-customer-churn) via OpenML — 7,043 customers.

| Feature | Example values |
|---|---|
| `tenure` | Months with the company |
| `MonthlyCharges` | Current monthly bill |
| `TotalCharges` | Cumulative bill (blank for new customers) |
| `Contract` | Month-to-month / One year / Two year |
| `PaymentMethod`, `InternetService`, `Support requests` | ...and 15 more |
| `Churn` | Yes / No — **target** (26.5% churn) |

## How to run it

```bash
pip install -r requirements.txt

python churn.py    # load, preprocess, train, compare all three models
```

## Project structure

```
customer-churn-prediction/
├── data/
│   └── customers.csv
├── src/
│   ├── load_data.py    # fetch + cache
│   ├── train_model.py  # pipeline, models, evaluation
│   └── predict.py
├── tests/
├── churn.py
├── requirements.txt
└── README.md
```

## What I learned

- That accuracy is nearly useless here: predicting "no churn" for everyone gives 73.5%.
- Why one-hot encoding and imputation belong inside a `Pipeline`, not bolted on manually.
- That logistic regression is genuinely competitive — and interpretable, which matters when you have to explain a decision to a customer.
- How messy real columns are. `TotalCharges` had to be coerced from text to numeric.

## Results

20% stratified test split, majority-class baseline = 0.735 accuracy:

| Model | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| **LogisticRegression** | **0.806** | **0.657** | **0.559** | **0.842** |
| RandomForest | 0.782 | 0.611 | 0.492 | 0.819 |
| XGBoost | 0.769 | 0.575 | 0.500 | 0.810 |

Logistic Regression takes the top spot on AUC (0.842) — comfortably above baseline. That surprised me after reading that tree models always win. They didn't here.

---

*Built with Python, pandas, scikit-learn, XGBoost, matplotlib. Real subscriber data, honestly measured.*
