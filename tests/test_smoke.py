import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSV = ROOT / "data" / "customers.csv"


def _require_csv():
    if not CSV.exists():
        pytest.skip(f"cached dataset missing ({CSV}); run the loader once to cache it")


def _frame():
    return pd.read_csv(CSV)


def test_readme_and_license_exist():
    assert (ROOT / "README.md").is_file(), "README.md is missing"
    assert (ROOT / "LICENSE").is_file(), "LICENSE is missing"


def test_load_module_imports_cleanly():
    from src import load_data

    assert callable(load_data.load)
    assert load_data.CSV_PATH.startswith("data/")
    assert pathlib.PurePosixPath(load_data.CSV_PATH).name == "customers.csv"


def test_train_model_module_imports_cleanly(monkeypatch):
    _require_csv()
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)
    monkeypatch.chdir(ROOT)

    import src.train_model as train_model

    assert train_model.TARGET == "Churn"
    assert train_model.ID_COLS == ["customerID"]
    assert len(train_model.df) == 7043


def test_cached_csv_shape_and_churn_rate():
    _require_csv()
    df = _frame()

    assert len(df) == 7043, f"expected the full Telco set, got {len(df)} rows"
    assert "Churn" in df.columns
    assert set(df["Churn"].dropna().unique()) == {"Yes", "No"}
    rate = (df["Churn"] == "Yes").mean()
    assert 0.20 < rate < 0.35, f"implausible churn rate {rate:.3f}"
    assert {"tenure", "MonthlyCharges", "TotalCharges", "Contract", "gender"} <= set(df.columns)
    # The cached frame ships without the identifier, which is why train_model
    # drops ID_COLS defensively rather than unconditionally.
    assert "customerID" not in df.columns


def test_totalcharges_blank_strings_coerce_to_nan():
    _require_csv()
    df = _frame()

    # Without errors="coerce" the 11 blank TotalCharges cells blow up the script.
    with pytest.raises(ValueError):
        pd.to_numeric(df["TotalCharges"])

    numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
    assert pd.api.types.is_numeric_dtype(numeric)
    coerced = int(numeric.isna().sum())
    assert coerced > 0, "expected the known blank TotalCharges cells to coerce to NaN"
    assert coerced == 11
    assert numeric.dropna().gt(0).all()
    assert numeric.max() < 10_000

    # Blank / whitespace-only / junk input must all become NaN, not raise.
    junk = pd.Series(["12.5", " ", "", "n/a", "  8.25 "], dtype="object")
    out = pd.to_numeric(junk, errors="coerce")
    assert out.isna().tolist() == [False, True, True, True, False]
    assert out.dropna().tolist() == [12.5, 8.25]


def test_preprocessing_turns_totalcharges_numeric_in_place():
    _require_csv()
    df = _frame()
    assert not pd.api.types.is_numeric_dtype(df["TotalCharges"]), "raw column should be text"

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == 11


def test_tiny_columntransformer_pipeline_predicts_probabilities():
    _require_csv()
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = _frame()
    df = df.drop(columns=[c for c in ["customerID"] if c in df.columns])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    sample = df.sample(n=300, random_state=0)
    y = (sample["Churn"] == "Yes").astype(int)
    X = sample.drop(columns=["Churn"])
    assert set(y.unique()) == {0, 1}, "slice lost a class"

    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    assert "TotalCharges" in numeric and "Contract" in categorical

    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    pipe = Pipeline([("pre", preprocess), ("clf", LogisticRegression(max_iter=2000))])

    pipe.fit(X, y)
    proba = pipe.predict_proba(X.head(25))

    assert proba.shape == (25, 2)
    assert np.isfinite(proba).all()
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert pipe.predict(X.head(25)).shape == (25,)
    assert list(pipe.named_steps) == ["pre", "clf"]
