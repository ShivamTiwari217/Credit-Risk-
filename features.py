"""Feature engineering shared by train.py and app.py, so training and scoring can't drift apart."""
import numpy as np
import pandas as pd

# Dependents and age are deliberately NOT model inputs (family status / age are fairness-sensitive).
FEATURES = ["utilization", "debt_ratio", "log_income", "income_missing",
            "late_30_59", "late_60_89", "late_90", "open_lines", "real_estate"]

LABELS = {
    "utilization": "Credit utilisation", "debt_ratio": "Debt payments vs income",
    "log_income": "Monthly income", "income_missing": "Income not provided",
    "late_30_59": "Payments 30-59 days late", "late_60_89": "Payments 60-89 days late",
    "late_90": "Payments 90+ days late", "open_lines": "Open credit lines",
    "real_estate": "Real-estate loans",
}

# Plain-language reasons used in adverse-action notices (only shown when a factor RAISES risk).
REASONS = {
    "utilization": "Balances are high relative to credit limits.",
    "debt_ratio": "Debt payments take a large share of income.",
    "log_income": "Income is low relative to the risk profile.",
    "income_missing": "Income could not be verified.",
    "late_30_59": "Recent payments were 30-59 days late.",
    "late_60_89": "Recent payments were 60-89 days late.",
    "late_90": "Recent payments were 90 or more days late.",
    "open_lines": "Number of open credit lines.",
    "real_estate": "Number of real-estate loans.",
}


def prep(df: pd.DataFrame, income_median: float) -> pd.DataFrame:
    """Raw 'Give Me Some Credit' columns -> model features."""
    X = pd.DataFrame(index=df.index)
    inc = df["MonthlyIncome"]
    X["utilization"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1.5)
    X["debt_ratio"] = np.log1p(df["DebtRatio"].clip(0, 5))
    X["income_missing"] = inc.isna().astype(int)
    X["log_income"] = np.log1p(inc.fillna(income_median).clip(upper=1e6))
    # 96/98 are sentinel codes in the Kaggle data; clipping handles them.
    X["late_30_59"] = df["NumberOfTime30-59DaysPastDueNotWorse"].clip(0, 6)
    X["late_60_89"] = df["NumberOfTime60-89DaysPastDueNotWorse"].clip(0, 6)
    X["late_90"] = df["NumberOfTimes90DaysLate"].clip(0, 6)
    X["open_lines"] = df["NumberOfOpenCreditLinesAndLoans"].clip(0, 25)
    X["real_estate"] = df["NumberRealEstateLoansOrLines"].clip(0, 6)
    return X[FEATURES].astype(float)
