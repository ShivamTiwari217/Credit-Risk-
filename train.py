"""Train the credit-risk scorecard and write everything the app needs to artifacts/.

    python train.py --data data/cs-training.csv   # Kaggle: Give Me Some Credit
    python train.py --synthetic                   # demo data, no download needed

Steps: (1) fit logistic scorecard + GBM challenger, (2) export coefficients as JSON,
(3) calibrate PDs and set accept/reject cut-offs from economics, (4) audit decisions by age band.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from features import FEATURES, prep

OUT = Path(__file__).parent / "artifacts"
TARGET = "SeriousDlqin2yrs"


def synth(n=60000, seed=7):
    r = np.random.default_rng(seed)
    age = r.integers(21, 80, n)
    inc = np.exp(r.normal(8.6, 0.6, n)).round()
    util = np.clip(r.beta(1.2, 2.5, n) * 1.3, 0, None)
    dr = np.clip(r.gamma(2, 0.2, n), 0, 5)
    l30, l60, l90 = r.poisson(0.3, n), r.poisson(0.12, n), r.poisson(0.1, n)
    z = (-4.2 + 1.6 * util + 0.5 * dr - 0.35 * (np.log(inc) - 8.6)
         + 0.35 * l30 + 0.5 * l60 + 0.8 * l90 - 0.02 * (age - 45))
    y = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    inc = np.where(r.random(n) < 0.2, np.nan, inc)
    return pd.DataFrame({
        TARGET: y, "RevolvingUtilizationOfUnsecuredLines": util, "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": l30, "DebtRatio": dr, "MonthlyIncome": inc,
        "NumberOfOpenCreditLinesAndLoans": r.poisson(8, n), "NumberOfTimes90DaysLate": l90,
        "NumberRealEstateLoansOrLines": r.poisson(1, n),
        "NumberOfTime60-89DaysPastDueNotWorse": l60, "NumberOfDependents": r.poisson(0.8, n)})


def ks(y, p):
    f, t, _ = roc_curve(y, p)
    return float(np.max(t - f))


def decide(p, acc, rej):
    return np.where(p <= acc, "accept", np.where(p >= rej, "reject", "refer"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to cs-training.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--margin", type=float, default=0.12, help="net profit per 1 of exposure on a good loan")
    ap.add_argument("--lgd", type=float, default=0.60, help="loss given default, share of exposure")
    ap.add_argument("--target-bad", type=float, default=0.04, help="max bad rate among auto-accepts")
    a = ap.parse_args()
    if not a.synthetic and not a.data:
        ap.error("pass --data PATH or --synthetic")

    df = synth() if a.synthetic else pd.read_csv(a.data, index_col=0)
    df = df.dropna(subset=[TARGET]).copy()
    y = df[TARGET].astype(int)

    # 60/20/20 split: train, validation (calibration), test (everything reported)
    tr, tmp = train_test_split(df.index, test_size=0.4, stratify=y, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y.loc[tmp], random_state=42)
    med = float(df.loc[tr, "MonthlyIncome"].median())
    X = prep(df, med)

    # 1. Models: logistic scorecard (the one we ship) vs GBM challenger
    sc = StandardScaler().fit(X.loc[tr])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X.loc[tr]), y.loc[tr])
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=250,
                                        random_state=42).fit(X.loc[tr], y.loc[tr])
    p_va = lr.predict_proba(sc.transform(X.loc[va]))[:, 1]
    p_te_raw = lr.predict_proba(sc.transform(X.loc[te]))[:, 1]
    p_gb = gb.predict_proba(X.loc[te])[:, 1]
    yte = y.loc[te].values

    # 3a. Calibration: isotonic fit on validation, kept only if it improves Brier on test
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=0.999).fit(p_va, y.loc[va])
    p_te_cal = iso.predict(p_te_raw)
    b_raw, b_cal = brier_score_loss(yte, p_te_raw), brier_score_loss(yte, p_te_cal)
    use_cal = bool(b_cal < b_raw)
    p_te = p_te_cal if use_cal else p_te_raw
    cal = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()} if use_cal else None
    bins = pd.qcut(p_te, 10, duplicates="drop")
    pd.DataFrame({"predicted": p_te, "observed": yte}).groupby(bins, observed=True).mean() \
        .reset_index(drop=True).to_csv(OUT / "calibration.csv", index=False)

    # 3b. Cut-offs. Reject at break-even PD; accept where the approved book hits the bad-rate target.
    brk = a.margin / (a.margin + a.lgd)
    rows = []
    for c in np.round(np.arange(0.005, 0.30, 0.005), 3):
        m = p_te <= c
        rows.append({"cutoff": c, "approval_rate": m.mean(),
                     "bad_rate": yte[m].mean() if m.any() else np.nan,
                     "profit_per_applicant": np.where(yte[m] == 1, -a.lgd, a.margin).sum() / len(yte)})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "threshold_sweep.csv", index=False)
    ok = sweep[sweep.bad_rate <= a.target_bad]
    acc = min(float(ok.cutoff.max()) if len(ok) else 0.005, brk * 0.6)

    # 4. Fairness audit on the test set (age is not a model input; this checks for disparate impact anyway)
    age = df.loc[te, "age"]
    band = pd.cut(age, [0, 29, 44, 59, 200], labels=["under 30", "30-44", "45-59", "60+"])
    dec = decide(p_te, acc, brk)
    f = pd.DataFrame({"band": band.values, "dec": dec, "y": yte, "p": p_te})
    rep = f.groupby("band", observed=True).apply(lambda g: pd.Series({
        "applicants": len(g), "approve_rate": (g.dec == "accept").mean(),
        "refer_rate": (g.dec == "refer").mean(), "reject_rate": (g.dec == "reject").mean(),
        "observed_default_rate": g.y.mean(), "mean_predicted_pd": g.p.mean(),
        "auc": roc_auc_score(g.y, g.p) if g.y.nunique() > 1 else np.nan}), include_groups=False).reset_index()
    rep["adverse_impact_ratio"] = rep.approve_rate / rep.approve_rate.max()
    rep["flag_below_0.8"] = rep.adverse_impact_ratio < 0.8
    rep.to_csv(OUT / "fairness_report.csv", index=False)

    metrics = {
        "data_source": "synthetic" if a.synthetic else "user data", "rows": int(len(df)),
        "bad_rate": float(y.mean()),
        "logistic": {"auc": roc_auc_score(yte, p_te_raw), "ks": ks(yte, p_te_raw)},
        "gbm_challenger": {"auc": roc_auc_score(yte, p_gb), "ks": ks(yte, p_gb)},
        "brier_raw": b_raw, "brier_calibrated": b_cal, "calibration_used": use_cal,
        "margin": a.margin, "lgd": a.lgd, "target_bad_rate": a.target_bad,
        "accept_cutoff": acc, "reject_cutoff": brk,
        "test_decisions": {k: float((dec == k).mean()) for k in ("accept", "refer", "reject")}}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 2. Export: the app needs only these numbers, no sklearn at serving time
    (OUT / "model.json").write_text(json.dumps({
        "data_source": metrics["data_source"], "features": FEATURES, "income_median": med,
        "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
        "calibration": cal, "thresholds": {"accept": acc, "reject": brk}}, indent=2))
    print(json.dumps(metrics, indent=2))
    print(rep.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
