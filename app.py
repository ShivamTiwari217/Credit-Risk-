import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from features import FEATURES, LABELS, REASONS, prep

A = Path(__file__).parent / "artifacts"
st.set_page_config(page_title="Loan underwriting desk", page_icon="🏦", layout="wide")

if not (A / "model.json").exists():
    st.error("No model found. Run `python train.py --synthetic` (or `--data path/to/cs-training.csv`) first.")
    st.stop()


@st.cache_resource
def load():
    return json.loads((A / "model.json").read_text()), json.loads((A / "metrics.json").read_text())


M, MET = load()


def score(row):
    X = prep(pd.DataFrame([row]), M["income_median"]).iloc[0].values.astype(float)
    contrib = (X - np.array(M["mean"])) / np.array(M["scale"]) * np.array(M["coef"])
    p = 1 / (1 + np.exp(-(M["intercept"] + contrib.sum())))
    if M["calibration"]:
        p = float(np.interp(p, M["calibration"]["x"], M["calibration"]["y"]))
    return float(p), contrib, X


st.title("Loan underwriting desk")
if M["data_source"] == "synthetic":
    st.warning("Demo model trained on synthetic data. Retrain on real data before trusting any number here.")

with st.sidebar:
    st.header("Application")
    age = st.number_input("Age", 18, 90, 35)
    income = st.number_input("Monthly income (0 if not provided)", 0, 1_000_000, 5000, step=250)
    debt = st.number_input("Monthly debt payments ÷ income", 0.0, 5.0, 0.35, 0.05)
    util = st.slider("Credit utilisation (balance ÷ limit)", 0.0, 1.5, 0.30, 0.05)
    l30 = st.number_input("Times 30-59 days late (last 2 yrs)", 0, 10, 0)
    l60 = st.number_input("Times 60-89 days late (last 2 yrs)", 0, 10, 0)
    l90 = st.number_input("Times 90+ days late", 0, 10, 0)
    lines = st.number_input("Open credit lines and loans", 0, 40, 6)
    re = st.number_input("Real-estate loans", 0, 10, 1)

row = {"RevolvingUtilizationOfUnsecuredLines": util, "age": age, "DebtRatio": debt,
       "MonthlyIncome": income if income > 0 else np.nan, "NumberOfTime30-59DaysPastDueNotWorse": l30,
       "NumberOfTime60-89DaysPastDueNotWorse": l60, "NumberOfTimes90DaysLate": l90,
       "NumberOfOpenCreditLinesAndLoans": lines, "NumberRealEstateLoansOrLines": re}
p, contrib, X = score(row)
acc, rej = M["thresholds"]["accept"], M["thresholds"]["reject"]

notes = []
if age < 21 or age > 70:
    decision = "Reject"; notes.append("Policy: applicant age is outside the 21-70 eligibility range.")
elif p >= rej:
    decision = "Reject"; notes.append(f"Default risk {p:.1%} is at or above the {rej:.1%} break-even cut-off.")
elif p <= acc and income > 0:
    decision = "Accept"; notes.append(f"Default risk {p:.1%} is within the {acc:.1%} auto-accept limit.")
else:
    decision = "Refer"
    notes.append("Income was not provided, so a credit officer must verify it." if income == 0 and p <= acc
                 else f"Default risk {p:.1%} sits between the accept ({acc:.1%}) and reject ({rej:.1%}) cut-offs.")

t1, t2, t3 = st.tabs(["Decision", "Model card", "Fairness"])
with t1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Decision", decision)
    c2.metric("Probability of default", f"{p:.1%}")
    c3.metric("Auto-accept / reject at", f"{acc:.1%} / {rej:.1%}")
    {"Accept": st.success, "Refer": st.warning, "Reject": st.error}[decision](" ".join(notes))
    st.subheader("What drove the score")
    st.caption("Impact on log-odds of default versus the average applicant. Right raises risk, left lowers it.")
    s = pd.Series(contrib, index=[LABELS[f] for f in FEATURES]).sort_values()
    st.bar_chart(s, horizontal=True)
    tbl = pd.DataFrame({"Factor": [LABELS[f] for f in FEATURES], "Value": X.round(2),
                        "Odds multiplier": np.exp(contrib).round(2)}).sort_values("Odds multiplier", ascending=False)
    st.dataframe(tbl, hide_index=True, width="stretch")
    if decision != "Accept":
        st.subheader("Adverse-action reasons")
        top = [f for f, c in sorted(zip(FEATURES, contrib), key=lambda t: -t[1]) if c > 0.02][:4]
        for f in top:
            st.write("• " + REASONS[f])
        if not top:
            st.write("• No single factor stands out. The decision rests on overall risk level.")

with t2:
    st.subheader("Performance on held-out test set")
    st.write(f"Data: **{MET['data_source']}**, {MET['rows']:,} rows, bad rate {MET['bad_rate']:.1%}")
    st.dataframe(pd.DataFrame({"Logistic scorecard (shipped)": MET["logistic"],
                               "GBM challenger": MET["gbm_challenger"]}).round(3), width="stretch")
    st.write(f"Brier score: raw {MET['brier_raw']:.4f}, calibrated {MET['brier_calibrated']:.4f} "
             f"({'calibration applied' if MET['calibration_used'] else 'raw probabilities kept'}).")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Calibration: predicted vs observed default rate by decile")
        st.line_chart(pd.read_csv(A / "calibration.csv"), x="predicted", y="observed")
    with c2:
        st.caption("Approval rate vs bad rate as the cut-off moves")
        sw = pd.read_csv(A / "threshold_sweep.csv")
        st.line_chart(sw, x="cutoff", y=["approval_rate", "bad_rate"])
    st.write(f"Cut-offs come from economics: margin {MET['margin']:.0%}, loss given default {MET['lgd']:.0%} "
             f"(reject above break-even PD), and a {MET['target_bad_rate']:.0%} bad-rate target for auto-accepts.")
    d = MET["test_decisions"]
    st.write(f"Test-set mix: {d['accept']:.0%} accept, {d['refer']:.0%} refer, {d['reject']:.0%} reject.")

with t3:
    st.subheader("Decision rates by age band (test set)")
    st.caption("Age is not a model input. This checks whether outcomes still differ. "
               "An adverse impact ratio under 0.8 (the four-fifths rule) is flagged for review.")
    fr = pd.read_csv(A / "fairness_report.csv")
    st.dataframe(fr.round(3), hide_index=True, width="stretch")
    if fr["flag_below_0.8"].any():
        st.warning("At least one band falls below 0.8. Investigate before deploying.")
    else:
        st.success("No age band falls below the 0.8 threshold.")
