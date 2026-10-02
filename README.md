# Loan underwriting desk (Streamlit)

Logistic scorecard + GBM challenger, calibrated PDs, economics-based cut-offs, reason codes, and an age-band fairness audit.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-train.txt
# Real data: download cs-training.csv from Kaggle "Give Me Some Credit" into data/
python train.py --data data/cs-training.csv        # or: python train.py --synthetic
streamlit run app.py
```
Useful flags: `--margin 0.12 --lgd 0.60 --target-bad 0.04` set the economics behind the cut-offs.

## Deploy on Streamlit Community Cloud
1. Retrain on real data first. The bundled `artifacts/` come from synthetic data and the app says so.
2. Push to GitHub. Commit `artifacts/`, `app.py`, `features.py`, `train.py`, `requirements.txt`. Do not commit `data/`.
3. Go to share.streamlit.io, click **Create app**, pick the repo, branch `main`, main file `app.py`.
4. Under **Advanced settings** choose Python 3.11 or 3.12, then **Deploy**. Every push to `main` redeploys.
5. Use a private repo and restrict viewers in the app's sharing settings. It's a credit tool.

The app needs only numpy, pandas and streamlit, because the model is exported as plain JSON.
