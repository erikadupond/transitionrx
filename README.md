---
title: TransitionRx Readmission Decision Support
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: streamlit
#sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# TransitionRx — Medication Complexity & Care-Transition Readmission Risk

Decision-support tool for hospital pharmacy and care-management leadership.

**Project question:** For a hospital facing CMS Hospital Readmissions Reduction Program (HRRP)
penalties, can discharge-level data already on hand — medication count, regimen churn, and
care-transition quality — identify which diabetes patients are at elevated risk of a preventable
30-day readmission, so limited pharmacist time is targeted where it matters?

## What's in the app

| Tab | Contents |
|---|---|
| 📖 **Start here** | Business problem, the decision supported, how to use the tool |
| 📊 **Risk dashboard** | Cohort risk distribution, capacity-vs-captured-risk curve, ranked worklist |
| 👤 **Patient brief** | Single-patient scoring with SHAP driver explanation |
| 💬 **Guidance Q&A** | RAG over CMS/AHRQ discharge-planning guidance, with citations |
| 💰 **Cost / value** | Scenario table, net-value sensitivity, resource impact |
| ⚠️ **Limitations** | Data gaps, performance ceiling, fairness caution, appropriate use |

Every headline figure is driven by sidebar assumptions the user controls — pharmacist capacity,
cost per readmission, cost per intervention, and assumed intervention effectiveness.

## Configuration

Set these under **Settings → Variables and secrets**:

| Name | Type | Required | Purpose |
|---|---|---|---|
| `HF_MODEL_REPO` | Variable | If artifacts aren't committed | Model repo, e.g. `username/transitionrx-readmission-xgboost` |
| `OPENAI_API_KEY` | Secret | No | Enables generated RAG answers. Without it the Q&A tab runs in retrieval-only mode |

## Files this Space expects

Committed directly, or downloaded from `HF_MODEL_REPO`:

- `transitionrx_xgb_model.pkl` — tuned XGBoost classifier
- `transitionrx_calibrated_model.pkl` — isotonic-calibrated version
- `transitionrx_metadata.json` — metrics, threshold, assumptions
- `transitionrx_feature_columns.csv` — required feature order
- `sample_cohort.csv` — demo cohort (optional; synthetic data used if absent)
- `transitionrx_faiss.index` + `transitionrx_chunks.parquet` — RAG index (optional)

## Model

Tuned XGBoost on the [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
(1999–2008). Cohort excludes expired/hospice discharges and keeps the first encounter per
patient. Target is `readmitted == '<30'`.

ROC-AUC is in the mid-0.60s — the realistic ceiling for administrative billing data and
consistent with published work on this dataset. **This ranks patients for a workflow decision.
It is not a clinical diagnostic.**

## Important limitations

- Training data is 1999–2008; practice patterns have changed
- No vitals, lab trends, social determinants, adherence, or clinical notes
- **Subgroup fairness across race, age, sex, and payer has not been audited**
- The model predicts readmission *risk*, not *preventability* or *responsiveness to intervention*
- The intervention-effectiveness assumption comes from literature, not measurement here

Educational course project. Not a medical device.
