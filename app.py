"""
TransitionRx — Medication Complexity & Care-Transition Readmission Risk
Hugging Face Space (Streamlit)

Decision-support tool: ranks diabetes discharges by 30-day readmission risk so limited
pharmacist / case-manager time can be targeted, and prices out that targeting decision.
"""

import os
import json
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "REPLACE_ME/transitionrx-readmission-xgboost")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

st.set_page_config(
    page_title="TransitionRx — Readmission Risk Decision Support",
    page_icon="🏥",
    layout="wide",
)

PRIMARY = "#2E5395"
ACCENT = "#C55A11"


# ----------------------------------------------------------------------------------
# Loaders (cached)
# ----------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_model():
    """Download model artifacts from the Hub, or fall back to local files."""
    import joblib

    files = {}
    names = [
        "transitionrx_xgb_model.pkl",
        "transitionrx_calibrated_model.pkl",
        "transitionrx_metadata.json",
        "transitionrx_feature_columns.csv",
    ]

    # Prefer local files (useful when artifacts are committed into the Space)
    if all(os.path.exists(n) for n in names):
        for n in names:
            files[n] = n
        source = "local files in Space"
    else:
        from huggingface_hub import hf_hub_download

        for n in names:
            files[n] = hf_hub_download(repo_id=HF_MODEL_REPO, filename=n)
        source = f"Hugging Face repo `{HF_MODEL_REPO}`"

    model = joblib.load(files["transitionrx_xgb_model.pkl"])
    calibrated = joblib.load(files["transitionrx_calibrated_model.pkl"])
    with open(files["transitionrx_metadata.json"]) as f:
        meta = json.load(f)
    cols = pd.read_csv(files["transitionrx_feature_columns.csv"])["feature"].tolist()
    return model, calibrated, meta, cols, source


@st.cache_data(show_spinner="Scoring cohort…")
def load_sample_cohort(_cols):
    """Load the bundled sample cohort, or synthesize one if absent."""
    if os.path.exists("sample_cohort.csv"):
        df = pd.read_csv("sample_cohort.csv")
        return df, "bundled sample_cohort.csv"

    # Synthetic fallback so the Space is never broken
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame({c: rng.integers(0, 2, n) for c in _cols})
    for c in _cols:
        if "num_medications" in c:
            df[c] = rng.integers(1, 40, n)
        elif "number_inpatient" in c:
            df[c] = rng.poisson(0.6, n)
        elif "number_emergency" in c:
            df[c] = rng.poisson(0.3, n)
        elif "time_in_hospital" in c:
            df[c] = rng.integers(1, 14, n)
        elif "complexity_index" in c:
            df[c] = rng.integers(0, 12, n)
    return df, "synthetic demo data (no sample_cohort.csv found)"


@st.cache_resource(show_spinner="Loading guidance index…")
def load_rag():
    """Load the FAISS index + chunks if they were committed to the Space."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    if not (os.path.exists("transitionrx_faiss.index")
            and os.path.exists("transitionrx_chunks.parquet")):
        return None

    index = faiss.read_index("transitionrx_faiss.index")
    chunks = pd.read_parquet("transitionrx_chunks.parquet").to_dict("records")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return {"index": index, "chunks": chunks, "embedder": embedder}


def align_features(df, cols):
    """Rebuild the matrix in the exact published column order."""
    return df.reindex(columns=cols, fill_value=0)


# ----------------------------------------------------------------------------------
# Load everything
# ----------------------------------------------------------------------------------
try:
    model, calibrated, META, COLS, model_source = load_model()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    MODEL_ERR = str(e)

if not MODEL_OK:
    st.error(
        f"**Could not load the model.**\n\n`{MODEL_ERR}`\n\n"
        f"Set the `HF_MODEL_REPO` variable in Space settings to your model repo "
        f"(currently `{HF_MODEL_REPO}`), or commit the four artifact files into this Space."
    )
    st.stop()

DEFAULT_THRESHOLD = float(META.get("operating_threshold", 0.5))
DEFAULT_CAPACITY = float(META.get("capacity_assumption", 0.20))

cohort_raw, cohort_source = load_sample_cohort(tuple(COLS))
X_cohort = align_features(cohort_raw, COLS)
risk_scores = model.predict_proba(X_cohort)[:, 1]
try:
    cal_scores = calibrated.predict_proba(X_cohort)[:, 1]
except Exception:
    cal_scores = risk_scores

RAG = load_rag()


# ----------------------------------------------------------------------------------
# Sidebar — the decision levers
# ----------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Decision settings")
    st.caption("Every number below is an assumption you control. "
               "Change them to match your hospital.")

    st.markdown("### Pharmacist capacity")
    capacity_pct = st.slider(
        "Share of discharges you can review", 1, 60,
        int(DEFAULT_CAPACITY * 100), 1,
        help="How many discharges pharmacy can realistically work up. "
             "This sets the flag threshold.",
    ) / 100

    st.markdown("### Cost assumptions")
    cost_readmit = st.number_input(
        "Cost per avoided readmission ($)", 1000, 50000, 10000, 500,
        help="Published diabetes readmission costs run roughly $7,800–$11,100.",
    )
    cost_intervention = st.number_input(
        "Cost per pharmacist intervention ($)", 10, 500, 60, 5,
        help="Roughly 45–60 min of pharmacist time at $55–65/hr.",
    )
    effect = st.slider(
        "Assumed risk reduction from intervention (pp)", 1, 30, 10, 1,
        help="NOT measured in this project. Drawn from medication-reconciliation "
             "literature. A prospective pilot is required to validate it.",
    ) / 100
    annual_discharges = st.number_input(
        "Annual diabetes discharges", 100, 50000, 5000, 100,
    )

    st.divider()
    st.caption(f"Model loaded from {model_source}")
    st.caption(f"ROC-AUC {META.get('test_roc_auc', float('nan')):.3f} · "
               f"{META.get('n_features', len(COLS))} features")


# Derive the operating threshold from the capacity slider
threshold = float(np.quantile(risk_scores, 1 - capacity_pct))
flags = (risk_scores >= threshold).astype(int)


# ----------------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------------
st.title("🏥 TransitionRx")
st.markdown(
    "#### Which diabetes patients should get limited pharmacist follow-up at discharge?"
)
st.markdown(
    "This tool ranks diabetes inpatient discharges by **30-day readmission risk**, using "
    "medication complexity and care-transition signals already captured at discharge — then "
    "prices out what targeting that limited pharmacist time is worth."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Discharges in cohort", f"{len(X_cohort):,}")
c2.metric("Flagged for review", f"{flags.sum():,}", f"{flags.mean():.0%} of cohort")
c3.metric("Flag threshold", f"{threshold:.2f}",
          help="Derived from your capacity slider, not a fixed 0.5 cutoff.")
c4.metric("Model ROC-AUC", f"{META.get('test_roc_auc', float('nan')):.3f}")

tabs = st.tabs([
    "📖 Start here",
    "📊 Risk dashboard",
    "👤 Patient brief",
    "💬 Guidance Q&A",
    "💰 Cost / value",
    "⚠️ Limitations",
])


# ----------------------------------------------------------------------------------
# TAB 1 — Instructions
# ----------------------------------------------------------------------------------
with tabs[0]:
    st.header("What this is, and how to use it")

    st.markdown("""
### The business problem

Diabetes patients are among the most frequently readmitted inpatients, and 30-day
readmissions carry direct financial penalties under CMS's **Hospital Readmissions Reduction
Program (HRRP)**. Published research finds roughly **a fifth of readmissions are
medication-related**, and most of those are judged potentially preventable.

The mechanism this project targets is what happens *at discharge*: too many medications,
too much last-minute change to the regimen, and a thin handoff. Those are **modifiable** —
unlike most other readmission risk factors.

### The decision this supports

> Pharmacy can only work up so many discharges a day. **Which ones?**

This is a resource-targeting decision, not a prediction exercise. The model ranks; a
pharmacist decides.

### How to use each tab

| Tab | What it's for |
|---|---|
| **📊 Risk dashboard** | See the cohort ranked by risk, and how capacity trades against captured risk |
| **👤 Patient brief** | Score one patient and see *why* they were flagged |
| **💬 Guidance Q&A** | Ask what CMS/AHRQ guidance says about discharge planning |
| **💰 Cost / value** | Price out the targeting decision under your own assumptions |
| **⚠️ Limitations** | What this model cannot do — read before acting on it |

### Start here

1. Set **pharmacist capacity** in the left sidebar to what your team can actually handle.
2. Watch the flag threshold and captured-risk numbers move in the **Risk dashboard**.
3. Adjust **cost assumptions** to your hospital's figures.
4. Read the **Limitations** tab before drawing conclusions.

> **The sidebar is the point.** Every headline number in this app is driven by assumptions
> you control. If you disagree with an assumption, change it and watch the recommendation
> move — that's more useful than any single number the model produces.
    """)

    st.info(
        f"**Data source:** {cohort_source}. The model itself was trained on the "
        "[UCI Diabetes 130-US Hospitals dataset]"
        "(https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) "
        "(1999–2008, ~70k patient encounters after cleaning)."
    )


# ----------------------------------------------------------------------------------
# TAB 2 — Risk dashboard
# ----------------------------------------------------------------------------------
with tabs[1]:
    st.header("Cohort risk dashboard")

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(risk_scores, bins=40, color=PRIMARY, alpha=.85)
        ax.axvline(threshold, color=ACCENT, ls="--", lw=2,
                   label=f"Flag threshold {threshold:.2f}")
        ax.set_xlabel("Predicted 30-day readmission risk")
        ax.set_ylabel("Patients")
        ax.set_title("Risk distribution across the cohort")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "Most patients sit at low risk. The tool's job is separating the right tail "
            "from the bulk — not assigning everyone an accurate probability."
        )

    with right:
        caps = np.arange(0.05, 0.61, 0.05)
        captured = []
        for cp in caps:
            t = np.quantile(risk_scores, 1 - cp)
            f = risk_scores >= t
            # Expected share of risk captured (risk-weighted, since labels are unknown here)
            captured.append(risk_scores[f].sum() / risk_scores.sum())

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(caps * 100, np.array(captured) * 100, marker="o", color="#7030A0", lw=2)
        ax.axvline(capacity_pct * 100, color=ACCENT, ls="--", lw=2,
                   label=f"Your capacity: {capacity_pct:.0%}")
        ax.set_xlabel("Pharmacist capacity (% of discharges reviewed)")
        ax.set_ylabel("% of total predicted risk captured")
        ax.set_title("Capacity vs. captured risk")
        ax.legend()
        ax.grid(alpha=.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "Diminishing returns: the first slice of capacity buys far more captured risk "
            "than the last. This curve is the argument for targeting over blanket review."
        )

    st.subheader("Ranked worklist")
    st.markdown(
        "What a case manager would actually work from — highest risk first, "
        "capped at your capacity."
    )

    worklist = pd.DataFrame({
        "Rank": np.arange(1, len(risk_scores) + 1),
        "Patient ID": [f"PT-{i:05d}" for i in range(len(risk_scores))],
        "Risk score": risk_scores.round(3),
        "Calibrated risk": cal_scores.round(3),
        "Action": np.where(flags == 1, "🔴 Flag for pharmacist", "⚪ Routine discharge"),
    }).sort_values("Risk score", ascending=False).reset_index(drop=True)
    worklist["Rank"] = np.arange(1, len(worklist) + 1)

    show_flagged = st.checkbox("Show flagged patients only", value=True)
    view = worklist[worklist["Action"].str.contains("Flag")] if show_flagged else worklist

    st.dataframe(view.head(200), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download worklist (CSV)",
        worklist.to_csv(index=False).encode(),
        "transitionrx_worklist.csv",
        "text/csv",
    )

    st.info(
        "**Business interpretation.** The value here is *lift*, not raw accuracy. Even a "
        "modest model concentrates risk into the top slice of the ranking, which is what "
        "makes a fixed pharmacist budget go further than reviewing patients in admission order."
    )


# ----------------------------------------------------------------------------------
# TAB 3 — Patient brief
# ----------------------------------------------------------------------------------
with tabs[2]:
    st.header("Individual patient brief")
    st.markdown("Score one patient and see the drivers behind the flag.")

    pick = st.selectbox(
        "Select a patient from the cohort",
        options=worklist["Patient ID"].head(50).tolist(),
        help="Sorted highest-risk first.",
    )
    row_pos = int(pick.split("-")[1])

    patient_X = X_cohort.iloc[[row_pos]]
    p_risk = float(model.predict_proba(patient_X)[:, 1][0])
    flagged = p_risk >= threshold

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk score", f"{p_risk:.3f}")
    m2.metric("Threshold", f"{threshold:.3f}")
    m3.metric("Decision", "🔴 FLAG" if flagged else "⚪ Routine")

    st.subheader("Why this patient scored the way they did")

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(patient_X)
        sv = sv[0] if isinstance(sv, list) else sv
        contrib = pd.DataFrame({
            "Feature": patient_X.columns,
            "Value": patient_X.iloc[0].values,
            "Impact": sv[0] if sv.ndim > 1 else sv,
        })
        contrib["abs"] = contrib["Impact"].abs()
        top = contrib.nlargest(10, "abs").sort_values("Impact")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = [ACCENT if v > 0 else PRIMARY for v in top["Impact"]]
        ax.barh(top["Feature"], top["Impact"], color=colors)
        ax.axvline(0, color="k", lw=.8)
        ax.set_xlabel("← lowers risk    |    raises risk →")
        ax.set_title("Top drivers for this patient (SHAP)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("**In plain language:**")
        for _, r in top.iloc[::-1].head(5).iterrows():
            direction = "raises" if r["Impact"] > 0 else "lowers"
            st.markdown(
                f"- `{r['Feature']}` = **{r['Value']:.0f}** — {direction} risk "
                f"({r['Impact']:+.3f})"
            )
        st.session_state["last_drivers"] = [
            {"feature": r["Feature"], "value": r["Value"],
             "direction": "increases" if r["Impact"] > 0 else "decreases"}
            for _, r in top.iloc[::-1].head(5).iterrows()
        ]
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

    st.info(
        "**Appropriate use.** This brief routes attention and explains a ranking. It is "
        "not a diagnosis, and it does not recommend drugs or doses. A pharmacist reviews "
        "the chart and decides."
    )


# ----------------------------------------------------------------------------------
# TAB 4 — RAG Guidance Q&A
# ----------------------------------------------------------------------------------
with tabs[3]:
    st.header("Discharge guidance Q&A")
    st.markdown(
        "Ask what **CMS and AHRQ guidance** says about discharge planning, medication "
        "reconciliation, and follow-up. Answers are grounded in a curated document corpus "
        "with citations — this is not a general chatbot."
    )

    if RAG is None:
        st.warning(
            "**Guidance index not loaded.** To enable this tab, commit "
            "`transitionrx_faiss.index` and `transitionrx_chunks.parquet` "
            "(produced by the RAG notebook) into this Space."
        )
        st.markdown("""
**What this tab does when enabled:**

1. Embeds your question with `all-MiniLM-L6-v2`
2. Retrieves the most similar passages from the CMS/AHRQ corpus via FAISS
3. Applies a **similarity floor** — off-topic questions are declined rather than answered
4. Generates a cited answer constrained to the retrieved excerpts
5. **Verifies every citation** maps to a real retrieved passage
        """)
    else:
        examples = [
            "What does AHRQ recommend for medication reconciliation at discharge?",
            "What are the required elements of hospital discharge planning?",
            "What follow-up contact is recommended after a patient goes home?",
            "How should staff educate patients with low health literacy?",
        ]
        picked = st.selectbox("Example questions", ["— type your own —"] + examples)
        default_q = "" if picked.startswith("—") else picked
        question = st.text_input("Your question", value=default_q)

        if st.button("Ask", type="primary") and question.strip():
            emb = RAG["embedder"].encode([question], normalize_embeddings=True).astype("float32")
            scores, idxs = RAG["index"].search(emb, 8)

            MIN_SIM, MAX_PER_SRC = 0.25, 2
            hits, per_src = [], {}
            for i, s in zip(idxs[0], scores[0]):
                if i < 0 or s < MIN_SIM:
                    continue
                ch = RAG["chunks"][i]
                src = ch.get("source", "unknown")
                if per_src.get(src, 0) >= MAX_PER_SRC:
                    continue
                per_src[src] = per_src.get(src, 0) + 1
                hits.append({**ch, "score": float(s)})
                if len(hits) == 4:
                    break

            if not hits:
                st.warning(
                    "The document corpus does not cover this question. Only readmission, "
                    "discharge-planning, care-transition, and healthcare-quality documents "
                    "are indexed."
                )
            else:
                context = "\n\n---\n\n".join(
                    f"[{n}] SOURCE: {h.get('source')}"
                    f"{', p.' + str(h['page']) if h.get('page') else ''}\n{h['text']}"
                    for n, h in enumerate(hits, 1)
                )

                if OPENAI_API_KEY:
                    try:
                        from openai import OpenAI

                        client = OpenAI(api_key=OPENAI_API_KEY)
                        prompt = (
                            "You are a discharge-planning guidance assistant for a hospital "
                            "pharmacy and care-management team.\n\n"
                            "Rules:\n"
                            "1. Answer ONLY from the excerpts. No outside clinical knowledge.\n"
                            "2. Cite [n] inline after each claim.\n"
                            "3. If the excerpts don't cover it, say so.\n"
                            "4. Write for a busy pharmacist: concrete, actionable, brief.\n"
                            "5. Recommend PROCESS steps only. Never specific drugs or doses.\n\n"
                            f"EXCERPTS:\n{context}\n\nQUESTION: {question}"
                        )
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini", max_tokens=700, temperature=0,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        answer = resp.choices[0].message.content
                        st.markdown("### Answer")
                        st.markdown(answer)

                        import re
                        cited = {int(x) for x in re.findall(r"\[(\d+)\]", answer or "")}
                        invalid = cited - set(range(1, len(hits) + 1))
                        if invalid:
                            st.error(f"⚠️ Invalid citations detected: {sorted(invalid)}")
                        elif not cited:
                            st.warning("⚠️ Answer contains no citations — review manually.")
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        st.info("Showing retrieved passages instead.")
                else:
                    st.info(
                        "**Retrieval-only mode.** No `OPENAI_API_KEY` set in Space secrets, "
                        "so the retrieved source passages are shown directly without a "
                        "generated summary. The retrieval and citation machinery is "
                        "unchanged."
                    )

                st.markdown("### Sources retrieved")
                for n, h in enumerate(hits, 1):
                    loc = f", p.{h['page']}" if h.get("page") else ""
                    with st.expander(f"[{n}] {h.get('source')}{loc} — similarity {h['score']:.3f}"):
                        st.write(h["text"])

    st.caption(
        "Grounding is enforced three ways: a retrieval similarity floor, a prompt that "
        "requires citation or refusal, and post-hoc verification that every [n] maps to a "
        "real retrieved passage."
    )


# ----------------------------------------------------------------------------------
# TAB 5 — Cost / value
# ----------------------------------------------------------------------------------
with tabs[4]:
    st.header("Cost / value and resource impact")
    st.markdown(
        "What is targeting pharmacist time actually worth? Every figure below flows from "
        "the sidebar assumptions — change them to match your hospital."
    )

    def scenario(cap_pct):
        t = np.quantile(risk_scores, 1 - cap_pct)
        f = risk_scores >= t
        n_flagged = annual_discharges * cap_pct
        # Expected true positives from calibrated risk among flagged
        exp_rate = float(cal_scores[f].mean()) if f.sum() else 0.0
        exp_readmits = n_flagged * exp_rate
        prevented = exp_readmits * effect
        avoided = prevented * cost_readmit
        spend = n_flagged * cost_intervention
        return {
            "Capacity": f"{cap_pct:.0%}",
            "Threshold": round(float(t), 3),
            "Patients flagged": int(n_flagged),
            "Expected readmits in group": round(exp_readmits, 1),
            "Readmissions prevented": round(prevented, 1),
            "Cost avoided": avoided,
            "Intervention cost": spend,
            "Net value": avoided - spend,
            "ROI": (avoided / spend) if spend else 0,
        }

    rows = [scenario(c) for c in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]]
    sdf = pd.DataFrame(rows)

    cur = scenario(capacity_pct)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Patients flagged / yr", f"{cur['Patients flagged']:,}")
    k2.metric("Readmissions prevented", f"{cur['Readmissions prevented']:.0f}")
    k3.metric("Net value", f"${cur['Net value']:,.0f}")
    k4.metric("ROI", f"{cur['ROI']:.1f}×")

    st.subheader("Scenario comparison")
    disp = sdf.copy()
    for c in ["Cost avoided", "Intervention cost", "Net value"]:
        disp[c] = disp[c].map(lambda v: f"${v:,.0f}")
    disp["ROI"] = disp["ROI"].map(lambda v: f"{v:.1f}×")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        caps = np.arange(0.05, 0.61, 0.05)
        for eff in [0.05, 0.10, 0.15, 0.20]:
            nets = []
            for cp in caps:
                t = np.quantile(risk_scores, 1 - cp)
                f = risk_scores >= t
                n_fl = annual_discharges * cp
                rate = float(cal_scores[f].mean()) if f.sum() else 0
                nets.append(n_fl * rate * eff * cost_readmit - n_fl * cost_intervention)
            ax.plot(caps * 100, nets, marker="o", label=f"{eff:.0%} reduction",
                    lw=2 if abs(eff - effect) < 1e-9 else 1.2,
                    alpha=1.0 if abs(eff - effect) < 1e-9 else .55)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.axvline(capacity_pct * 100, color=ACCENT, ls=":", lw=2)
        ax.set_xlabel("Pharmacist capacity (%)")
        ax.set_ylabel("Estimated annual net value ($)")
        ax.set_title("Net value sensitivity")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(sdf))
        ax.bar(x - .2, sdf["Cost avoided"], .4, label="Cost avoided", color=PRIMARY)
        ax.bar(x + .2, sdf["Intervention cost"], .4, label="Intervention cost", color=ACCENT)
        ax.set_xticks(x)
        ax.set_xticklabels(sdf["Capacity"])
        ax.set_xlabel("Pharmacist capacity")
        ax.set_ylabel("Dollars")
        ax.set_title("Avoided cost vs. spend")
        ax.legend()
        ax.grid(alpha=.3, axis="y")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.warning(
        f"**These are stated assumptions, not hospital-reported figures.** The "
        f"{effect:.0%} risk reduction in particular is drawn from medication-reconciliation "
        "literature and was **not measured in this project**. It is the single assumption "
        "most likely to be wrong, and net value scales almost linearly with it — which is "
        "exactly why it's a slider rather than a hard-coded constant. A prospective pilot "
        "would be required to validate it."
    )

    st.info(
        "**Resource impact.** At your current setting, pharmacy reviews "
        f"**{cur['Patients flagged']:,} patients/year** "
        f"(~{cur['Patients flagged'] / 250:.0f}/working day). Check that against actual "
        "staffing before treating the net-value figure as achievable — the model can rank "
        "patients, but it cannot create pharmacist hours."
    )


# ----------------------------------------------------------------------------------
# TAB 6 — Limitations
# ----------------------------------------------------------------------------------
with tabs[5]:
    st.header("Limitations, cautions, and appropriate use")

    st.error("""
### Not for clinical decision-making

This tool **ranks patients for a workflow decision**. It does not diagnose, does not
recommend treatment, and must not be used to deny or restrict care. A clinician reviews
the chart and decides. Every flag is a suggestion to *look*, not a conclusion.
    """)

    st.subheader("What the model can't see")
    st.markdown("""
| Missing | Why it matters |
|---|---|
| Vital signs and lab trends | Clinical instability at discharge is a known readmission driver |
| Social determinants | Housing, transport, food security, and caregiver support drive bouncebacks |
| Medication adherence | The model sees what was *prescribed*, never what was *taken* |
| Discharge summary text | The handoff quality itself is only crudely proxied |
| Outpatient follow-up | Whether the patient actually saw anyone post-discharge |
    """)

    st.subheader("Performance ceiling")
    st.markdown(f"""
ROC-AUC is **{META.get('test_roc_auc', float('nan')):.3f}** — consistent with published
work on this dataset, and a realistic ceiling for administrative billing data.

**Interpretation:** this is a *ranking* tool that concentrates risk into the top of a
worklist. It is **not** a diagnostic, and individual scores should not be read as
literal probabilities of readmission.
    """)

    st.subheader("Data vintage")
    st.warning(
        "Training data is the UCI Diabetes 130-US Hospitals dataset, **1999–2008**. "
        "Coding practice, diabetes pharmacotherapy, and discharge workflows have all "
        "changed since. Case mix at any given hospital will differ from the 130 hospitals "
        "in the training set. Re-training on local recent data is required before real use."
    )

    st.subheader("Fairness — not yet audited")
    st.error(
        "**Subgroup performance across race, age, sex, and payer has not been evaluated.** "
        "The training data has known demographic imbalance and substantial missingness in "
        "race. A model that allocates a beneficial resource can entrench disparities if it "
        "underperforms for some groups. **A subgroup audit is required before deployment.**"
    )

    st.subheader("The assumption most likely to be wrong")
    st.markdown("""
The model predicts **readmission risk**. The value case assumes intervention **reduces**
that risk by a given amount. Those are different claims:

- *Risk* is what the model estimates from data.
- *Preventability* is not modeled at all.
- *Responsiveness to pharmacist follow-up* is assumed from literature, not measured here.

A patient can be genuinely high-risk and still not benefit from medication reconciliation.
This gap is the strongest argument for a **prospective pilot** before scaling.
    """)

    st.subheader("Appropriate use summary")
    st.success("""
**Appropriate:** prioritizing a worklist · scenario planning for staffing ·
starting a conversation about which patients need a closer look

**Not appropriate:** clinical diagnosis · treatment or dosing decisions ·
denying or limiting care · individual determinations without clinician review ·
any use without a local subgroup fairness audit
    """)


st.divider()
st.caption(
    "TransitionRx — Decision Analytics course project. Model: tuned XGBoost on the UCI "
    "Diabetes 130-US Hospitals dataset. RAG corpus: CMS and AHRQ discharge-planning "
    "guidance. Educational demonstration — not a medical device."
)
