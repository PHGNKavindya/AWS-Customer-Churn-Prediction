"""
AWS Customer Churn Prediction - Interactive Analytics & Machine Learning Dashboard
Project: AWS Customer Churn Prediction

Run with:  streamlit run app.py
Reads the same artifacts as before:
  data/processed/customer_churn_cleaned.csv
  data/output/final_churn_model.joblib, final_churn_threshold.joblib
  data/output/model_comparison.csv, cross_validation_results.csv, random_forest_feature_importance.csv
"""

import inspect
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# DESIGN TOKENS
# ==============================================================================
INK = "#14213D"        # primary text, header
SLATE = "#5B6478"      # secondary text
LINE = "#E3E7EE"       # borders
CANVAS = "#F4F6FA"     # page background
TEAL = "#0F766E"       # retained / good
CRIMSON = "#BE123C"    # churn / risk
INDIGO = "#4338CA"     # neutral accent
AMBER = "#B45309"      # watch
FONT = "Plus Jakarta Sans, Segoe UI, Helvetica, Arial, sans-serif"

TENURE_ORDER = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]
SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
DIMENSIONS = {
    "Contract type": "Contract",
    "Internet service": "InternetService",
    "Payment method": "PaymentMethod",
    "Tenure group": "TenureGroup",
    "Senior citizen": "SeniorGroup",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "Paperless billing": "PaperlessBilling",
    "Tech support": "TechSupport",
    "Online security": "OnlineSecurity",
    "Gender": "gender",
}
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

# Streamlit renamed use_container_width -> width="stretch"; support both.
_USES_WIDTH_API = "width" in inspect.signature(st.plotly_chart).parameters
STRETCH = {"width": "stretch"} if _USES_WIDTH_API else {"use_container_width": True}

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea, select {{
    font-family: {FONT};
}}
.stApp {{ background: {CANVAS}; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}

/* Header */
.hero {{
    background: {INK};
    color: #fff;
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 18px;
    display: flex; justify-content: space-between; align-items: center; gap: 24px; flex-wrap: wrap;
}}
.hero h1 {{ font-size: 1.65rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; color: #fff; padding: 0; }}
.hero p {{ margin: 4px 0 0 0; color: #B8C2D9; font-size: 0.95rem; }}
.chip {{
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; margin-left: 6px;
    background: rgba(255,255,255,0.12); color: #E6ECFA; border: 1px solid rgba(255,255,255,0.18);
}}
.chip.ok {{ background: rgba(15,118,110,0.35); border-color: rgba(94,234,212,0.5); color: #CCFBF1; }}
.chip.warn {{ background: rgba(180,83,9,0.35); border-color: rgba(251,191,36,0.5); color: #FEF3C7; }}

/* Filter chips under the header */
.filter-row {{ margin: -4px 0 14px 0; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; color: {SLATE}; font-size: 0.85rem; }}
.fchip {{ background: #fff; border: 1px solid {LINE}; border-radius: 999px; padding: 3px 11px; font-size: 0.8rem; color: {INK}; font-weight: 600; }}

/* KPI cards */
.kpi {{
    background: #fff; border: 1px solid {LINE}; border-radius: 12px;
    padding: 16px 18px; height: 100%;
}}
.kpi .t {{ font-size: 0.82rem; font-weight: 600; color: {SLATE}; margin-bottom: 6px; }}
.kpi .v {{ font-size: 1.8rem; font-weight: 800; color: {INK}; line-height: 1.1; letter-spacing: -0.02em; }}
.kpi .s {{ font-size: 0.8rem; color: {SLATE}; margin-top: 6px; }}
.pill {{ display: inline-block; font-size: 0.74rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; margin-left: 8px; vertical-align: middle; }}
.pill.good {{ background: #D5F0EC; color: {TEAL}; }}
.pill.bad {{ background: #FCE0E7; color: {CRIMSON}; }}
.pill.neutral {{ background: #EDF0F6; color: {SLATE}; }}

/* Bordered containers = cards */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border-radius: 12px; border-color: {LINE}; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 10px 16px; font-weight: 600; color: {SLATE}; border-radius: 8px 8px 0 0; }}
.stTabs [aria-selected="true"] {{ color: {INK}; background: #fff; }}

/* Section headings */
.section-title {{ font-size: 1.05rem; font-weight: 700; color: {INK}; margin: 0 0 2px 0; }}
.section-sub {{ font-size: 0.85rem; color: {SLATE}; margin: 0 0 8px 0; }}

/* Risk result */
.risk-card {{ border-radius: 12px; padding: 18px 20px; border: 1px solid; }}
.risk-card h3 {{ margin: 0 0 4px 0; font-size: 1.25rem; font-weight: 800; }}
.risk-card p {{ margin: 0; font-size: 0.92rem; }}
.risk-card.high {{ background: #FDECF0; border-color: #F3B6C4; color: #7F1030; }}
.risk-card.watch {{ background: #FEF5E6; border-color: #F1D29B; color: #7A3E06; }}
.risk-card.low {{ background: #E6F5F2; border-color: #A8DDD5; color: #0B4F49; }}
.flag {{ display: inline-block; background: #EEF0FA; color: {INDIGO}; font-weight: 600; font-size: 0.8rem; padding: 3px 10px; border-radius: 999px; margin: 2px 4px 2px 0; }}

/* Buttons & focus */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {{
    border-radius: 8px; font-weight: 600;
}}
:focus-visible {{ outline: 2px solid {INDIGO} !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# DATA & MODEL LOADING
# ==============================================================================
def assign_tenure_group(t):
    """Matches sql/churn_analysis.sql exactly."""
    if t <= 12:
        return "0-12 months"
    if t <= 24:
        return "13-24 months"
    if t <= 48:
        return "25-48 months"
    return "49+ months"


@st.cache_data
def load_cleaned_data(file_path: str = "data/processed/customer_churn_cleaned.csv") -> pd.DataFrame:
    """Load and validate the local cleaned customer churn dataset."""
    if not os.path.exists(file_path):
        alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), file_path)
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(
                f"Cleaned dataset not found at '{file_path}'. "
                "Run the data preparation notebook to generate this file."
            )

    df = pd.read_csv(file_path)

    required_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges", "Churn",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Cleaned dataset is missing required schema columns: {missing}")

    df["TenureGroup"] = pd.Categorical(
        df["tenure"].apply(assign_tenure_group), categories=TENURE_ORDER, ordered=True
    )
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorGroup"] = np.where(df["SeniorCitizen"] == 1, "Senior", "Non-senior")
    return df


@st.cache_resource
def load_churn_model():
    """Load the Gradient Boosting pipeline and calibrated threshold."""
    import joblib

    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "data", "output", "final_churn_model.joblib")
    thresh_path = os.path.join(base_dir, "data", "output", "final_churn_threshold.joblib")

    if not os.path.exists(model_path):
        return None, None, f"Model file not found at {model_path}."
    try:
        model = joblib.load(model_path)
    except Exception as e:
        return None, None, f"Could not load the model: {e}"

    threshold = 0.322712
    if os.path.exists(thresh_path):
        try:
            threshold = float(joblib.load(thresh_path))
        except Exception:
            pass
    return model, threshold, "Model loaded"


@st.cache_data
def load_model_evaluation_metrics():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    out = os.path.join(base_dir, "data", "output")

    def _read(name):
        p = os.path.join(out, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    return (
        _read("model_comparison.csv"),
        _read("cross_validation_results.csv"),
        _read("random_forest_feature_importance.csv"),
    )


# ==============================================================================
# HELPERS
# ==============================================================================
def style_fig(fig, height=340, legend_bottom=False):
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=20, l=10, r=10),
        hoverlabel=dict(font_family=FONT, bgcolor="#fff", bordercolor=LINE, font_color=INK),
        title=None,
    )
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
    fig.update_xaxes(gridcolor="#EEF1F6", zerolinecolor="#EEF1F6")
    fig.update_yaxes(gridcolor="#EEF1F6", zerolinecolor="#EEF1F6")
    return fig


def show_chart(fig, key=None):
    st.plotly_chart(fig, config=PLOTLY_CONFIG, key=key, **STRETCH)


def section(title, sub=""):
    html = f'<div class="section-title">{title}</div>'
    if sub:
        html += f'<div class="section-sub">{sub}</div>'
    st.markdown(html, unsafe_allow_html=True)


def churn_by(df, col):
    g = (
        df.groupby(col, observed=True)
        .agg(customers=("ChurnFlag", "size"), churned=("ChurnFlag", "sum"))
        .reset_index()
    )
    g["churn_rate"] = g["churned"] / g["customers"] * 100
    return g


def kpi_card(title, value, sub="", delta_text=None, tone="neutral", value_color=None):
    pill = f'<span class="pill {tone}">{delta_text}</span>' if delta_text else ""
    color = f'style="color:{value_color};"' if value_color else ""
    return (
        f'<div class="kpi"><div class="t">{title}</div>'
        f'<div class="v" {color}>{value}{pill}</div>'
        f'<div class="s">{sub}</div></div>'
    )


def delta_pill(current, baseline, unit="", fmt="{:+.1f}", higher_is_bad=True):
    diff = current - baseline
    if abs(diff) < 0.05:
        return "same as all", "neutral"
    bad = diff > 0 if higher_is_bad else diff < 0
    return f"{fmt.format(diff)}{unit} vs all", ("bad" if bad else "good")


def risk_tier(prob, thr):
    if prob >= thr:
        return "High churn risk", "high"
    if prob >= thr * 0.6:
        return "Worth watching", "watch"
    return "Low churn risk", "low"


def build_insights(fdf, min_n=30):
    items = []
    for col, label in [("Contract", "contract"), ("InternetService", "internet service"),
                       ("PaymentMethod", "payment method"), ("TenureGroup", "tenure")]:
        g = churn_by(fdf, col)
        g = g[g["customers"] >= min_n]
        if len(g) < 2:
            continue
        top = g.loc[g["churn_rate"].idxmax()]
        low = g.loc[g["churn_rate"].idxmin()]
        gap = top["churn_rate"] - low["churn_rate"]
        items.append((gap, f"By **{label}**: **{top[col]}** customers churn at **{top['churn_rate']:.1f}%**, "
                           f"versus **{low['churn_rate']:.1f}%** for **{low[col]}**."))
    items.sort(key=lambda x: -x[0])
    return [t for _, t in items[:3]]


# ==============================================================================
# PREDICTOR STATE
# ==============================================================================
PRED_DEFAULTS = {
    "p_gender": "Female", "p_senior": "No", "p_partner": "No", "p_dependents": "No",
    "p_phone": "Yes", "p_multilines": "No", "p_internet": "Fiber optic",
    "p_security": "No", "p_backup": "No", "p_device": "No", "p_tech": "No",
    "p_tv": "No", "p_movies": "No",
    "p_contract": "Month-to-month", "p_paperless": "Yes", "p_payment": "Electronic check",
    "p_tenure": 3, "p_monthly": 75.0, "p_auto_total": True, "p_total": 225.0,
}
PRESETS = {
    "New fiber customer, month-to-month": {},
    "Established DSL customer, one-year contract": {
        "p_internet": "DSL", "p_contract": "One year", "p_tenure": 30, "p_monthly": 55.0,
        "p_payment": "Bank transfer (automatic)", "p_paperless": "No", "p_partner": "Yes",
        "p_security": "Yes", "p_tech": "Yes", "p_backup": "Yes",
    },
    "Loyal two-year bundle customer": {
        "p_internet": "Fiber optic", "p_contract": "Two year", "p_tenure": 60, "p_monthly": 95.0,
        "p_payment": "Credit card (automatic)", "p_partner": "Yes", "p_dependents": "Yes",
        "p_security": "Yes", "p_tech": "Yes", "p_backup": "Yes", "p_device": "Yes",
        "p_tv": "Yes", "p_movies": "Yes", "p_multilines": "Yes",
    },
    "Senior on a budget plan": {
        "p_senior": "Yes", "p_internet": "No", "p_contract": "Month-to-month", "p_tenure": 8,
        "p_monthly": 25.0, "p_payment": "Mailed check", "p_paperless": "No",
    },
}


def init_predictor_state():
    for k, v in PRED_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("preset", "New fiber customer, month-to-month")


def apply_preset():
    name = st.session_state.get("preset")
    if name in PRESETS:
        for k, v in {**PRED_DEFAULTS, **PRESETS[name]}.items():
            st.session_state[k] = v


def customer_from_state(s):
    """Build one model-ready record; enforces consistency between related fields."""
    no_internet = s["p_internet"] == "No"
    no_phone = s["p_phone"] == "No"
    svc = lambda k: "No internet service" if no_internet else s[k]
    total = round(float(s["p_tenure"] * s["p_monthly"]), 2) if s["p_auto_total"] else float(s["p_total"])
    return {
        "gender": s["p_gender"],
        "SeniorCitizen": 1 if s["p_senior"] == "Yes" else 0,
        "Partner": s["p_partner"],
        "Dependents": s["p_dependents"],
        "tenure": int(s["p_tenure"]),
        "PhoneService": s["p_phone"],
        "MultipleLines": "No phone service" if no_phone else s["p_multilines"],
        "InternetService": s["p_internet"],
        "OnlineSecurity": svc("p_security"),
        "OnlineBackup": svc("p_backup"),
        "DeviceProtection": svc("p_device"),
        "TechSupport": svc("p_tech"),
        "StreamingTV": svc("p_tv"),
        "StreamingMovies": svc("p_movies"),
        "Contract": s["p_contract"],
        "PaperlessBilling": s["p_paperless"],
        "PaymentMethod": s["p_payment"],
        "MonthlyCharges": float(s["p_monthly"]),
        "TotalCharges": total,
    }


def retention_scenarios(base):
    """What-if changes that apply to this customer."""
    out = []
    if base["Contract"] == "Month-to-month":
        out.append(("Move to a one-year contract", {"Contract": "One year"}))
    if base["Contract"] != "Two year":
        out.append(("Move to a two-year contract", {"Contract": "Two year"}))
    if base["PaymentMethod"] in ("Electronic check", "Mailed check"):
        out.append(("Switch to automatic bank transfer", {"PaymentMethod": "Bank transfer (automatic)"}))
    if base["InternetService"] != "No":
        if base["TechSupport"] == "No":
            out.append(("Add tech support", {"TechSupport": "Yes"}))
        if base["OnlineSecurity"] == "No":
            out.append(("Add online security", {"OnlineSecurity": "Yes"}))
    return out


def profile_flags(c):
    flags = []
    if c["Contract"] == "Month-to-month":
        flags.append("Month-to-month contract")
    if c["PaymentMethod"] == "Electronic check":
        flags.append("Pays by electronic check")
    if c["InternetService"] == "Fiber optic":
        flags.append("Fiber optic plan")
    if c["tenure"] <= 12:
        flags.append("Under 12 months tenure")
    if c["InternetService"] != "No" and c["TechSupport"] == "No":
        flags.append("No tech support")
    if c["InternetService"] != "No" and c["OnlineSecurity"] == "No":
        flags.append("No online security")
    if c["MonthlyCharges"] >= 80:
        flags.append("High monthly charges")
    return flags


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    try:
        df = load_cleaned_data()
    except Exception as e:
        st.error(f"Could not load the dataset: {e}")
        st.info("Check that `data/processed/customer_churn_cleaned.csv` exists in the project.")
        return

    model, threshold, model_status = load_churn_model()
    comparison_df, cv_df, feat_df = load_model_evaluation_metrics()
    init_predictor_state()

    charge_bounds = (round(float(df["MonthlyCharges"].min()), 2), round(float(df["MonthlyCharges"].max()), 2))
    st.session_state.setdefault("f_charges", charge_bounds)

    # ---------------------------------------------------------------- sidebar
    def reset_filters():
        for k in ("f_contract", "f_internet", "f_payment", "f_tenure"):
            st.session_state[k] = []
        st.session_state["f_charges"] = charge_bounds

    with st.sidebar:
        st.markdown(f"<div style='font-size:1.15rem;font-weight:800;color:{INK};'>Churn Analytics</div>"
                    f"<div style='color:{SLATE};font-size:0.85rem;margin-bottom:12px;'>Filter the customer cohort</div>",
                    unsafe_allow_html=True)

        f_contract = st.multiselect("Contract type", sorted(df["Contract"].unique()), key="f_contract",
                                    placeholder="All contracts")
        f_internet = st.multiselect("Internet service", sorted(df["InternetService"].unique()), key="f_internet",
                                    placeholder="All services")
        f_payment = st.multiselect("Payment method", sorted(df["PaymentMethod"].unique()), key="f_payment",
                                   placeholder="All methods")
        f_tenure = st.multiselect("Tenure group", TENURE_ORDER, key="f_tenure", placeholder="All tenures")
        f_charges = st.slider("Monthly charges ($)", min_value=charge_bounds[0], max_value=charge_bounds[1],
                              key="f_charges")

        st.button("Reset filters", on_click=reset_filters, **STRETCH)

        st.divider()
        with st.expander("About this dashboard"):
            st.markdown(
                "Reads the cleaned dataset and local model files. No AWS credentials or billing needed. "
                "All figures are anonymized aggregates.\n\n"
                "**Project roles**\n"
                "- Member 1: Data science, EDA & ML modeling\n"
                "- Member 2: AWS data engineering (S3 / Glue / Athena)\n"
                "- Member 3: AWS ML deployment (SageMaker)\n"
                "- Member 4: Dashboard, architecture, docs & testing"
            )
        st.caption(f"Full cohort: {len(df):,} customers")

    # ---------------------------------------------------------------- filtering
    fdf = df
    if f_contract:
        fdf = fdf[fdf["Contract"].isin(f_contract)]
    if f_internet:
        fdf = fdf[fdf["InternetService"].isin(f_internet)]
    if f_payment:
        fdf = fdf[fdf["PaymentMethod"].isin(f_payment)]
    if f_tenure:
        fdf = fdf[fdf["TenureGroup"].isin(f_tenure)]
    fdf = fdf[(fdf["MonthlyCharges"] >= f_charges[0]) & (fdf["MonthlyCharges"] <= f_charges[1])]

    # ---------------------------------------------------------------- header
    model_chip = ('<span class="chip ok">Model ready</span>' if model is not None
                  else '<span class="chip warn">Model unavailable</span>')
    st.markdown(
        f"""<div class="hero">
            <div><h1>Customer churn analytics</h1>
            <p>See who is leaving, why, and what to do about it.</p></div>
            <div><span class="chip">{len(df):,} customers</span>
            <span class="chip">Local data</span>{model_chip}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    chips = []
    for label, sel in [("Contract", f_contract), ("Internet", f_internet),
                       ("Payment", f_payment), ("Tenure", f_tenure)]:
        if sel:
            chips.append(f"{label}: {', '.join(sel)}")
    if tuple(f_charges) != charge_bounds:
        chips.append(f"Charges: ${f_charges[0]:.0f} to ${f_charges[1]:.0f}")
    chip_html = "".join(f'<span class="fchip">{c}</span>' for c in chips) or "No filters applied"
    st.markdown(
        f'<div class="filter-row"><span>Showing <b>{len(fdf):,}</b> of {len(df):,} customers</span>'
        f'<span>|</span>{chip_html}</div>',
        unsafe_allow_html=True,
    )

    if len(fdf) == 0:
        st.warning("No customers match these filters. Widen a filter or select Reset filters in the sidebar.")
        return

    tab_overview, tab_drivers, tab_predict, tab_models, tab_data = st.tabs(
        ["Overview", "Churn drivers", "Risk predictor", "Model performance", "Data explorer"]
    )

    overall_rate = df["ChurnFlag"].mean() * 100

    # ======================================================================
    # TAB 1 - OVERVIEW
    # ======================================================================
    with tab_overview:
        total = len(fdf)
        churned = int(fdf["ChurnFlag"].sum())
        retained = total - churned
        churn_rate = churned / total * 100
        avg_monthly = fdf["MonthlyCharges"].mean()
        avg_tenure = fdf["tenure"].mean()
        lost_revenue = fdf.loc[fdf["ChurnFlag"] == 1, "MonthlyCharges"].sum()
        lost_share = lost_revenue / fdf["MonthlyCharges"].sum() * 100

        d_rate, t_rate = delta_pill(churn_rate, overall_rate, " pts")
        d_chg, t_chg = delta_pill(avg_monthly, df["MonthlyCharges"].mean(), "", "{:+.2f}", higher_is_bad=False)
        d_ten, t_ten = delta_pill(avg_tenure, df["tenure"].mean(), " mo", "{:+.1f}", higher_is_bad=False)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(kpi_card("Customers", f"{total:,}", f"{total / len(df) * 100:.1f}% of full cohort"),
                    unsafe_allow_html=True)
        k2.markdown(kpi_card("Churn rate", f"{churn_rate:.1f}%", f"{churned:,} churned, {retained:,} retained",
                             d_rate, t_rate, value_color=CRIMSON if churn_rate > 25 else TEAL),
                    unsafe_allow_html=True)
        k3.markdown(kpi_card("Monthly revenue lost", f"${lost_revenue:,.0f}", f"{lost_share:.1f}% of monthly billing",
                             value_color=CRIMSON), unsafe_allow_html=True)
        k4.markdown(kpi_card("Avg monthly charge", f"${avg_monthly:.2f}", f"Avg lifetime: ${fdf['TotalCharges'].mean():,.0f}",
                             d_chg, t_chg), unsafe_allow_html=True)
        k5.markdown(kpi_card("Avg tenure", f"{avg_tenure:.1f} mo", f"Median: {fdf['tenure'].median():.0f} months",
                             d_ten, t_ten), unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            section("Key takeaways", "Biggest gaps in churn rate across segments in your current selection (segments with 30+ customers).")
            insights = build_insights(fdf)
            if insights:
                for line in insights:
                    st.markdown(f"- {line}")
            else:
                st.caption("Not enough customers per segment to compare. Widen your filters.")

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                section("Retained vs churned", "Share of customers in the current selection.")
                counts = fdf["Churn"].map({"No": "Retained", "Yes": "Churned"}).value_counts().reset_index()
                counts.columns = ["Status", "Count"]
                fig = px.pie(counts, names="Status", values="Count", hole=0.62, color="Status",
                             color_discrete_map={"Retained": TEAL, "Churned": CRIMSON})
                fig.update_traces(textinfo="percent+label", textposition="outside",
                                  hovertemplate="<b>%{label}</b><br>%{value:,} customers (%{percent})<extra></extra>")
                fig.update_layout(showlegend=False,
                                  annotations=[dict(text=f"<b>{churn_rate:.1f}%</b><br>churned", x=0.5, y=0.5,
                                                    showarrow=False, font=dict(size=16, color=INK))])
                show_chart(style_fig(fig, 330))
        with c2:
            with st.container(border=True):
                section("Churn rate by tenure group", "New customers are the most likely to leave. Hover for counts.")
                ts = churn_by(fdf, "TenureGroup")
                fig = px.bar(ts, x="TenureGroup", y="churn_rate",
                             text=ts["churn_rate"].map(lambda v: f"{v:.1f}%"),
                             labels={"TenureGroup": "Tenure", "churn_rate": "Churn rate (%)"})
                fig.update_traces(marker_color=INDIGO, textposition="outside",
                                  customdata=np.stack((ts["churned"], ts["customers"]), axis=-1),
                                  hovertemplate="<b>%{x}</b><br>Churn rate: %{y:.1f}%<br>"
                                                "Churned: %{customdata[0]:,} of %{customdata[1]:,}<extra></extra>")
                fig.add_hline(y=overall_rate, line_dash="dash", line_color=SLATE,
                              annotation_text=f"All customers {overall_rate:.1f}%", annotation_position="top right")
                fig.update_yaxes(range=[0, max(ts["churn_rate"].max(), overall_rate) * 1.25])
                show_chart(style_fig(fig, 330))

        c3, c4 = st.columns(2)
        with c3:
            with st.container(border=True):
                section("Churn rate by month of tenure", "Smoothed with a 3-month rolling average.")
                tm = fdf.groupby("tenure").agg(rate=("ChurnFlag", "mean"), n=("ChurnFlag", "size")).reset_index()
                tm["rate"] = tm["rate"].rolling(3, min_periods=1, center=True).mean() * 100
                fig = go.Figure(go.Scatter(
                    x=tm["tenure"], y=tm["rate"], mode="lines", fill="tozeroy",
                    line=dict(color=CRIMSON, width=2.5), fillcolor="rgba(190,18,60,0.10)",
                    customdata=tm["n"],
                    hovertemplate="Month %{x}<br>Churn rate: %{y:.1f}%<br>Customers: %{customdata:,}<extra></extra>"))
                fig.update_xaxes(title="Tenure (months)")
                fig.update_yaxes(title="Churn rate (%)", rangemode="tozero")
                show_chart(style_fig(fig, 320))
        with c4:
            with st.container(border=True):
                section("Monthly charges, retained vs churned", "Churned customers skew toward higher bills.")
                fig = px.histogram(fdf, x="MonthlyCharges", color="Churn", nbins=30, barmode="overlay",
                                   opacity=0.7, color_discrete_map={"No": TEAL, "Yes": CRIMSON},
                                   labels={"MonthlyCharges": "Monthly charges ($)", "Churn": "Churned"})
                fig.update_yaxes(title="Customers")
                show_chart(style_fig(fig, 320, legend_bottom=True))

    # ======================================================================
    # TAB 2 - DRIVERS
    # ======================================================================
    with tab_drivers:
        dims = {k: v for k, v in DIMENSIONS.items() if v in fdf.columns}

        with st.container(border=True):
            section("Explore any segment", "Pick a customer attribute to compare churn across its groups.")
            ctl1, ctl2 = st.columns([2, 3])
            with ctl1:
                dim_label = st.selectbox("Compare by", list(dims.keys()), key="drv_dim")
            with ctl2:
                metric_label = st.radio("Show", ["Churn rate (%)", "Churned customers", "Customers"],
                                        horizontal=True, key="drv_metric")
            dim_col = dims[dim_label]
            metric_col = {"Churn rate (%)": "churn_rate", "Churned customers": "churned",
                          "Customers": "customers"}[metric_label]

            seg = churn_by(fdf, dim_col)
            seg["share"] = seg["customers"] / seg["customers"].sum() * 100
            seg = seg.sort_values(metric_col, ascending=True)
            seg["vs"] = np.where(seg["churn_rate"] > overall_rate, "Above all-customer rate", "At or below all-customer rate")

            g1, g2 = st.columns([3, 2])
            with g1:
                fig = px.bar(seg, x=metric_col, y=dim_col, orientation="h", color="vs",
                             color_discrete_map={"Above all-customer rate": CRIMSON,
                                                 "At or below all-customer rate": TEAL},
                             text=seg[metric_col].map(lambda v: f"{v:.1f}%" if metric_col == "churn_rate" else f"{v:,.0f}"),
                             labels={metric_col: metric_label, dim_col: dim_label, "vs": ""})
                fig.update_traces(textposition="outside",
                                  customdata=np.stack((seg["churned"], seg["customers"], seg["churn_rate"]), axis=-1),
                                  hovertemplate="<b>%{y}</b><br>Churn rate: %{customdata[2]:.1f}%<br>"
                                                "Churned: %{customdata[0]:,} of %{customdata[1]:,}<extra></extra>")
                if metric_col == "churn_rate":
                    fig.add_vline(x=overall_rate, line_dash="dash", line_color=SLATE,
                                  annotation_text=f"All {overall_rate:.1f}%", annotation_position="top")
                    fig.update_xaxes(range=[0, max(seg["churn_rate"].max(), overall_rate) * 1.2])
                else:
                    fig.update_xaxes(range=[0, seg[metric_col].max() * 1.2])
                fig.update_yaxes(title=None)
                show_chart(style_fig(fig, max(280, 70 * len(seg) + 80), legend_bottom=True))
            with g2:
                tbl = seg.sort_values("churn_rate", ascending=False)[[dim_col, "customers", "churned", "churn_rate", "share"]]
                tbl.columns = [dim_label, "Customers", "Churned", "Churn rate", "Share of customers"]
                st.dataframe(
                    tbl, hide_index=True, **STRETCH,
                    column_config={
                        "Churn rate": st.column_config.ProgressColumn("Churn rate", format="%.1f%%", min_value=0, max_value=100),
                        "Share of customers": st.column_config.NumberColumn("Share", format="%.1f%%"),
                    },
                )

        h1, h2 = st.columns([3, 2])
        with h1:
            with st.container(border=True):
                section("Where segments overlap", "Churn rate for every combination of two attributes. Small groups are hidden.")
                hc1, hc2, hc3 = st.columns(3)
                with hc1:
                    row_label = st.selectbox("Rows", list(dims.keys()), index=0, key="hm_row")
                with hc2:
                    col_label = st.selectbox("Columns", list(dims.keys()), index=3 if len(dims) > 3 else 1, key="hm_col")
                with hc3:
                    hm_min = st.number_input("Min customers per cell", 1, 500, 20, 5, key="hm_min")
                if row_label == col_label:
                    st.info("Choose two different attributes to see the heatmap.")
                else:
                    r, c = dims[row_label], dims[col_label]
                    rate = fdf.pivot_table(index=r, columns=c, values="ChurnFlag", aggfunc="mean", observed=True) * 100
                    cnt = fdf.pivot_table(index=r, columns=c, values="ChurnFlag", aggfunc="size", observed=True).reindex_like(rate)
                    rate = rate.where(cnt >= hm_min)
                    text = rate.map(lambda v: "" if pd.isna(v) else f"{v:.0f}%") if hasattr(rate, "map") else rate.applymap(lambda v: "" if pd.isna(v) else f"{v:.0f}%")
                    fig = go.Figure(go.Heatmap(
                        z=rate.values, x=[str(x) for x in rate.columns], y=[str(y) for y in rate.index],
                        text=text.values, texttemplate="%{text}", customdata=cnt.values,
                        colorscale=[[0, "#E6F5F2"], [0.5, "#FBE3C0"], [1, CRIMSON]],
                        zmin=0, zmax=max(60, float(np.nanmax(rate.values)) if rate.notna().any().any() else 60),
                        hovertemplate=f"{row_label}: %{{y}}<br>{col_label}: %{{x}}<br>"
                                      "Churn rate: %{z:.1f}%<br>Customers: %{customdata:,.0f}<extra></extra>",
                        colorbar=dict(title="Churn %", thickness=12)))
                    fig.update_yaxes(autorange="reversed")
                    show_chart(style_fig(fig, 340))
        with h2:
            with st.container(border=True):
                section("Do add-on services help?", "Churn rate for internet customers with vs without each service.")
                inet = fdf[fdf["InternetService"] != "No"]
                rows = []
                for s in SERVICE_COLS:
                    for val in ("Yes", "No"):
                        sub = inet[inet[s] == val]
                        if len(sub) >= 20:
                            rows.append({"Service": s, "Has service": val, "churn_rate": sub["ChurnFlag"].mean() * 100, "n": len(sub)})
                if rows:
                    sv = pd.DataFrame(rows)
                    fig = px.bar(sv, x="churn_rate", y="Service", color="Has service", barmode="group", orientation="h",
                                 color_discrete_map={"Yes": TEAL, "No": CRIMSON}, custom_data=["n"],
                                 labels={"churn_rate": "Churn rate (%)"})
                    fig.update_traces(hovertemplate="<b>%{y}</b><br>Churn rate: %{x:.1f}%<br>Customers: %{customdata[0]:,}<extra></extra>")
                    show_chart(style_fig(fig, 340, legend_bottom=True))
                else:
                    st.caption("Not enough internet customers in this selection.")

        with st.container(border=True):
            section("Highest-risk segments", "Combine attributes to find the groups that churn most. Sorted by churn rate.")
            sc1, sc2 = st.columns([3, 1])
            with sc1:
                combo_labels = st.multiselect("Group customers by", list(dims.keys()),
                                              default=["Contract type", "Internet service", "Tenure group"], key="seg_dims")
            with sc2:
                seg_min = st.slider("Min customers", 10, 300, 50, 10, key="seg_min")
            if combo_labels:
                cols = [dims[x] for x in combo_labels]
                combo = (fdf.groupby(cols, observed=True)
                         .agg(customers=("ChurnFlag", "size"), churned=("ChurnFlag", "sum"),
                              avg_monthly=("MonthlyCharges", "mean")).reset_index())
                combo["churn_rate"] = combo["churned"] / combo["customers"] * 100
                combo = combo[combo["customers"] >= seg_min].sort_values("churn_rate", ascending=False)
                combo = combo.rename(columns={v: k for k, v in dims.items()})
                if len(combo):
                    st.dataframe(
                        combo.head(25), hide_index=True, **STRETCH,
                        column_config={
                            "churn_rate": st.column_config.ProgressColumn("Churn rate", format="%.1f%%", min_value=0, max_value=100),
                            "avg_monthly": st.column_config.NumberColumn("Avg monthly", format="$%.2f"),
                            "customers": "Customers", "churned": "Churned",
                        },
                    )
                else:
                    st.caption("No segments reach that minimum size. Lower the minimum or group by fewer attributes.")
            else:
                st.caption("Select at least one attribute to group by.")

    # ======================================================================
    # TAB 3 - PREDICTOR
    # ======================================================================
    with tab_predict:
        if model is None:
            st.error(f"The prediction engine is unavailable: {model_status}")
            st.info("The analytics tabs still work without the model.")
        else:
            st.selectbox("Start from a profile", list(PRESETS.keys()), key="preset", on_change=apply_preset,
                         help="Loads typical settings. You can change any field afterward and the result updates instantly.")

            left, right = st.columns([3, 2], gap="large")
            with left:
                with st.container(border=True):
                    section("Customer profile")
                    a1, a2, a3, a4 = st.columns(4)
                    a1.selectbox("Gender", ["Female", "Male"], key="p_gender")
                    a2.selectbox("Senior citizen", ["No", "Yes"], key="p_senior")
                    a3.selectbox("Partner", ["Yes", "No"], key="p_partner")
                    a4.selectbox("Dependents", ["No", "Yes"], key="p_dependents")

                with st.container(border=True):
                    section("Services", "Add-ons are disabled when the customer has no internet or phone service.")
                    b1, b2 = st.columns(2)
                    b1.selectbox("Internet service", ["Fiber optic", "DSL", "No"], key="p_internet")
                    b2.selectbox("Phone service", ["Yes", "No"], key="p_phone")
                    no_net = st.session_state["p_internet"] == "No"
                    no_phone = st.session_state["p_phone"] == "No"
                    b3, b4, b5 = st.columns(3)
                    b3.selectbox("Multiple lines", ["No", "Yes"], key="p_multilines", disabled=no_phone)
                    b4.selectbox("Online security", ["No", "Yes"], key="p_security", disabled=no_net)
                    b5.selectbox("Online backup", ["No", "Yes"], key="p_backup", disabled=no_net)
                    b6, b7, b8, b9 = st.columns(4)
                    b6.selectbox("Device protection", ["No", "Yes"], key="p_device", disabled=no_net)
                    b7.selectbox("Tech support", ["No", "Yes"], key="p_tech", disabled=no_net)
                    b8.selectbox("Streaming TV", ["No", "Yes"], key="p_tv", disabled=no_net)
                    b9.selectbox("Streaming movies", ["No", "Yes"], key="p_movies", disabled=no_net)

                with st.container(border=True):
                    section("Contract and billing")
                    c1, c2, c3 = st.columns(3)
                    c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="p_contract")
                    c2.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                    "Bank transfer (automatic)", "Credit card (automatic)"], key="p_payment")
                    c3.selectbox("Paperless billing", ["Yes", "No"], key="p_paperless")
                    d1, d2 = st.columns(2)
                    d1.slider("Tenure (months)", 0, 72, key="p_tenure")
                    d2.slider("Monthly charges ($)", 18.0, 150.0, step=0.5, key="p_monthly")
                    st.checkbox("Estimate total charges from tenure and monthly charges", key="p_auto_total")
                    if st.session_state["p_auto_total"]:
                        st.caption(f"Total charges: ${st.session_state['p_tenure'] * st.session_state['p_monthly']:,.2f}")
                    else:
                        st.number_input("Total charges ($)", 0.0, 10000.0, step=10.0, key="p_total")

            customer = customer_from_state(st.session_state)
            input_df = pd.DataFrame([customer])
            try:
                prob = float(model.predict_proba(input_df)[:, 1][0])
                error = None
            except Exception as ex:
                prob, error = None, ex

            with right:
                if error is not None:
                    st.error(f"The model could not score this profile: {error}")
                else:
                    tier_label, tier_cls = risk_tier(prob, threshold)
                    with st.container(border=True):
                        st.markdown(
                            f"""<div class="risk-card {tier_cls}">
                            <h3>{tier_label}</h3>
                            <p>Predicted churn probability: <b>{prob * 100:.1f}%</b></p>
                            <p style="margin-top:4px;font-size:0.82rem;">Decision threshold: {threshold * 100:.1f}% (tuned for churn recall)</p>
                            </div>""", unsafe_allow_html=True)
                        bar_color = {"high": CRIMSON, "watch": AMBER, "low": TEAL}[tier_cls]
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 34}},
                            gauge={"axis": {"range": [0, 100]}, "bar": {"color": bar_color, "thickness": 0.3},
                                   "steps": [{"range": [0, threshold * 100], "color": "#E6F5F2"},
                                             {"range": [threshold * 100, 100], "color": "#FDECF0"}],
                                   "threshold": {"line": {"color": INK, "width": 3}, "thickness": 0.8,
                                                 "value": threshold * 100}}))
                        show_chart(style_fig(fig, 230))
                        flags = profile_flags(customer)
                        if flags:
                            st.markdown("**Profile flags** (patterns linked to churn in the data)")
                            st.markdown("".join(f'<span class="flag">{f}</span>' for f in flags), unsafe_allow_html=True)
                        else:
                            st.caption("No common churn-linked flags on this profile.")

                    # What-if scenarios
                    scen = retention_scenarios(customer)
                    with st.container(border=True):
                        section("What could lower this risk?", "Re-scores the customer with one change at a time.")
                        if scen:
                            rows = []
                            batch = pd.DataFrame([{**customer, **mods} for _, mods in scen])
                            probs = model.predict_proba(batch)[:, 1]
                            for (label, _), p in zip(scen, probs):
                                rows.append({"Action": label, "New probability": p * 100, "Change (pts)": (p - prob) * 100})
                            sdf = pd.DataFrame(rows).sort_values("Change (pts)")
                            fig = px.bar(sdf.sort_values("Change (pts)", ascending=False), x="Change (pts)", y="Action",
                                         orientation="h", text=sdf.sort_values("Change (pts)", ascending=False)["Change (pts)"].map(lambda v: f"{v:+.1f}"))
                            fig.update_traces(marker_color=[TEAL if v < 0 else CRIMSON for v in sdf.sort_values("Change (pts)", ascending=False)["Change (pts)"]],
                                              textposition="outside", cliponaxis=False,
                                              hovertemplate="<b>%{y}</b><br>Change: %{x:+.1f} pts<extra></extra>")
                            fig.update_yaxes(title=None)
                            fig.update_xaxes(title="Change in churn probability (points)")
                            show_chart(style_fig(fig, 60 * len(sdf) + 90))
                            best = sdf.iloc[0]
                            if best["Change (pts)"] < 0:
                                st.success(f"Best single action: **{best['Action']}**. "
                                           f"Risk drops from {prob * 100:.1f}% to {best['New probability']:.1f}%.")
                            else:
                                st.info("None of these single changes lowers the predicted risk for this profile.")
                        else:
                            st.caption("This profile already has the contract, payment and service options we test.")

                    out = input_df.assign(churn_probability=round(prob, 4), risk_tier=tier_label, threshold=threshold)
                    st.download_button("Download this prediction (CSV)", out.to_csv(index=False).encode(),
                                       file_name="churn_prediction.csv", mime="text/csv", **STRETCH)

    # ======================================================================
    # TAB 4 - MODELS
    # ======================================================================
    with tab_models:
        st.caption("Results from the machine learning notebook (`notebooks/02_machine_learning.ipynb`). "
                   "These are static validation results and do not change with the filters.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test accuracy", "79.86%")
        m2.metric("ROC-AUC", "0.8396")
        m3.metric("Churn recall at tuned threshold", "71.24%", delta="+19.36 pts vs 0.50 threshold")
        m4.metric("F1-score", "0.6265", help=f"Decision threshold {threshold:.4f}")

        mc1, mc2 = st.columns(2)
        with mc1:
            with st.container(border=True):
                section("Model comparison", "Test set, default 0.50 threshold.")
                if comparison_df is not None:
                    metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"] if c in comparison_df.columns]
                    name_col = "Model" if "Model" in comparison_df.columns else comparison_df.columns[0]
                    if metric_cols:
                        long = comparison_df.melt(id_vars=name_col, value_vars=metric_cols, var_name="Metric", value_name="Score")
                        fig = px.bar(long, x="Metric", y="Score", color=name_col, barmode="group",
                                     color_discrete_sequence=[INDIGO, TEAL, CRIMSON, AMBER, SLATE])
                        fig.update_traces(hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.2%}<extra></extra>")
                        fig.update_yaxes(tickformat=".0%", range=[0, 1])
                        show_chart(style_fig(fig, 320, legend_bottom=True))
                    disp = comparison_df.copy()
                    for c in metric_cols:
                        disp[c] = disp[c].map(lambda v: f"{v * 100:.2f}%")
                    st.dataframe(disp, hide_index=True, **STRETCH)
                else:
                    st.info("`model_comparison.csv` was not found in `data/output/`.")

            with st.container(border=True):
                section("5-fold cross-validation", "Average across stratified folds.")
                if cv_df is not None:
                    disp = cv_df.copy()
                    for c in disp.columns:
                        if c != "Model" and pd.api.types.is_numeric_dtype(disp[c]):
                            disp[c] = disp[c].map(lambda v: f"{v * 100:.2f}%")
                    st.dataframe(disp, hide_index=True, **STRETCH)
                else:
                    st.info("`cross_validation_results.csv` was not found in `data/output/`.")
        with mc2:
            with st.container(border=True):
                section("Top predictive features", "Random Forest importance.")
                if feat_df is not None:
                    top_n = st.slider("Features to show", 5, min(30, len(feat_df)), 12, key="feat_n")
                    top = feat_df.head(top_n).copy()
                    top["Feature name"] = top["Feature"].map(
                        lambda x: x.replace("num__", "").replace("cat__", "").replace("_", ": "))
                    fig = px.bar(top.sort_values("Importance"), x="Importance", y="Feature name", orientation="h",
                                 color="Importance", color_continuous_scale=[[0, "#C7D2FE"], [1, INDIGO]])
                    fig.update_layout(coloraxis_showscale=False)
                    fig.update_yaxes(title=None)
                    show_chart(style_fig(fig, max(320, 28 * top_n + 60)))
                else:
                    st.info("`random_forest_feature_importance.csv` was not found in `data/output/`.")

        with st.container(border=True):
            section("Why Gradient Boosting")
            st.markdown(
                "- It had the highest test accuracy (**79.86%**) and precision (**64.98%**), with an ROC-AUC of **0.8396**.\n"
                "- Missing churners is costly, so 5-fold cross-validation was used to tune the decision threshold to "
                f"**{threshold:.6f}**.\n"
                "- At that threshold, churn recall rises from **51.88%** to **71.24%**, catching nearly 3 in 4 churners, "
                "with an F1-score of **0.6265**."
            )

    # ======================================================================
    # TAB 5 - DATA EXPLORER
    # ======================================================================
    with tab_data:
        with st.container(border=True):
            section("Customer records", "Anonymized rows that match your sidebar filters.")
            e1, e2 = st.columns([2, 3])
            with e1:
                status = st.radio("Churn status", ["All", "Churned", "Retained"], horizontal=True, key="ex_status")
            with e2:
                default_cols = ["gender", "tenure", "Contract", "InternetService", "PaymentMethod",
                                "MonthlyCharges", "TotalCharges", "Churn"]
                shown = st.multiselect("Columns", [c for c in fdf.columns if c not in ("ChurnFlag", "SeniorGroup")],
                                       default=[c for c in default_cols if c in fdf.columns], key="ex_cols")
            view = fdf
            if status == "Churned":
                view = view[view["ChurnFlag"] == 1]
            elif status == "Retained":
                view = view[view["ChurnFlag"] == 0]
            if shown:
                st.dataframe(view[shown], hide_index=True, height=420, **STRETCH)
                st.download_button(f"Download {len(view):,} rows (CSV)", view[shown].to_csv(index=False).encode(),
                                   file_name="filtered_customers.csv", mime="text/csv")
            else:
                st.caption("Select at least one column to display.")


if __name__ == "__main__":
    main()