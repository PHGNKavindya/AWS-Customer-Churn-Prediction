# Dashboard User & Operations Guide

## 1. Overview

The **Interactive Customer Churn Analytics Dashboard** (`app/dashboard.py`) is Member 4's primary visualization deliverable. Built using **Streamlit** and **Plotly**, it enables evaluators, business analysts, and retention teams to explore customer churn patterns, test interactive machine learning scenarios, and inspect model evaluation metrics.

### Key Architecture Attributes:
* **100% Local & Offline Capable:** Does not require AWS cloud credentials, billing, or active internet connectivity.
* **Direct Alignment with SQL Queries:** Visualizations directly map to the queries executed in Amazon Athena (`sql/churn_analysis.sql`).
* **Privacy Preserving:** Processes anonymized customer feature records without exposing PII.

---

## 2. Launching the Dashboard

### Prerequisites
Ensure your virtual environment is active and dependencies are installed:
```bash
# 1. Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 2. Verify dependencies
pip install -r requirements.txt

# 3. Launch Dashboard
streamlit run app/dashboard.py
```

Upon launch, Streamlit will open your default browser at:
`http://localhost:8501`

---

## 3. Dashboard Structure & Features

The dashboard is structured into **four dedicated tabs** accessible from the main navigation:

### Tab 1: Executive KPIs & Overview
* **Top Metric Cards:**
  * **Total Customers:** Filtered customer count and proportion of the total 7,021 cohort.
  * **Churned Customers:** Number of lost customers vs. retained customers.
  * **Overall Churn Rate:** Percentage of customers churned (`churned / total * 100`). Color-coded (red if > 25%, green if lower).
  * **Average Monthly Charges:** Mean monthly charge across the cohort.
  * **Average Tenure:** Average customer lifespan in months alongside median tenure.
* **Visualizations:**
  * **Retention vs. Churn Donut Chart:** Visual breakdown of retained (green) vs. churned (red) customers.
  * **Churn Rate by Tenure Group Bar Chart:** Displays how customer longevity impacts churn across brackets (`0-12 months`, `13-24 months`, `25-48 months`, `49+ months`).

### Tab 2: Churn Drivers & Segment Analysis
Directly reflects the Athena analytical SQL queries in `sql/churn_analysis.sql`:
* **Contract Commitment:** Compares churn rates across Month-to-month, One-year, and Two-year contracts (demonstrates that Month-to-month contracts have ~42.7% churn vs. <3% for Two-year contracts).
* **Payment Method:** Analyzes churn across Electronic check, Mailed check, Bank transfer, and Credit card (highlights Electronic check as the highest-risk payment channel).
* **Internet Service Architecture:** Compares Fiber optic, DSL, and No Internet service (Fiber optic customers exhibit the highest churn propensity).
* **Monthly Charges Distribution:** Interactive box plot highlighting that churned customers have significantly higher median monthly charges (~$80/mo vs ~$65/mo for retained customers).

### Tab 3: Live ML Churn Risk Predictor
An interactive scenario simulator that runs Member 1's trained Gradient Boosting pipeline (`final_churn_model.joblib`):
1. **Input Customer Profile:**
   * Demographics: Gender, Senior Citizen, Partner, Dependents.
   * Services: Phone, Multiple Lines, Internet Service, Security, Backup, Device Protection, Tech Support, Streaming.
   * Billing & Contract: Contract Type, Paperless Billing, Payment Method, Tenure, Monthly Charges, Total Charges.
2. **Click "Run Model Inference":**
   * The model computes the churn probability via `model.predict_proba()`.
   * Evaluates against the calibrated threshold (**0.322712**).
   * Displays an intuitive **Risk Gauge Chart** and classification badge (**HIGH CHURN RISK** vs **LOW CHURN RISK**).
   * Generates actionable, context-aware retention advice.

### Tab 4: Model Benchmarks & Feature Importance
Presents Member 1's verified model validation results:
* **Baseline Test-Set Comparison Table:** Compares Accuracy, Precision, Recall, F1-Score, and ROC-AUC across Logistic Regression, Random Forest, Gradient Boosting, and SVM.
* **5-Fold Stratified Cross-Validation Table:** Validates cross-fold generalization.
* **Feature Importance Chart:** Bar chart illustrating the top 12 predictive features from Random Forest (led by TotalCharges, tenure, MonthlyCharges, and Month-to-month contract).
* **Threshold Tuning Explanation:** Documents why the 0.322712 threshold was chosen to boost recall from 51.88% to 71.24%.

---

## 4. Sidebar Controls & Dynamic Filtering

The left sidebar provides interactive cohort filters:
* **Contract Type:** Filter by Month-to-month, One year, Two year, or All.
* **Internet Service:** Filter by Fiber optic, DSL, No, or All.
* **Payment Method:** Filter by payment channel.
* **Tenure Group:** Select specific tenure brackets.
* **Monthly Charges Slider:** Filter by minimum and maximum monthly billing amounts.

> [!NOTE]
> All metric cards and analytical charts dynamically recalculate based on the active sidebar filter selection. If a filter combination results in 0 matching records, a graceful warning prompt is shown without application crash.

---

## 5. Troubleshooting & FAQ

* **Port 8501 is already in use:**
  Run on a custom port:
  ```bash
  streamlit run app/dashboard.py --server.port 8502
  ```
* **"Cleaned dataset not found" error:**
  Verify that `data/processed/customer_churn_cleaned.csv` is present in the workspace.
* **Model unpickling warning:**
  Scikit-learn may emit a standard version compatibility warning if your local scikit-learn version differs slightly from the training environment. This is benign — our automated tests verify that inference executes with 100% mathematical validity.
