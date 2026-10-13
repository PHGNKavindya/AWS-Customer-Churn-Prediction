# Testing & Verification Checklist

## 1. Overview & Verification Policy

This document tracks all test suites, schema checks, and operational verifications performed for the **AWS Customer Churn Prediction** project. 

> [!IMPORTANT]
> **Strict Verification Policy:**
> Only tests that have been executed and passed in the active environment are marked as **PASSED**. Cloud tests that cannot be executed locally without live AWS credentials are explicitly marked as **UNVERIFIED / PENDING CLOUD RUN**.

---

## 2. Automated Test Suite Results (`tests/test_pipeline.py`)

* **Execution Command:** `python tests/test_pipeline.py`
* **Test Framework:** Python Standard Library `unittest`
* **Total Executed:** 9 Tests
* **Passed:** 9
* **Failed:** 0
* **Execution Duration:** ~4.0 seconds

| Test ID | Test Name | Description | Status | Verification Detail |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | `test_01_cleaned_dataset_exists_and_loads` | Verify `data/processed/customer_churn_cleaned.csv` loads. | **PASSED** | Verified file exists and loads exactly 7,021 rows. |
| **TC-02** | `test_02_required_schema_columns` | Check presence of all 20 required feature columns. | **PASSED** | Verified presence of all 15 categorical and 4 numerical features + `Churn`. |
| **TC-03** | `test_03_churn_target_integrity` | Validate target column `Churn` values and completeness. | **PASSED** | Target contains only `['No', 'Yes']`; 0 nulls; 5,164 retained, 1,857 churned. |
| **TC-04** | `test_04_dashboard_aggregation_math` | Validate mathematical KPI logic. | **PASSED** | Verified overall churn rate = 26.45%; average monthly charges = $64.82. |
| **TC-05** | `test_05_tenure_group_assignment` | Verify tenure grouping logic matches `sql/churn_analysis.sql`. | **PASSED** | Verified boundary conditions: `0-12 mo`, `13-24 mo`, `25-48 mo`, `49+ mo`. |
| **TC-06** | `test_06_empty_dataframe_handling` | Zero-division protection on empty filter queries. | **PASSED** | Tested empty DataFrame: returns 0% churn rate with no division error. |
| **TC-07** | `test_07_model_artifact_and_inference` | Model artifact unpickling and inference execution. | **PASSED** | Verified `final_churn_model.joblib` loads; threshold is `0.322712`; inference produces probabilities in `[0.0, 1.0]`. |
| **TC-08** | `test_08_no_pii_customer_identifiers` | Customer privacy and PII protection check. | **PASSED** | Confirmed zero customerID, names, addresses, or phone numbers in dataset. |
| **TC-09** | `test_09_evaluation_files_present` | Existence of model evaluation output CSVs. | **PASSED** | Verified presence of `model_comparison.csv`, `cross_validation_results.csv`, and `random_forest_feature_importance.csv`. |

---

## 3. Dashboard UI & Functional Verification (`app/dashboard.py`)

| Feature Area | Verification Action | Expected Behavior | Observed Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Startup** | Launch `streamlit run app/dashboard.py` | UI loads cleanly on `localhost:8501` without exceptions. | Dashboard renders with dark enterprise cards. | **PASSED** |
| **KPI Cards** | Load un-filtered cohort | Displays 7,021 customers, 1,857 churned, 26.45% rate. | Matches exact Athena SQL Query 1 outputs. | **PASSED** |
| **Sidebar Filters** | Change Contract to "Two year" | Metrics recalculate; churn drops to ~2.8%. | Recalculated accurately in real time. | **PASSED** |
| **Zero Match Filter** | Filter charges outside dataset range | Warning notice appears gracefully without crash. | "⚠️ No customer records match" notice shown. | **PASSED** |
| **Live Predictor** | Fill form and submit inference | Computes probability, renders gauge chart, shows risk. | Displayed gauge chart, classification, and advice. | **PASSED** |
| **Benchmarks Tab** | Navigate to Tab 4 | Displays baseline comparison table and feature chart. | Tables formatted with percentages; top features shown. | **PASSED** |

---

## 4. AWS Cloud Integration Boundaries (Planned / Pending Teammate Confirmation)

| Cloud Boundary | Target AWS Resource | Owner | Verification Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **S3 Storage** | `s3://customer-churn-analytics-ash1010-2026/` | Member 2 | **PENDING CLOUD RUN** | Bucket referenced in script; upload pending cloud credentials. |
| **Glue Data Catalog** | `customer_churn_db.customer_churn` | Member 2 | **PENDING CLOUD RUN** | DDL provided in `sql/create_customer_churn_table.sql`. |
| **Amazon Athena** | SQL Query Execution | Member 2 | **VERIFIED LOCALLY / PENDING CLOUD RUN** | Queries verified locally; execution in Athena pending AWS session. |
| **SageMaker Job** | `customer-churn-gradient-boosting` | Member 3 | **PENDING CLOUD RUN** | `sagemaker/launch_training.py` ready for execution in AWS account. |
| **Amazon QuickSight**| Cloud BI Dashboard | Member 4 / Team | **PENDING CLOUD RUN** | QuickSight deployment requires active subscription and Athena table. |
