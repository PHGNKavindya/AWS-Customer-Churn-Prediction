# Team Responsibilities & Member Contributions

## 1. Project Role Allocation

The **AWS Customer Churn Prediction** project was structured into four distinct functional specializations to cover the complete lifecycle from data ingestion to cloud integration, visualization, and quality assurance.

| Team Role | Member Responsibility | Core Work Products | Execution Boundary |
| :--- | :--- | :--- | :--- |
| **Member 1** | **Data Science & ML Modeling** | • `data/raw/IT_customer_churn.csv`<br>• `notebooks/01_data_exploration.ipynb`<br>• `data/processed/customer_churn_cleaned.csv`<br>• `notebooks/02_machine_learning.ipynb`<br>• `data/output/final_churn_model.joblib`<br>• `data/output/final_churn_threshold.joblib`<br>• Model comparison & feature importance CSVs | Verified Complete (Local) |
| **Member 2** | **AWS Data Engineering** | • `sql/churn_analysis.sql`<br>• Amazon S3 bucket architecture (`customer-churn-analytics-...`)<br>• AWS Glue Data Catalog & Crawler configuration<br>• Amazon Athena database & SQL analytical execution | Scripts Verified; Cloud Deployment Planned |
| **Member 3** | **AWS ML Deployment** | • `sagemaker/train.py`<br>• `sagemaker/launch_training.py`<br>• `sagemaker/inference.py`<br>• SageMaker training job configuration (`ml.m5.xlarge`, SKLearn `1.4-2`)<br>• Planned SageMaker Endpoint / Batch Transform | Scripts Verified; Cloud Deployment Planned |
| **Member 4**<br>*(Current Author)* | **Dashboard, Integration, Architecture, Testing & Demo** | • `app/dashboard.py` (Streamlit + Plotly interactive UI)<br>• `sql/create_customer_churn_table.sql` (Athena DDL)<br>• `tests/test_pipeline.py` (9 automated test cases)<br>• `requirements.txt` & `.env.example`<br>• Complete documentation suite in `docs/`<br>• Root `README.md` comprehensive upgrade<br>• Final 5–7 minute demonstration script | Verified Complete (Local) |

---

## 2. Detailed Breakdown of Member 4 Contributions

As **Member 4**, my contributions focused on bridging data science and cloud engineering into an accessible, validated, and demonstrable business product:

### 1. Interactive Analytics & Prediction Dashboard (`app/dashboard.py`)
* Designed and built an enterprise-grade Streamlit application featuring four comprehensive tabs:
  1. **Executive KPIs & Overview:** Dynamic metric cards (Total Customers, Churned, Churn Rate, Charges, Tenure) and retention donut chart.
  2. **Churn Drivers & Segment Analysis:** Visualizations directly mirroring Athena SQL queries (Contract type, Payment method, Internet service, Tenure group, Monthly charges box plot).
  3. **Live ML Churn Risk Predictor:** Scenario simulator allowing evaluators to input customer parameters, invoke Member 1's saved Gradient Boosting pipeline, view probability gauges against the calibrated `0.322712` threshold, and receive retention advice.
  4. **Model Benchmarks & Feature Importance:** Offline model comparison table and feature ranking chart.
* Engineered custom CSS with accessible dark-mode styling, responsive layouts, and zero-division error handling for empty filter queries.
* Implemented privacy-first data handling with zero exposure of PII.

### 2. End-to-End Architecture & Cloud Integration Specification
* Authored `docs/architecture.md` featuring a detailed Mermaid architecture diagram that clearly distinguishes verified local components from planned AWS cloud services.
* Created `sql/create_customer_churn_table.sql` providing the exact Athena `OpenCSVSerde` external table definition required by Member 2.
* Authored `docs/aws-deployment-guide.md` specifying least-privilege IAM policies, S3 folder structure, Athena validation steps, SageMaker training execution, and cloud cost cleanup instructions.

### 3. Automated Testing & Verification Suite (`tests/test_pipeline.py`)
* Created a complete test suite with 9 unit and integration tests executing under standard Python `unittest`.
* Tested dataset presence, schema completeness (all 20 columns), target distribution (7,021 rows, 26.45% churn rate), KPI aggregation math, tenure grouping boundary conditions, zero-division resilience, model inference accuracy, PII absence, and output CSV availability.
* Achieved 100% pass rate in automated testing.

### 4. Technical Documentation & Demonstration Material
* Revamped root `README.md` from a 9-line placeholder into a comprehensive capstone reference with architecture flow, setup commands, model benchmark metrics, and execution steps.
* Authored `docs/dashboard-guide.md`, `docs/testing-checklist.md`, `docs/demo-script.md`, and `.env.example`.
* Created a 5–7 minute timed demonstration script with structured talking points and commands.

---

## 3. Acknowledgement of Teammates' Contributions

* All exploratory data analysis, data cleaning logic, baseline model benchmarking, hyperparameter selection, threshold optimization, and feature importance rankings were conceived and executed by **Member 1**.
* All Athena SQL queries and cloud storage naming conventions were authored by **Member 2**.
* All SageMaker containerization entrypoints (`train.py`, `launch_training.py`, `inference.py`) were authored by **Member 3**.
* Member 4's work directly integrates these upstream contributions without modifying original model weights, threshold values, or training pipelines.
