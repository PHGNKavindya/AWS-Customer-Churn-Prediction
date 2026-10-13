# 5–7 Minute Project Demonstration Script

**Project Title:** AWS Customer Churn Prediction  
**Presenter:** Member 4 (Dashboard, Architecture/Integration, Documentation, Testing, & Demo)  
**Total Target Time:** 6 Minutes (with 1 minute buffer for Q&A)

---

## Demonstration Setup (Before Presentation Starts)
1. Terminal 1 open at repository root with virtual environment activated.
2. Terminal 2 ready to execute automated tests.
3. Browser open at `http://localhost:8501` (or ready to launch).

---

## Minute 1: Introduction & Project Scope (0:00 – 1:00)

* **Talking Points:**
  * "Good morning everyone. Today our team is presenting the **AWS Customer Churn Prediction** system."
  * "Customer churn is a critical challenge in telecommunications. Acquiring a new customer costs 5 to 7 times more than retaining an existing one. Our objective is twofold: analyze the primary drivers of churn using scalable cloud analytics, and deploy a machine learning pipeline to proactively identify at-risk customers before they cancel service."
  * "Our four-member team divided responsibilities across the lifecycle:
    * Member 1 led Data Exploration and Machine Learning.
    * Member 2 led AWS Data Engineering with S3, Glue, and Athena.
    * Member 3 led Cloud Machine Learning with Amazon SageMaker.
    * I am **Member 4**, responsible for the interactive analytics dashboard, architecture integration, documentation, automated testing, and this final demonstration."

---

## Minute 2: Data Engineering & ML Foundations (1:00 – 2:15)

* **Talking Points:**
  * "Let's examine the data foundation created by Member 1. We started with the IBM Telco Customer Churn dataset of 7,043 records."
  * "During data preparation in `01_data_exploration.ipynb`, whitespace anomalies in `TotalCharges` were resolved, missing values for new customers were imputed, and 22 exact duplicate records were eliminated, yielding a high-integrity cohort of **7,021 customers** with a **26.45% baseline churn rate**."
  * "In `02_machine_learning.ipynb`, four algorithms were benchmarked: Logistic Regression, Random Forest, SVM, and Gradient Boosting."
  * "**Gradient Boosting** demonstrated superior performance with **79.86% accuracy** and an **ROC-AUC of 0.8396**."
  * "Crucially, a standard 0.50 probability cutoff caught only 51.8% of churners. Through 5-fold cross-validation precision-recall optimization, our team calibrated the decision threshold to **`0.3227`**. This boosted recall to **71.24%**—allowing retention teams to capture nearly 3 out of 4 churning customers."

---

## Minute 3: Architecture & AWS Cloud Integration (2:15 – 3:15)

* **Visual Aid:** Refer to `docs/architecture.md` diagram.
* **Talking Points:**
  * "Our architecture follows modern cloud-native best practices across two complementary pipelines:
    1. **Data Analytics Pipeline:** Cleaned customer data is staged in Amazon S3, cataloged via AWS Glue, and queried serverlessly through Amazon Athena using the 6 analytical queries in `sql/churn_analysis.sql`."
    2. **Machine Learning Pipeline:** AWS SageMaker trains our Scikit-Learn Gradient Boosting pipeline and generates probability outputs."
  * "To ensure our demonstration is 100% reproducible today without requiring billable cloud credits, I built a local interactive Streamlit and Plotly dashboard directly connected to our verified data and model artifacts."

---

## Minute 4: Interactive Dashboard Demonstration (3:15 – 4:30)

* **Action:** Switch to browser showing `http://localhost:8501`.
* **Talking Points:**
  * "Here is our interactive dashboard. At the top, you immediately see our executive KPI cards: **7,021 Total Customers**, **1,857 Churned**, and an **Overall Churn Rate of 26.45%** with average monthly billing of **$64.82**."
  * "Under Tab 1, our donut chart shows retention balance, and our tenure bracket chart shows that customer churn is heavily front-loaded: customers in their first 12 months churn at **47.5%**, dropping dramatically to under **10%** after 48 months."
  * *Action: Click on Sidebar Filter 'Contract Type' -> Select 'Month-to-month'.*
  * "Notice how dynamically filtering to Month-to-month contracts causes the churn rate to spike to **42.7%**, while filtering to Two-year contracts drops churn to under **3%**."
  * *Action: Click Tab 2 'Churn Drivers & Segment Analysis'.*
  * "In Tab 2, we see exact visual representations of Member 2's Athena SQL queries: Electronic check is by far the riskiest payment method, Fiber optic customers churn at almost double the rate of DSL, and churned customers carry significantly higher monthly bills."

---

## Minute 5: Live ML Churn Risk Predictor (4:30 – 5:30)

* **Action:** Click Tab 3 'Live ML Churn Risk Predictor'.
* **Talking Points:**
  * "Now let's examine the interactive machine learning predictor powered by Member 1's saved Gradient Boosting pipeline and our calibrated threshold."
  * "Let's simulate a high-risk scenario: a new customer with Month-to-month contract, Fiber optic internet, Electronic check payment, tenure of only 3 months, and $85 monthly charge."
  * *Action: Click 'Run Model Inference'.*
  * "Instantly, the model calculates a **~75% churn probability**. The gauge chart lights up in red with a **HIGH CHURN RISK** alert and recommends targeted proactive retention incentives."
  * *Action: Change Contract to 'Two year', Tenure to 36 months, and click 'Run Model Inference'.*
  * "Changing contract to Two-year immediately drops the predicted probability below our 32.27% threshold, switching the classification to **LOW RISK**."

---

## Minute 6: Automated Testing, Quality Assurance, & Wrap-Up (5:30 – 6:30)

* **Action:** Switch to terminal and run:
  ```bash
  python tests/test_pipeline.py
  ```
* **Talking Points:**
  * "To ensure software quality and academic reproducibility, I developed an automated test suite with 9 rigorous tests."
  * "As you can see on the screen, all 9 tests run and pass in 4 seconds: validating data loading, schema completeness, target label distributions, KPI aggregation math, zero-division resilience, model inference accuracy, and customer privacy compliance."
  * "In summary: our team has delivered a complete end-to-end churn prediction solution with robust data preparation, validated machine learning models, cloud architecture designs, an interactive analytics dashboard, and automated test coverage."
  * "Thank you, and I am happy to take any questions!"
