# AWS Deployment & Cloud Integration Guide

## 1. Important Notice & Cost Safety

> [!CAUTION]
> **Zero Unapproved Cloud Charges:**
> Do NOT create or run billable cloud resources (SageMaker training jobs, hosted endpoints, NAT Gateways, or QuickSight Enterprise subscriptions) without explicit approval from your project supervisor. This guide provides exact, reproducible steps for your teammates (Members 2 & 3) to execute when cloud deployment is formally authorized.

---

## 2. Architecture Prerequisites & Setup Checklist

Before initiating any AWS deployment steps, ensure the following are available:
1. **AWS Account Access:** An active AWS account with permissions in region `ap-southeast-2` (Asia Pacific - Sydney) or your preferred region.
2. **AWS CLI v2:** Installed and authenticated locally using `aws configure` (avoid embedding credentials into project files).
3. **Python Environment:** Python 3.11 with `boto3` and `sagemaker` SDKs installed.

---

## 3. Least-Privilege IAM Roles & Permissions

Never use `AdministratorAccess` policies for service execution. Use the following targeted IAM role definitions:

### Role 1: SageMaker Execution Role (`CustomerChurnSageMakerExecutionRole`)
* **Trusted Entity:** `sagemaker.amazonaws.com`
* **Attached Policies:**
  * `AmazonSageMakerFullAccess` (or custom scoped policy to SageMaker actions)
  * Custom Inline Policy for S3 Data Access:
    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Sid": "S3ScopedAccess",
          "Effect": "Allow",
          "Action": [
            "s3:GetObject",
            "s3:PutObject",
            "s3:ListBucket"
          ],
          "Resource": [
            "arn:aws:s3:::customer-churn-analytics-ash1010-2026",
            "arn:aws:s3:::customer-churn-analytics-ash1010-2026/*"
          ]
        }
      ]
    }
    ```

### Role 2: AWS Glue Crawler Role (`AWSGlueServiceRole-CustomerChurn`)
* **Trusted Entity:** `glue.amazonaws.com`
* **Attached Policies:**
  * `AWSGlueServiceRole`
  * Scoped read access to `s3://customer-churn-analytics-ash1010-2026/processed/*`

---

## 4. Amazon S3 Bucket Architecture

Organize the target S3 bucket (`customer-churn-analytics-ash1010-2026`) into dedicated prefixes:

```
s3://customer-churn-analytics-ash1010-2026/
├── raw/                              # Original raw dataset
├── processed/                        # Cleaned CSV (customer_churn_cleaned.csv)
├── model/                            # SageMaker training outputs (model.tar.gz)
├── predictions/                      # Batch inference scored outputs
└── athena-query-results/             # Athena query output spill location
```

### Upload Command (Execute only when authorized):
```bash
aws s3 cp data/processed/customer_churn_cleaned.csv s3://customer-churn-analytics-ash1010-2026/processed/
```

---

## 5. AWS Glue & Amazon Athena Integration (Member 2 Responsibility)

### Step 1: Create Database in Athena
Open the Amazon Athena query editor and execute:
```sql
CREATE DATABASE IF NOT EXISTS customer_churn_db;
```

### Step 2: Create External Table
Execute the script from `sql/create_customer_churn_table.sql`:
```sql
CREATE EXTERNAL TABLE IF NOT EXISTS customer_churn_db.customer_churn (
    gender STRING,
    seniorcitizen INT,
    partner STRING,
    dependents STRING,
    tenure INT,
    phoneservice STRING,
    multiplelines STRING,
    internetservice STRING,
    onlinesecurity STRING,
    onlinebackup STRING,
    deviceprotection STRING,
    techsupport STRING,
    streamingtv STRING,
    streamingmovies STRING,
    contract STRING,
    paperlessbilling STRING,
    paymentmethod STRING,
    monthlycharges DOUBLE,
    totalcharges DOUBLE,
    churn STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
    'separatorChar' = ',',
    'quoteChar'     = '\"',
    'escapeChar'    = '\\'
)
STORED AS TEXTFILE
LOCATION 's3://customer-churn-analytics-ash1010-2026/processed/'
TBLPROPERTIES (
    'skip.header.line.count' = '1'
);
```

### Step 3: Run Analytical Queries
Execute the queries in `sql/churn_analysis.sql` to verify table population and confirm baseline metrics:
* Query 1: Total customers (7,021), Churned (1,857), Churn Rate (26.45%).
* Query 2: Churn by contract (Month-to-month: ~42.7%, Two-year: ~2.8%).

---

## 6. AWS SageMaker Model Training & Inference (Member 3 Responsibility)

### Step 1: Launch SageMaker Training Job
Once data resides in `s3://customer-churn-analytics-ash1010-2026/processed/`:
```bash
python sagemaker/launch_training.py
```
This triggers an `SKLearn` estimator job using container version `1.4-2` on an `ml.m5.xlarge` instance. Model artifacts will be written to `s3://customer-churn-analytics-ash1010-2026/model/`.

### Step 2: Model Inference Compatibility Note
* **Cloud Execution:** The inference handler in `sagemaker/inference.py` runs inside AWS SageMaker scikit-learn container `1.4-2` where `model.joblib` loads without version friction.
* **Local Execution:** If testing `inference.py` locally in modern Python 3.11 environments running `scikit-learn >= 1.6`, use Member 1's local artifact `data/output/final_churn_model.joblib` to prevent Cython loss module discrepancies.

---

## 7. Amazon QuickSight BI Dashboard Setup (Planned)

When QuickSight enterprise subscription and permissions are active:
1. **Grant Athena & S3 Permissions:** In QuickSight > Manage QuickSight > Security & Permissions, grant QuickSight access to Athena and the S3 bucket `customer-churn-analytics-ash1010-2026`.
2. **Create Dataset:**
   * Data Source: Amazon Athena
   * Database: `customer_churn_db`
   * Table: `customer_churn`
   * Import mode: Direct Query or SPICE (SPICE recommended for faster interactive rendering).
3. **Build Dashboard Visuals:**
   * **KPIs:** Total Customers (`COUNT(gender)`), Churned Customers (`SUM(ifelse(churn = 'Yes', 1, 0))`), Churn Rate.
   * **Donut Chart:** Churn vs Retained breakdown.
   * **Bar Chart 1:** Churn Rate by Contract.
   * **Bar Chart 2:** Churn Rate by Payment Method.
   * **Bar Chart 3:** Churn Rate by Internet Service.
   * **Box Plot / Histogram:** Monthly charges distribution by churn status.

---

## 8. Troubleshooting Common Cloud Integration Errors

| Error Symptom | Root Cause | Remediation Step |
| :--- | :--- | :--- |
| **Athena: HIVE_CURSOR_ERROR / Cannot cast header** | `skip.header.line.count` missing in table properties | Ensure `TBLPROPERTIES ('skip.header.line.count' = '1')` is added to DDL. |
| **Athena: Query Results S3 Location Error** | Athena output bucket not configured in query editor settings | Set query result location to `s3://your-bucket/athena-query-results/` in Athena Settings. |
| **SageMaker: AccessDenied during Training** | IAM Role lacks S3 read/write permissions for bucket | Verify `CustomerChurnSageMakerExecutionRole` has policies allowing GetObject/PutObject on the S3 bucket. |
| **QuickSight: Table Not Found in Athena** | QuickSight IAM role lacks Athena workgroup or S3 access | Enable QuickSight access to Athena and S3 bucket in QuickSight Security Settings. |

---

## 9. Cloud Resource Cleanup (Cost Prevention)

After completing university demonstrations or evaluations, ensure all billable resources are terminated:
1. **Delete SageMaker Endpoints:**
   ```bash
   aws sagemaker delete-endpoint --endpoint-name <endpoint-name>
   aws sagemaker delete-endpoint-config --endpoint-config-name <config-name>
   ```
2. **Delete QuickSight SPICE Datasets & Cancel Subscriptions:**
   Delete unused SPICE datasets to avoid monthly per-GB storage fees. Cancel QuickSight subscription if not needed further.
3. **Empty & Delete S3 Buckets (If project complete):**
   ```bash
   aws s3 rm s3://customer-churn-analytics-ash1010-2026 --recursive
   ```
