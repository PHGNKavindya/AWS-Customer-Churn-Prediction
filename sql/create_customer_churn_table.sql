-- ============================================================================
-- Amazon Athena DDL: Customer Churn External Table Definition
-- Project: AWS Customer Churn Prediction
-- Description: Maps the cleaned customer churn CSV dataset residing in Amazon S3
--              to an Athena/Glue Data Catalog table.
-- Note: Replace '<your-bucket-name>' with your actual S3 bucket name.
-- ============================================================================

CREATE EXTERNAL TABLE IF NOT EXISTS customer_churn (
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
LOCATION 's3://<your-bucket-name>/processed/'
TBLPROPERTIES (
    'skip.header.line.count' = '1'
);
