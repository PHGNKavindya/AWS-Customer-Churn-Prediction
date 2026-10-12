import boto3
import sagemaker

from sagemaker.sklearn.estimator import SKLearn

REGION = "ap-southeast-2"

ROLE_ARN = (
    "arn:aws:iam::906915241255:"
    "role/CustomerChurnSageMakerExecutionRole"
)

BUCKET = "customer-churn-analytics-ash1010-2026"

TRAINING_DATA = (
    f"s3://{BUCKET}/processed/"
)

OUTPUT_PATH = (
    f"s3://{BUCKET}/model/"
)

boto_session = boto3.Session(
    region_name=REGION
)

sagemaker_session = sagemaker.Session(
    boto_session=boto_session
)

estimator = SKLearn(
    entry_point="train.py",
    source_dir="sagemaker",

    role=ROLE_ARN,

    framework_version="1.4-2",

    instance_type="ml.m5.xlarge",
    instance_count=1,

    output_path=OUTPUT_PATH,

    sagemaker_session=sagemaker_session,

    base_job_name="customer-churn-gradient-boosting"
)

print("Starting SageMaker training job...")

print(
    "Training input:",
    TRAINING_DATA
)

print(
    "Model output:",
    OUTPUT_PATH
)


estimator.fit(
    {
        "train": TRAINING_DATA
    },
    wait=True,
    logs=True
)


print("\nTraining completed successfully.")

print(
    "Model artifact:",
    estimator.model_data
)