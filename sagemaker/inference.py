import os
import json
import joblib
import pandas as pd


def model_fn(model_dir):
    """
    Load the trained model and threshold from the model directory.
    SageMaker calls this function when the endpoint container starts.
    """

    model_path = os.path.join(
        model_dir,
        "model.joblib"
    )

    threshold_path = os.path.join(
        model_dir,
        "threshold.joblib"
    )

    model = joblib.load(
        model_path
    )

    threshold = joblib.load(
        threshold_path
    )

    return {
        "model": model,
        "threshold": threshold
    }


def input_fn(request_body, request_content_type):
    """
    Convert incoming JSON into a Pandas DataFrame.
    """

    if request_content_type == "application/json":

        data = json.loads(
            request_body
        )

        # Support one customer as a dictionary
        if isinstance(data, dict):
            data = [data]

        # Support multiple customers as a list
        if not isinstance(data, list):
            raise ValueError(
                "JSON input must be an object or a list of objects."
            )

        return pd.DataFrame(data)

    raise ValueError(
        f"Unsupported content type: {request_content_type}"
    )


def predict_fn(input_data, model_data):
    """
    Generate churn probabilities and apply the selected threshold.
    """

    model = model_data["model"]
    threshold = model_data["threshold"]

    probabilities = model.predict_proba(
        input_data
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results = []

    for prediction, probability in zip(
        predictions,
        probabilities
    ):
        results.append({
            "churn_prediction":
                "Yes"
                if int(prediction) == 1
                else "No",

            "churn_probability":
                float(probability),

            "threshold":
                float(threshold)
        })

    return results


def output_fn(prediction, response_content_type):
    """
    Convert prediction output to JSON.
    """

    if response_content_type == "application/json":
        return json.dumps(
            prediction
        )

    raise ValueError(
        f"Unsupported response type: {response_content_type}"
    )


# ---------------------------------------------------------
# Optional local test
# ---------------------------------------------------------
if __name__ == "__main__":

    local_model_dir = os.path.join(
        "data",
        "output",
        "sagemaker_model"
    )

    if not os.path.exists(local_model_dir):
        print(
            "Model directory not found."
        )
        print(
            "Run this first:"
        )
        print(
            "python sagemaker/train.py"
        )
        raise SystemExit(1)

    model_data = model_fn(
        local_model_dir
    )

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    request_body = json.dumps(
        sample_customer
    )

    input_data = input_fn(
        request_body,
        "application/json"
    )

    prediction = predict_fn(
        input_data,
        model_data
    )

    response = output_fn(
        prediction,
        "application/json"
    )

    print("Local test prediction:")
    print(response)