import argparse
import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get(
            "SM_MODEL_DIR",
            "data/output/sagemaker_model"
        )
    )

    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get(
            "SM_CHANNEL_TRAIN",
            "data/processed"
        )
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load training data
    # ---------------------------------------------------------
    train_file = os.path.join(
        args.train,
        "customer_churn_cleaned.csv"
    )

    print("Loading training data from:", train_file)

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Training file not found: {train_file}"
        )

    df = pd.read_csv(train_file)

    print("Dataset shape:", df.shape)

    # ---------------------------------------------------------
    # Prepare features and target
    # ---------------------------------------------------------
    target = "Churn"

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in dataset."
        )

    X = df.drop(columns=[target])

    y = df[target].map({
        "No": 0,
        "Yes": 1
    })

    if y.isnull().any():
        raise ValueError(
            "Unexpected values found in the Churn column."
        )

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "str"]
    ).columns.tolist()

    print("Numerical features:")
    print(numerical_features)

    print("Categorical features:")
    print(categorical_features)

    print(
        "Number of numerical features:",
        len(numerical_features)
    )

    print(
        "Number of categorical features:",
        len(categorical_features)
    )

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    # ---------------------------------------------------------
    # Final Gradient Boosting model
    # ---------------------------------------------------------
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            )
        ]
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------
    print("Training Gradient Boosting model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # ---------------------------------------------------------
    # Basic local evaluation
    # ---------------------------------------------------------
    accuracy = model.score(
        X_test,
        y_test
    )

    print(
        "Test accuracy:",
        round(accuracy, 4)
    )

    # ---------------------------------------------------------
    # Save model artifacts
    # ---------------------------------------------------------
    os.makedirs(
        args.model_dir,
        exist_ok=True
    )

    model_path = os.path.join(
        args.model_dir,
        "model.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    # Selected from the original model evaluation notebook
    selected_threshold = 0.322712

    threshold_path = os.path.join(
        args.model_dir,
        "threshold.joblib"
    )

    joblib.dump(
        selected_threshold,
        threshold_path
    )

    print("Model saved to:", model_path)
    print(
        "Threshold saved to:",
        threshold_path
    )
    print(
        "Selected threshold:",
        selected_threshold
    )


if __name__ == "__main__":
    main()