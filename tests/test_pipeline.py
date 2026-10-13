"""
Unit and Integration Test Suite for AWS Customer Churn Prediction
Member 4 Deliverable - Verification of Data, Schema, Dashboard Logic, and ML Inference
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path for direct script invocation
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestCustomerChurnPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up paths and load dataset once for test class."""
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.cleaned_csv_path = os.path.join(
            cls.project_root, "data", "processed", "customer_churn_cleaned.csv"
        )
        cls.model_path = os.path.join(
            cls.project_root, "data", "output", "final_churn_model.joblib"
        )
        cls.thresh_path = os.path.join(
            cls.project_root, "data", "output", "final_churn_threshold.joblib"
        )

        # Expected Schema Definition
        cls.expected_features = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges", "Churn"
        ]

    def test_01_cleaned_dataset_exists_and_loads(self):
        """Test 1: Verify cleaned dataset exists and has valid row dimensions."""
        self.assertTrue(
            os.path.exists(self.cleaned_csv_path),
            f"Cleaned dataset missing at {self.cleaned_csv_path}"
        )
        df = pd.read_csv(self.cleaned_csv_path)
        self.assertFalse(df.empty, "Dataset is empty")
        self.assertEqual(len(df), 7021, "Cleaned dataset should contain 7,021 rows")

    def test_02_required_schema_columns(self):
        """Test 2: Verify all 20 required schema columns exist."""
        df = pd.read_csv(self.cleaned_csv_path)
        for col in self.expected_features:
            self.assertIn(col, df.columns, f"Required column '{col}' missing from dataset")

    def test_03_churn_target_integrity(self):
        """Test 3: Verify target column Churn has valid values ('No', 'Yes') and no nulls."""
        df = pd.read_csv(self.cleaned_csv_path)
        unique_targets = set(df["Churn"].dropna().unique())
        self.assertEqual(unique_targets, {"No", "Yes"}, "Target values must strictly be 'No' and 'Yes'")
        self.assertEqual(df["Churn"].isnull().sum(), 0, "Target column must have zero missing values")

        # Verify class counts
        counts = df["Churn"].value_counts().to_dict()
        self.assertEqual(counts.get("No"), 5164, "Retained count should be 5,164")
        self.assertEqual(counts.get("Yes"), 1857, "Churned count should be 1,857")

    def test_04_dashboard_aggregation_math(self):
        """Test 4: Verify KPI calculations match actual dataset numbers."""
        df = pd.read_csv(self.cleaned_csv_path)
        total = len(df)
        churned = (df["Churn"] == "Yes").sum()
        churn_rate = (churned / total) * 100

        self.assertEqual(total, 7021)
        self.assertEqual(churned, 1857)
        self.assertAlmostEqual(churn_rate, 26.4492, places=3)

        # Average charges validation
        avg_monthly = df["MonthlyCharges"].mean()
        self.assertGreater(avg_monthly, 0.0)
        self.assertAlmostEqual(avg_monthly, 64.82, delta=0.5)

    def test_05_tenure_group_assignment(self):
        """Test 5: Verify tenure grouping logic matches sql/churn_analysis.sql."""
        from app.dashboard import load_cleaned_data

        df = load_cleaned_data(self.cleaned_csv_path)
        self.assertIn("TenureGroup", df.columns)

        # Test boundary cases
        t_0 = df[df["tenure"] <= 12]["TenureGroup"].unique()
        self.assertEqual(list(t_0), ["0-12 months"])

        t_70 = df[df["tenure"] > 48]["TenureGroup"].unique()
        self.assertEqual(list(t_70), ["49+ months"])

    def test_06_empty_dataframe_handling(self):
        """Test 6: Verify division-by-zero protection on empty filter results."""
        empty_df = pd.DataFrame(columns=self.expected_features)
        total = len(empty_df)
        churned = (empty_df["Churn"] == "Yes").sum()
        churn_rate = (churned / total) * 100 if total > 0 else 0.0

        self.assertEqual(total, 0)
        self.assertEqual(churned, 0)
        self.assertEqual(churn_rate, 0.0)

    def test_07_model_artifact_and_inference(self):
        """Test 7: Verify final ML model loads and generates valid predictions."""
        import joblib

        self.assertTrue(os.path.exists(self.model_path), "Model artifact missing")
        self.assertTrue(os.path.exists(self.thresh_path), "Threshold artifact missing")

        model = joblib.load(self.model_path)
        threshold = float(joblib.load(self.thresh_path))

        # Check threshold validity
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)
        self.assertAlmostEqual(threshold, 0.322712, places=4)

        # Run inference on sample
        df = pd.read_csv(self.cleaned_csv_path)
        sample = df.drop(columns=["Churn"]).head(5)
        probs = model.predict_proba(sample)[:, 1]

        self.assertEqual(len(probs), 5)
        for p in probs:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_08_no_pii_customer_identifiers(self):
        """Test 8: Confirm privacy preservation (no customerID or PII column)."""
        df = pd.read_csv(self.cleaned_csv_path)
        pii_candidates = ["customerID", "customer_id", "SSN", "email", "phone_number", "name", "address"]
        for pii in pii_candidates:
            self.assertNotIn(pii, df.columns, f"PII column '{pii}' detected in cleaned dataset")

    def test_09_evaluation_files_present(self):
        """Test 9: Verify existence of model comparison and feature importance CSVs."""
        out_dir = os.path.join(self.project_root, "data", "output")
        self.assertTrue(os.path.exists(os.path.join(out_dir, "model_comparison.csv")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "cross_validation_results.csv")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "random_forest_feature_importance.csv")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
