import io
import json
import tempfile
import time
import unittest
from pathlib import Path

import ml_service
from app import app


class MLFlowTest(unittest.TestCase):
    def test_shared_theme_is_applied_to_all_pages(self):
        client = app.test_client()
        for path in ["/", "/passengers/new", "/passengers/1/edit", "/ml/dashboard", "/ml/train", "/ml/models", "/ml/predict", "/ml/batch"]:
            response = client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn(b"wandb-theme.css", response.data, path)
            self.assertIn(b'href="/ml/dashboard"', response.data, path)
            self.assertIn(b'href="/ml/train"', response.data, path)
            self.assertIn(b'href="/ml/models"', response.data, path)

    def test_equivalent_model_is_reused(self):
        metrics = {"accuracy": 0.8}
        params = {"model__C": 1.0}
        preprocessing = "test preprocessing"
        with tempfile.TemporaryDirectory() as directory:
            metadata = {
                "version": "test",
                "model_name": "logistic_regression",
                "search_mode": "quick",
                "best_params": params,
                "metrics": metrics,
                "feature_columns": ml_service.FEATURE_COLUMNS,
                "preprocessing_info": preprocessing,
            }
            Path(directory, "titanic_model_test.json").write_text(json.dumps(metadata), encoding="utf-8")
            self.assertEqual(
                ml_service._find_equivalent_model(directory, "logistic_regression", "quick", params, metrics, preprocessing),
                metadata,
            )

    def test_complete_ml_flow(self):
        client = app.test_client()
        self.assertEqual(client.get("/api/passengers?per_page=1").status_code, 200)
        dashboard_page = client.get("/ml/dashboard")
        self.assertIn(b"Data quality", dashboard_page.data)
        self.assertIn(b"Pipeline", dashboard_page.data)
        self.assertIn(b"Extra Trees", client.get("/ml/train").data)
        self.assertIn(b"Model registry", client.get("/ml/models").data)
        self.assertEqual(client.get("/api/ml/dashboard").get_json()["total"], 891)

        response = client.post("/api/ml/train", json={"model_name": "logistic_regression", "search_mode": "quick"})
        self.assertEqual(response.status_code, 202)
        for _ in range(120):
            state = client.get("/api/ml/train/status").get_json()
            if state["status"] != "training":
                break
            time.sleep(0.25)
        self.assertEqual(state["status"], "completed", state.get("error"))
        self.assertTrue(client.get("/api/ml/models").get_json()["items"])

        passenger = {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"}
        prediction = client.post("/api/ml/predict", json=passenger)
        self.assertEqual(prediction.status_code, 200)
        self.assertIn("survival_probability", prediction.get_json())
        self.assertEqual(client.post("/api/ml/predict/what-if", json=passenger).status_code, 200)

        csv_data = "Pclass,Sex,Age,SibSp,Parch,Fare,Embarked\n3,male,22,1,0,7.25,S\n"
        batch = client.post(
            "/api/ml/predict/csv",
            data={"file": (io.BytesIO(csv_data.encode()), "passengers.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(batch.status_code, 200)
        self.assertIn(b"survival_probability", batch.data)


if __name__ == "__main__":
    unittest.main()
