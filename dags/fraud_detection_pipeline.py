"""
Fraud Detection Pipeline — orchestrated by Airflow.

Same business logic as the DVC pipeline (generate -> preprocess -> train ->
evaluate), but here:
  - Each stage is an Airflow @task running in its own ephemeral pod
  - Data flows between tasks via MinIO (not local filesystem)
  - Task outputs (S3 paths, MLflow run_id) are passed via XCom
  - Training task logs params/metrics/model to MLflow
  - Failures retry automatically (default_args.retries)

Production talking points this demonstrates:
  - Use the data layer (MinIO) for data, XCom only for metadata
  - Each task is independent — re-runnable from any failed point
  - Tracking server URI uses K8s internal DNS (minio.mlops.svc.cluster.local)
"""

from datetime import datetime, timedelta

from airflow.sdk import dag, task

# ── Configuration ────────────────────────────────────────────────────────────
# In production, these would come from Airflow Variables or a secrets manager.
MINIO_ENDPOINT      = "http://minio.mlops.svc.cluster.local:9000"
MINIO_ACCESS_KEY    = "mlops-admin"
MINIO_SECRET_KEY    = "mlops-minio-123"
DATA_BUCKET         = "airflow-data"
MLFLOW_URI          = "http://mlflow.mlops.svc.cluster.local:5000"
MLFLOW_EXPERIMENT   = "fraud-detection-airflow"

# Training hyperparameters
PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "class_weight": "balanced",
    "threshold": 0.1,
    "fraud_rate": 0.05,
    "n_samples": 10_000,
}


def _s3_client():
    """Return a boto3 S3 client pointed at MinIO."""
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


def _ensure_bucket(s3, bucket: str):
    """Create bucket if it doesn't exist (idempotent)."""
    try:
        s3.head_bucket(Bucket=bucket)
    except s3.exceptions.ClientError:
        s3.create_bucket(Bucket=bucket)


@dag(
    dag_id="fraud_detection_pipeline",
    description="End-to-end fraud detection pipeline: generate -> preprocess -> train -> evaluate",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={
        "owner": "ahmed",
        "retries": 2,
        "retry_delay": timedelta(seconds=30),
    },
    tags=["mlops", "fraud-detection", "mlflow"],
)
def fraud_detection_pipeline():

    @task
    def generate_data() -> str:
        """Generate synthetic transactions, upload raw CSV to MinIO."""
        import pandas as pd
        from airflow.sdk import get_current_context
        from sklearn.datasets import make_classification

        ctx = get_current_context()
        run_id = ctx["dag_run"].run_id.replace(":", "_").replace("+", "_")

        fraud_rate = PARAMS["fraud_rate"]
        X, y = make_classification(
            n_samples=PARAMS["n_samples"],
            n_features=20, n_informative=15, n_redundant=5,
            weights=[1 - fraud_rate, fraud_rate],
            random_state=42,
        )
        feature_cols = [f"feature_{i:02d}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_cols)
        df["is_fraud"] = y

        s3 = _s3_client()
        _ensure_bucket(s3, DATA_BUCKET)
        key = f"{run_id}/raw/transactions.csv"
        s3.put_object(Bucket=DATA_BUCKET, Key=key, Body=df.to_csv(index=False).encode())

        path = f"s3://{DATA_BUCKET}/{key}"
        print(f"Generated {len(df):,} transactions ({df['is_fraud'].sum()} fraud) → {path}")
        return path

    @task
    def preprocess(raw_path: str) -> dict:
        """Read raw, train/test split, scale, write back to MinIO."""
        import io

        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        bucket = raw_path.split("/")[2]
        key = "/".join(raw_path.split("/")[3:])
        run_prefix = key.split("/")[0]

        s3 = _s3_client()
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))

        feature_cols = [c for c in df.columns if c != "is_fraud"]
        X = df[feature_cols].values
        y = df["is_fraud"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42,
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_df = pd.DataFrame(X_train, columns=feature_cols)
        train_df["is_fraud"] = y_train
        test_df = pd.DataFrame(X_test, columns=feature_cols)
        test_df["is_fraud"] = y_test

        train_key = f"{run_prefix}/processed/train.csv"
        test_key  = f"{run_prefix}/processed/test.csv"
        s3.put_object(Bucket=bucket, Key=train_key, Body=train_df.to_csv(index=False).encode())
        s3.put_object(Bucket=bucket, Key=test_key,  Body=test_df.to_csv(index=False).encode())

        print(f"Train: {len(train_df):,} rows ({train_df['is_fraud'].sum()} fraud)")
        print(f"Test:  {len(test_df):,} rows ({test_df['is_fraud'].sum()} fraud)")
        return {
            "train": f"s3://{bucket}/{train_key}",
            "test":  f"s3://{bucket}/{test_key}",
        }

    @task
    def train(processed: dict) -> str:
        """Train a RandomForest, log params/model to MLflow, return run_id."""
        import io

        import mlflow
        import mlflow.sklearn
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        # Load training data
        train_path = processed["train"]
        bucket = train_path.split("/")[2]
        key = "/".join(train_path.split("/")[3:])

        s3 = _s3_client()
        obj = s3.get_object(Bucket=bucket, Key=key)
        train_df = pd.read_csv(io.BytesIO(obj["Body"].read()))

        feature_cols = [c for c in train_df.columns if c != "is_fraud"]
        X = train_df[feature_cols].values
        y = train_df["is_fraud"].values

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name="airflow-pipeline-train") as run:
            mlflow.set_tag("orchestrator", "airflow")
            mlflow.set_tag("dag_id", "fraud_detection_pipeline")
            mlflow.log_params(PARAMS)

            model = RandomForestClassifier(
                n_estimators=PARAMS["n_estimators"],
                max_depth=PARAMS["max_depth"],
                class_weight=PARAMS["class_weight"],
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X, y)
            mlflow.sklearn.log_model(model, name="model")

            print(f"Trained on {len(X):,} samples")
            print(f"MLflow run_id: {run.info.run_id}")
            return run.info.run_id

    @task
    def evaluate(processed: dict, mlflow_run_id: str) -> dict:
        """Load model from MLflow, evaluate on test set, log metrics back."""
        import io

        import mlflow
        import pandas as pd
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        mlflow.set_tracking_uri(MLFLOW_URI)
        model_uri = f"runs:/{mlflow_run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        # Load test data
        test_path = processed["test"]
        bucket = test_path.split("/")[2]
        key = "/".join(test_path.split("/")[3:])

        s3 = _s3_client()
        obj = s3.get_object(Bucket=bucket, Key=key)
        test_df = pd.read_csv(io.BytesIO(obj["Body"].read()))

        feature_cols = [c for c in test_df.columns if c != "is_fraud"]
        X = test_df[feature_cols].values
        y = test_df["is_fraud"].values

        # Predict with our chosen threshold
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= PARAMS["threshold"]).astype(int)

        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        metrics = {
            "accuracy":        float(accuracy_score(y, y_pred)),
            "precision":       float(precision_score(y, y_pred, zero_division=0)),
            "recall":          float(recall_score(y, y_pred)),
            "f1":              float(f1_score(y, y_pred, zero_division=0)),
            "auc_roc":         float(roc_auc_score(y, y_prob)),
            "true_positives":  int(tp),
            "false_negatives": int(fn),
            "false_positives": int(fp),
        }

        # Log metrics back to the same MLflow run
        with mlflow.start_run(run_id=mlflow_run_id):
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

        print(f"Caught {tp}/{tp+fn} frauds ({metrics['recall']:.1%} recall)")
        print(f"Precision: {metrics['precision']:.1%}  AUC-ROC: {metrics['auc_roc']:.4f}")
        return metrics

    # DAG flow
    raw = generate_data()
    processed = preprocess(raw)
    mlflow_run = train(processed)
    evaluate(processed, mlflow_run)


fraud_detection_pipeline()
