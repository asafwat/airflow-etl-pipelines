"""
feast_feature_pipeline — daily Feast operations: materialize features into
Redis (online store), then train the fraud-detection model using Feast's
point-in-time-correct historical feature retrieval.

Runs the same image as feast_apply_dag (feast-feature-server:0.5), but the
two tasks override the entrypoint with different verbs:
  - feast_materialize_incremental: `feast materialize-incremental {{ ts }}`
  - train_with_feast:               `python /app/train_with_feast.py`

The training script is baked into the image (it's "tooling" — doesn't evolve
with feature definitions). It reads feature_store.yaml from $FEAST_REPO_PATH
(which we set to /git/current, the gitSync mount), pulls labels from MinIO,
calls fs.get_historical_features(...) for the point-in-time join, trains a
sklearn Pipeline, and registers the model as `fraud-detector-feast` in MLflow.

Materialize is idempotent on the same logical date — re-runs are safe.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow.sdk import dag
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import (
    V1Container,
    V1EmptyDirVolumeSource,
    V1EnvVar,
    V1EnvVarSource,
    V1SecretKeySelector,
    V1Volume,
    V1VolumeMount,
)

# ── Shared constants (mirrored in feast_apply_dag.py) ────────────────────────
FEAST_NS = "feast"
FEAST_IMAGE = "feast-feature-server:0.7"
GIT_SYNC_IMAGE = "registry.k8s.io/git-sync/git-sync:v4.3.0"
FEATURE_REPO_URL = "https://github.com/asafwat/feast-feature-repo.git"

FEATURE_REPO_VOLUME = V1Volume(
    name="feature-repo",
    empty_dir=V1EmptyDirVolumeSource(),
)
FEATURE_REPO_MOUNT = V1VolumeMount(name="feature-repo", mount_path="/git")

GIT_SYNC_INIT_CONTAINER = V1Container(
    name="git-sync",
    image=GIT_SYNC_IMAGE,
    args=[
        f"--repo={FEATURE_REPO_URL}",
        "--ref=main",
        "--depth=1",
        "--root=/git",
        "--link=current",
        "--one-time",
        "--verbose=2",
    ],
    volume_mounts=[FEATURE_REPO_MOUNT],
)

# Env shared by both tasks. Plus MLflow URIs added to the train task only.
FEAST_ENV = [
    V1EnvVar(name="REDIS_PASSWORD",
             value_from=V1EnvVarSource(secret_key_ref=V1SecretKeySelector(
                 name="redis-auth", key="password"))),
    V1EnvVar(name="AWS_ACCESS_KEY_ID",
             value_from=V1EnvVarSource(secret_key_ref=V1SecretKeySelector(
                 name="feast-minio-creds", key="access-key"))),
    V1EnvVar(name="AWS_SECRET_ACCESS_KEY",
             value_from=V1EnvVarSource(secret_key_ref=V1SecretKeySelector(
                 name="feast-minio-creds", key="secret-key"))),
    V1EnvVar(name="AWS_ENDPOINT_URL",
             value="http://minio.mlops.svc.cluster.local:9000"),
    V1EnvVar(name="AWS_ENDPOINT_URL_S3",
             value="http://minio.mlops.svc.cluster.local:9000"),
    V1EnvVar(name="AWS_DEFAULT_REGION", value="us-east-1"),
    V1EnvVar(name="FEAST_USAGE", value="false"),
]

TRAIN_EXTRA_ENV = [
    V1EnvVar(name="FEAST_REPO_PATH", value="/git/current"),
    V1EnvVar(name="MLFLOW_TRACKING_URI",
             value="http://mlflow.mlops.svc.cluster.local:5000"),
    V1EnvVar(name="MLFLOW_S3_ENDPOINT_URL",
             value="http://minio.mlops.svc.cluster.local:9000"),
]


@dag(
    dag_id="feast_feature_pipeline",
    description="Daily Feast materialize-to-Redis + train fraud-detector-feast model",
    schedule="@daily",
    start_date=datetime(2026, 5, 1),
    catchup=False,
    default_args={
        "owner": "ml-platform",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["feast", "mlops", "fraud"],
)
def feast_feature_pipeline():

    materialize = KubernetesPodOperator(
        task_id="feast_materialize_incremental",
        namespace=FEAST_NS,
        image=FEAST_IMAGE,
        image_pull_policy="IfNotPresent",
        cmds=["sh", "-c"],
        # `materialize-incremental` reads features from the offline parquet
        # (s3://feast-offline/) and writes the latest snapshot per entity into
        # Redis. {{ logical_date.isoformat() }} is the Airflow Jinja-rendered
        # timestamp for this DAG run; Feast uses it as the "end time" of the
        # incremental window and remembers it for next time (idempotent).
        arguments=[
            "cd /git/current && "
            "feast materialize-incremental {{ logical_date.isoformat() }}"
        ],
        env_vars=FEAST_ENV,
        init_containers=[GIT_SYNC_INIT_CONTAINER],
        volumes=[FEATURE_REPO_VOLUME],
        volume_mounts=[FEATURE_REPO_MOUNT],
        get_logs=True,
        on_finish_action="delete_pod",
    )

    train = KubernetesPodOperator(
        task_id="train_with_feast",
        namespace=FEAST_NS,
        image=FEAST_IMAGE,
        image_pull_policy="IfNotPresent",
        # train_with_feast.py is baked into the image at /app/. It reads the
        # feature_repo path from $FEAST_REPO_PATH (set below to /git/current),
        # loads labels from s3, runs get_historical_features, trains, logs to
        # MLflow + registers the model.
        cmds=["python"],
        arguments=["/app/train_with_feast.py"],
        env_vars=FEAST_ENV + TRAIN_EXTRA_ENV,
        init_containers=[GIT_SYNC_INIT_CONTAINER],
        volumes=[FEATURE_REPO_VOLUME],
        volume_mounts=[FEATURE_REPO_MOUNT],
        get_logs=True,
        on_finish_action="delete_pod",
    )

    materialize >> train


feast_feature_pipeline()
