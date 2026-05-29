"""
feast_apply — manual-trigger DAG that registers/updates feature definitions
in the Feast s3 registry (s3://feast-offline/registry.db).

Replaces the Helm post-install Job that previously ran `feast apply` on every
helm install/upgrade of the feast chart. Decoupling apply from Helm means
feature catalog evolution doesn't require touching the chart — push to the
feast-feature-repo on GitHub and trigger this DAG.

Architecture (single task, KubernetesPodOperator):

  ┌──────────────────────────────────────────────────────┐
  │ Pod                                                  │
  │  init: git-sync (one-time)                           │
  │    clones github.com/asafwat/feast-feature-repo      │
  │    into /git/current (shared emptyDir)               │
  │                                                      │
  │  main: feast-feature-server:0.5                      │
  │    workingDir: /git/current                          │
  │    command: feast apply                              │
  │    env: REDIS_PASSWORD + AWS_* (k8s Secret refs)     │
  └──────────────────────────────────────────────────────┘

Trigger this DAG manually from the Airflow UI after pushing a change to
feast-feature-repo. In a future iteration a GitHub webhook can fire this DAG
automatically on push (Airflow REST API: POST /dags/feast_apply/dagRuns).
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

# ── Shared constants (mirrored in feast_feature_pipeline_dag.py) ─────────────
FEAST_NS = "feast"
FEAST_IMAGE = "feast-feature-server:0.7"
GIT_SYNC_IMAGE = "registry.k8s.io/git-sync/git-sync:v4.3.0"
FEATURE_REPO_URL = "https://github.com/asafwat/feast-feature-repo.git"

# Shared volume between the gitSync initContainer and the main container.
FEATURE_REPO_VOLUME = V1Volume(
    name="feature-repo",
    empty_dir=V1EmptyDirVolumeSource(),
)
FEATURE_REPO_MOUNT = V1VolumeMount(name="feature-repo", mount_path="/git")

# gitSync as an INIT container — one-time clone, then the main container runs
# Feast against /git/current. (The Feast Feature Server Deployment uses gitSync
# as a continuous sidecar; here we just need the catalog once per Job run.)
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

# Env shared by the apply task and (in the other DAG) the materialize + train
# tasks. Same secret refs as the chart's feast.env helper.
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


@dag(
    dag_id="feast_apply",
    description="Register/update Feast feature definitions in the s3 registry",
    schedule=None,   # manual trigger only (future: GitHub webhook on feast-feature-repo)
    start_date=datetime(2026, 5, 1),
    catchup=False,
    default_args={
        "owner": "ml-platform",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["feast", "mlops", "feature-store"],
)
def feast_apply():

    KubernetesPodOperator(
        task_id="feast_apply",
        namespace=FEAST_NS,
        image=FEAST_IMAGE,
        image_pull_policy="IfNotPresent",
        # Override ENTRYPOINT (`feast serve ...`) — run `feast apply` instead,
        # against the gitSync'd catalog at /git/current.
        cmds=["sh", "-c"],
        arguments=["cd /git/current && feast apply"],
        env_vars=FEAST_ENV,
        init_containers=[GIT_SYNC_INIT_CONTAINER],
        volumes=[FEATURE_REPO_VOLUME],
        volume_mounts=[FEATURE_REPO_MOUNT],
        get_logs=True,
        on_finish_action="delete_pod",
    )


feast_apply()
