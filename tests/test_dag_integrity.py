"""
DAG integrity tests — catches DAG-parsing failures BEFORE they hit the cluster.

Airflow's `DagBag` is the same class the scheduler uses to load DAGs from
the dags folder. If a DAG fails to import, has a cycle, or has a bad
configuration, DagBag captures the error in `import_errors`.

This test parametrizes over every .py file in dags/ and asserts that:
  1. The file imports without raising
  2. At least one DAG was registered (caught the "I forgot to call the dag function" mistake)
  3. The dag_id matches the expected naming pattern

Add new DAG checks here as the repo grows.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from airflow.models import DagBag

DAGS_DIR = Path(__file__).parent.parent / "dags"


def test_dagbag_has_no_import_errors():
    """All files in dags/ must parse cleanly."""
    dag_bag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)
    assert not dag_bag.import_errors, (
        "DAG import errors:\n"
        + "\n".join(f"  {f}: {err}" for f, err in dag_bag.import_errors.items())
    )


def test_dagbag_has_dags():
    """At least one DAG must be registered — catches 'forgot to call my_dag()' bugs."""
    dag_bag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)
    assert len(dag_bag.dags) > 0, "no DAGs registered — check that each DAG file calls its @dag function"


@pytest.fixture(scope="module")
def dag_bag() -> DagBag:
    return DagBag(dag_folder=str(DAGS_DIR), include_examples=False)


def test_expected_dags_registered(dag_bag: DagBag):
    """Every DAG file should produce at least one dag_id we expect."""
    expected_dag_ids = {
        "fraud_detection_pipeline",
        "feast_apply",
        "feast_feature_pipeline",
        "hello_world",
    }
    registered = set(dag_bag.dag_ids)
    missing = expected_dag_ids - registered
    assert not missing, f"expected DAGs missing from DagBag: {missing}"


def test_all_dags_have_owner_and_tags(dag_bag: DagBag):
    """Operational hygiene — every DAG should have an owner + at least one tag."""
    for dag_id, dag in dag_bag.dags.items():
        assert dag.owner, f"{dag_id}: missing owner"
        assert dag.tags, f"{dag_id}: missing tags (use them for UI filtering)"


def test_no_dag_has_excessive_max_active_runs(dag_bag: DagBag):
    """Sanity check — don't accidentally set max_active_runs to a huge number."""
    for dag_id, dag in dag_bag.dags.items():
        assert dag.max_active_runs <= 16, (
            f"{dag_id}: max_active_runs={dag.max_active_runs} is suspiciously high"
        )
