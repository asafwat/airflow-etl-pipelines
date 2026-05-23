"""
First DAG — sanity check that Airflow is running and picking up DAGs.

Uses the TaskFlow API (Airflow 2.0+ way), which is cleaner than the older
PythonOperator/BashOperator boilerplate. Each @task function becomes a node
in the DAG; chaining them with >> defines dependencies.
"""

from datetime import datetime, timedelta
from airflow.sdk import dag, task


@dag(
    dag_id="hello_world",
    description="Sanity-check DAG: prints messages and verifies Airflow + KubernetesExecutor work",
    schedule=None,                        # manual trigger only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={
        "owner": "ahmed",
        "retries": 2,
        "retry_delay": timedelta(seconds=30),
    },
    tags=["sanity-check", "tutorial"],
)
def hello_world():

    @task
    def greet():
        print("Hello from Airflow running on Kubernetes!")
        return "greeted"

    @task
    def add_two_numbers(x: int, y: int) -> int:
        result = x + y
        print(f"{x} + {y} = {result}")
        return result

    @task
    def announce_result(value: int):
        print(f"Final value passed through XCom: {value}")

    # Define the DAG flow
    greeting = greet()
    total = add_two_numbers(7, 35)
    announcement = announce_result(total)

    # Order: greet runs first, then add_two_numbers, then announce_result
    greeting >> total >> announcement


hello_world()
