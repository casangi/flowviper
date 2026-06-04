# Workflow Tests

This repository contains Prefect workflow integration examples and tests for TestVIPER.

## What is Prefect?

Prefect is an open-source workflow orchestration engine that turns Python functions into production-grade pipelines with minimal friction. You can build and schedule workflows in pure Python—no DSLs or complex config files—and run them anywhere Python runs. Prefect handles the heavy lifting for you out of the box, including automatic state tracking, failure handling, real-time monitoring, and more.

[Prefect Introduction](https://docs.prefect.io/v3/get-started)

## Workflow Terminology
[Prefect Terminology](https://docs.prefect.io/v3/concepts)

### Flow

Flows are defined as decorated Python functions.
Flows accept inputs, perform work, and potentially return a result.

Generally, flows behave like Python functions, but they have some additional capabilities:
- Metadata about their execution, like each state the flow enters, is automatically tracked.
- Input arguments can be type validated as workflow parameters.
- Retries can be performed on failure, with configurable delay and retry limits.
- Timeouts can be enforced to prevent unintentional, long-running workflows.
- A flow can be deployed, which exposes an API for interacting with it remotely.

### Task
Tasks are defined as decorated Python functions. Tasks are cacheable, retryable units of work that can execute concurrently and support transactional semantics.
Like flows, tasks can call other tasks or flows; there is no required nesting pattern.
Generally, tasks behave like normal Python functions, but they have additional capabilities:
- metadata about task runs, such as run time and final state, is automatically tracked
- each state the task enters is recorded, enabling observability and state-based logic
- futures from upstream tasks are automatically resolved by downstream tasks
- retries can be performed on failure, with configurable delay and retry limits
- caching enables result reuse across workflow executions
- concurrency via `.submit()` and `.map()` allows concurrent execution within and across workflows
- timeouts can be enforced to prevent unintentional, long-running operations

Tasks are uniquely identified by a task key, which is a hash composed of the task name and the fully qualified function name.

## Testing Workflows

Test Prefect flows and tasks by running them against an isolated, temporary backend or by calling the underlying function directly.

[Prefect Testing Workflows](https://docs.prefect.io/v3/how-to-guides/workflows/test-workflows)

### Writing workflow tests

Use prefect_test_harness as a context manager to run flows and tasks against a temporary local SQLite database:

```python
from prefect import flow
from prefect.testing.utilities import prefect_test_harness

@flow
def my_favorite_flow():
    return 42

def test_my_favorite_flow():
  with prefect_test_harness():
      # run the flow against a temporary testing database
      assert my_favorite_flow() == 42
```

For more extensive testing, use prefect_test_harness as a fixture in your unit testing framework. For example, when using pytest:

```python
from prefect import flow
import pytest
from prefect.testing.utilities import prefect_test_harness

@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield

@flow
def my_favorite_flow():
    return 42

def test_my_favorite_flow():
    assert my_favorite_flow() == 42
```

### Prefect testing behavior
- `prefect_test_harness` is used to run Prefect in an isolated local backend environment.
- The tests are designed to verify the flow output and the task logic directly.
