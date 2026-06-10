"""
The code defines a simple **Prefect workflow** that adds two integers.

It has two parts:

1. **`compute_data` task** — takes two integers `x` and `y` and returns their sum.

2. **`math_flow` flow** — the main entry point. It calls `compute_data` and returns the result.

The `@task` and `@flow` decorators are from the [Prefect](https://www.prefect.io/) orchestration library, which means the code gets Prefect's built-in features for free — things like logging, retries, observability, and scheduling — without changing the core logic at all.

In practice, calling `math_flow(3, 4)` would return `7`.
"""

from prefect import flow, task

@task
def compute_data(x: int, y: int) -> int:
    """
    Add two integers and return the result.

    Args:
        x: The first integer operand.
        y: The second integer operand.

    Returns:
        The sum of x and y.
    """
    return x + y

@flow
def math_flow(x: int, y: int) -> int:
    """
    Prefect flow that computes the sum of two integers.

    Delegates the addition to the `compute_data` task and returns
    its result.

    Args:
        x: The first integer operand.
        y: The second integer operand.

    Returns:
        The sum of x and y.
    """
    result = compute_data(x, y)
    return result
