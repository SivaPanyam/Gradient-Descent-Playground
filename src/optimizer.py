"""Gradient descent optimization engine."""

import math


def gradient_descent(
    function,
    gradient,
    initial_x,
    learning_rate,
    iterations,
):
    """Minimize a function and record the optimization history."""

    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite.")

    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise ValueError("Iterations must be an integer.")

    if iterations < 0:
        raise ValueError("Iterations cannot be negative.")

    if not math.isfinite(initial_x):
        raise ValueError("Initial parameter must be finite.")

    x = float(initial_x)
    history = []

    def record(step, parameter):
        try:
            loss = float(function(parameter))
            grad = float(gradient(parameter))
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            raise ValueError(
                f"Function or gradient is undefined at x={parameter:g}."
            ) from error

        if not math.isfinite(loss) or not math.isfinite(grad):
            raise ValueError(
                f"Non-finite loss or gradient at iteration {step}. "
                "Try a different initial value or smaller learning rate."
            )

        history.append({
            "iteration": step,
            "x": parameter,
            "gradient": grad,
            "loss": loss,
        })

    record(0, x)

    for step in range(1, iterations + 1):
        current_gradient = history[-1]["gradient"]
        next_x = x - learning_rate * current_gradient

        if not math.isfinite(next_x):
            raise ValueError(
                f"Parameter diverged at iteration {step}. "
                "Try a smaller learning rate."
            )

        x = next_x
        record(step, x)

    return history