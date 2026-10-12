
import pytest

from src.functions import quadratic, shifted_quadratic
from src.gradients import quadratic_gradient, shifted_quadratic_gradient
from src.optimizer import gradient_descent


def test_quadratic_optimizer_converges():
    history = gradient_descent(
        function=quadratic,
        gradient=quadratic_gradient,
        initial_x=10,
        learning_rate=0.1,
        iterations=100,
    )

    assert len(history) == 101
    assert abs(history[-1]["x"]) < 1e-8
    assert history[-1]["loss"] < history[0]["loss"]


def test_shifted_quadratic_optimizer_converges():
    history = gradient_descent(
        function=shifted_quadratic,
        gradient=shifted_quadratic_gradient,
        initial_x=0,
        learning_rate=0.1,
        iterations=100,
    )

    assert abs(history[-1]["x"] - 5) < 1e-8
    assert history[-1]["loss"] < history[0]["loss"]


def test_history_contains_required_fields():
    history = gradient_descent(
        quadratic,
        quadratic_gradient,
        initial_x=10,
        learning_rate=0.1,
        iterations=2,
    )

    assert len(history) == 3
    assert set(history[0]) == {
        "iteration", "x", "gradient", "loss"
    }


def test_zero_iterations():
    history = gradient_descent(
        quadratic,
        quadratic_gradient,
        initial_x=3,
        learning_rate=0.1,
        iterations=0,
    )

    assert len(history) == 1
    assert history[0]["x"] == 3
    assert history[0]["loss"] == 9


@pytest.mark.parametrize(
    "learning_rate, iterations",
    [(0, 10), (-0.1, 10), (0.1, -1)],
)
def test_invalid_optimizer_settings(learning_rate, iterations):
    with pytest.raises(ValueError):
        gradient_descent(
            quadratic,
            quadratic_gradient,
            initial_x=10,
            learning_rate=learning_rate,
            iterations=iterations,
        )
