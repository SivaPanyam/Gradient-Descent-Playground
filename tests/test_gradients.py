
from src.gradients import (
    quadratic_gradient,
    shifted_quadratic_gradient,
    quartic_gradient,
)


def test_quadratic_gradient():
    assert quadratic_gradient(3) == 6
    assert quadratic_gradient(0) == 0
    assert quadratic_gradient(-2) == -4


def test_shifted_quadratic_gradient():
    assert shifted_quadratic_gradient(3) == -4
    assert shifted_quadratic_gradient(5) == 0
    assert shifted_quadratic_gradient(7) == 4


def test_quartic_gradient():
    assert quartic_gradient(2) == 32
    assert quartic_gradient(0) == 0
    assert quartic_gradient(-2) == -32
