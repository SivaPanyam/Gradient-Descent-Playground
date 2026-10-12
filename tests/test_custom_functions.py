
import pytest

from src.custom_functions import build_custom_function


def test_polynomial_and_gradient():
    function, gradient = build_custom_function(
        "x**2 + 3*x + 2"
    )

    assert function(2) == pytest.approx(12)
    assert gradient(2) == pytest.approx(7)


def test_shifted_quadratic():
    function, gradient = build_custom_function("(x - 5)**2")

    assert function(3) == pytest.approx(4)
    assert gradient(3) == pytest.approx(-4)


def test_trigonometric_function():
    function, gradient = build_custom_function("sin(x) + x**2")

    assert function(0) == pytest.approx(0)
    assert gradient(0) == pytest.approx(1)


def test_rejects_unsupported_names():
    with pytest.raises(ValueError):
        build_custom_function("unknown(x)")


def test_rejects_python_code():
    with pytest.raises(ValueError):
        build_custom_function("__import__('os').system('whoami')")


def test_rejects_invalid_syntax():
    with pytest.raises(ValueError):
        build_custom_function("x**")
