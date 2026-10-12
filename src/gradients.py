"""Analytical gradients for the built-in functions."""


def quadratic_gradient(x):
    """Derivative of x² is 2x."""
    return 2 * x


def shifted_quadratic_gradient(x):
    """Derivative of (x - 5)² is 2(x - 5)."""
    return 2 * (x - 5)


def quartic_gradient(x):
    """Derivative of x⁴ is 4x³."""
    return 4 * x ** 3