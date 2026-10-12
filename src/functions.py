"""Objective functions used in the playground."""


def quadratic(x):
    """f(x) = x²"""
    return x ** 2


def shifted_quadratic(x):
    """f(x) = (x - 5)²"""
    return (x - 5) ** 2


def quartic(x):
    """f(x) = x⁴"""
    return x ** 4