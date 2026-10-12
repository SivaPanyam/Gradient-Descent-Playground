# Gradient Descent Playground

An interactive, beginner-friendly application for understanding gradient descent and mathematical optimization. Experiment with learning rates, initial values, and iterations while watching the optimization process through live visualizations.

## Features

- **Interactive Streamlit interface** for experimenting with gradient descent.
- **Built-in mathematical functions**
  - Quadratic: `f(x) = x²`
  - Shifted quadratic: `f(x) = (x - 5)²`
  - Quartic: `f(x) = x⁴`
- **Custom mathematical expressions** with symbolic differentiation.
- **Adjustable parameters** for initial value, learning rate, and iteration count.
- **Step-by-step optimization** with Next Step, Run to End, Animate, and Reset controls.
- **Optimization visualizations** showing the function curve and optimization path.
- **Loss tracking** to observe how the function value changes over iterations.
- **Training history** for inspecting individual optimization steps.
- **Learning-rate comparison** to explore how different step sizes affect convergence.
- **Automated tests** for core functionality.

## What Is Gradient Descent?

Gradient descent is an optimization algorithm that repeatedly updates a parameter in the direction that reduces a function's value.

The update rule is:

\[
x_{\text{new}} = x_{\text{old}} - \eta f'(x_{\text{old}})
\]

Where:

- \(x_{\text{old}}\) is the current parameter value.
- \(\eta\) is the learning rate.
- \(f'(x_{\text{old}})\) is the derivative at the current value.
- \(x_{\text{new}}\) is the updated parameter value.

A small learning rate may lead to slow convergence, while a large learning rate may cause overshooting or divergence. This playground helps you explore these behaviors interactively.

## Tech Stack

- **Python** — core implementation
- **Streamlit** — interactive web interface
- **NumPy** — numerical computation
- **Matplotlib** — visualization
- **SymPy** — symbolic differentiation
- **pytest** — automated testing

## Project Structure

```text
gradient-descent-playground/
├── src/
│   ├── __init__.py
│   ├── functions.py
│   ├── gradients.py
│   ├── optimizer.py
│   ├── visualization.py
│   ├── custom_functions.py
│   └── main.py
├── tests/
│   ├── test_gradients.py
│   ├── test_optimizer.py
│   └── test_custom_functions.py
├── requirements.txt
├── README.md
└── .gitignore
