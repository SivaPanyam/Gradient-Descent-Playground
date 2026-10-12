# Gradient Descent Playground

An interactive, beginner-friendly application for understanding gradient descent and mathematical optimization. Experiment with learning rates, initial values, and iterations while watching the optimization process through visualizations.

## Features

- Interactive Streamlit interface for experimenting with gradient descent.
- Built-in mathematical functions:
  - Quadratic: `f(x) = x²`
  - Shifted quadratic: `f(x) = (x - 5)²`
  - Quartic: `f(x) = x⁴`
- Custom mathematical expressions with symbolic differentiation.
- Adjustable initial value, learning rate, and iteration count.
- Step-by-step optimization with Next Step, Run to End, Animate, and Reset controls.
- Visualizations of the function curve and optimization path.
- Loss tracking to observe changes over iterations.
- Training history for inspecting optimization steps.
- Learning-rate comparison to explore convergence behavior.
- Automated tests for core functionality.

## What Is Gradient Descent?

Gradient descent is an optimization algorithm that repeatedly updates a parameter in the direction that reduces a function's value.

The update rule is:

$$
x_{\text{new}} = x_{\text{old}} - \eta f'(x_{\text{old}})
$$

Where:

- `x_old` is the current parameter value.
- `η` is the learning rate, which controls the step size.
- `f'(x_old)` is the derivative at the current value.
- `x_new` is the updated parameter value.

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
```

## Getting Started

### Prerequisites

- Python 3.10 or a compatible newer version
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/SivaPanyam/Gradient-Descent-Playground.git
cd Gradient-Descent-Playground
```

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Application

```bash
python -m streamlit run src/main.py
```

Open the local URL displayed in your terminal, usually:

`http://localhost:8501`

## How to Use

1. Launch the application.
2. Select a built-in function or choose the custom-expression option.
3. Adjust the initial value, learning rate, and number of iterations.
4. Click **Next Step** to inspect one optimization update.
5. Use **Run to End** to complete optimization or **Animate** to watch the updates.
6. Inspect the current parameter, gradient, and loss.
7. Explore the optimization path, loss chart, and training history.
8. Compare learning rates to understand their impact on convergence.
9. Reset the experiment and try different settings.

### Example Experiment

Select the quadratic function:

`f(x) = x²`

Set the following parameters:

- Initial value: `5`
- Learning rate: `0.1`
- Iterations: `20`

Observe how the parameter moves toward the minimum at `x = 0` and how the loss decreases.

## Run Tests

Run the automated test suite from the project root:

```bash
python -m pytest -v
```

The tests help verify gradient calculations, optimization behavior, and custom-expression handling.

## Learning Outcomes

This project provides hands-on practice with:

- Derivatives and gradients
- The gradient descent update rule
- Learning-rate selection
- Convergence and divergence
- Loss functions and optimization history
- Mathematical function visualization
- Symbolic differentiation
- Testing numerical algorithms

## Custom Expressions and Safety

Custom expressions are processed using a restricted parser rather than unrestricted Python `eval`. Only the mathematical operators and functions supported by the implementation are available. Unsupported expressions and invalid numerical results may be rejected.

Refer to `src/custom_functions.py` for the supported syntax.

## Contributing

Contributions, bug reports, and feature suggestions are welcome.

1. Fork the repository.
2. Create a branch for your changes.
3. Implement your changes and add or update tests.
4. Run `python -m pytest -v`.
5. Submit a pull request describing your changes.

## License

No license has been specified yet. Until a license is added, others do not automatically have permission to reuse, modify, or redistribute this project. Consider adding an open-source license if you want to encourage community contributions.

---

**Gradient Descent Playground** — Learn optimization by experimenting, visualizing, and exploring.
