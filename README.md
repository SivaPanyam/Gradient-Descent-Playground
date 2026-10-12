Gradient Descent Playground

An interactive, beginner-friendly playground for learning how gradient
descent minimizes mathematical functions. Explore how the initial value,
learning rate, and number of iterations affect optimization, and
visualize the process step by step.

Features

Interactive Streamlit interface for experimenting with gradient
descent.

Built-in functions to explore different optimization landscapes:

Quadratic: f(x) = x²

Shifted quadratic: f(x) = (x - 5)²

Quartic: f(x) = x⁴

Custom mathematical expressions using a restricted expression
parser and symbolic differentiation.

Optimization controls for the initial value, learning rate, and
iteration count.

Step-by-step optimization with controls to advance, run to the
end, animate, and reset.

Visualizations of the function, optimization path, and loss over
iterations.

Training history showing the values recorded at each
optimization step.

Learning-rate comparison to help illustrate how different
learning rates change the optimization process.

Automated tests for core functionality.

What is gradient descent?

Gradient descent is an optimization algorithm that repeatedly updates a
parameter in the direction that reduces a function's value.

The update rule is:

x{\text{new}{=tex}} = x{\text{old}{=tex}} -
\eta {=tex}f'(x{\text{old}{=tex}})

Where:

(x{\text{old}{=tex}}) is the current parameter value.

(\eta{=tex}) (eta) is the learning rate, which controls the step
size.

(f'(x{\text{old}{=tex}})) is the derivative (gradient in one
dimension) at the current value.

(x{\text{new}{=tex}}) is the updated parameter value.

A learning rate that is too small can make progress slow. A learning
rate that is too large can cause the updates to overshoot or diverge.
Use the playground to observe these effects.

Tech stack

Python

Streamlit

NumPy

Matplotlib

SymPy

pytest

Project structure

gradient-descent-playground/
├── src/
│ ├── **init**.py
│ ├── functions.py
│ ├── gradients.py
│ ├── optimizer.py
│ ├── visualization.py
│ ├── custom_functions.py
│ └── main.py
├── tests/
│ ├── test_gradients.py
│ ├── test_optimizer.py
│ └── test_custom_functions.py
├── requirements.txt
├── README.md
└── .gitignore

Requirements

Python 3.10 or a compatible newer Python version

pip

Installation

1. Clone the repository

git clone https://github.com/SivaPanyam/Gradient-Descent-Playground.git
cd Gradient-Descent-Playground

2. Create a virtual environment

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

macOS/Linux:

python -m venv .venv
source .venv/bin/activate

3. Install dependencies

python -m pip install -r requirements.txt

Run the application

From the project root, run:

python -m streamlit run src/main.py

Streamlit will display a local URL in the terminal, usually:

http://localhost:8501

Open that URL in your browser.

How to use

Start the application.

Select one of the built-in functions or choose the custom-expression

option.

Set the initial value of x, learning rate, and number of

iterations.

Use Next Step to inspect one update at a time, Run to End to

complete the optimization, or Animate to watch the updates.

Review the parameter, gradient, and loss values.

Inspect the optimization path, loss chart, and training history.

Compare learning rates to see how step size changes the optimization

behavior.

Try a custom expression supported by the application and observe its

derivative and optimization path.

For a simple starting experiment, select x², set the initial value to
5, and use a learning rate of 0.1. Observe how the parameter moves
toward the minimum at x = 0.

Run tests

From the project root, run:

python -m pytest -v

All tests should pass before you commit changes. If a test fails, read
the traceback and fix the issue before publishing.

Custom expressions and safety

Custom expressions are intended for mathematical experimentation. The
application uses a restricted parser rather than unrestricted Python
eval. Only the operators and mathematical functions explicitly
supported by the implementation can be used. Unsupported expressions or
invalid numerical results may be rejected.

The exact supported expressions are determined by
src/custom_functions.py.

Learning goals

This project is designed to help learners build intuition for:

Derivatives and gradients

The gradient descent update rule

Learning-rate selection

Convergence and divergence

Loss tracking during optimization

The relationship between a function's shape and its optimization
path

Contributing

Contributions, bug reports, and suggestions are welcome.

Fork the repository.

Create a branch for your change.

Make the change and add or update tests.

Run python -m pytest -v.

Open a pull request describing the change.

License

No license has been specified yet. Until a license is added to this
repository, reuse and redistribution are not automatically granted.
Consider adding an open-source license if you want others to reuse or
modify the project.

Built as a hands-on learning project for understanding gradient descent
and optimization.
