
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.functions import (
    quadratic,
    shifted_quadratic,
    quartic,
)
from src.gradients import (
    quadratic_gradient,
    shifted_quadratic_gradient,
    quartic_gradient,
)
from src.optimizer import gradient_descent
from src.custom_functions import build_custom_function


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="Gradient Descent Playground",
    page_icon="📉",
    layout="wide",
)

st.title("Gradient Descent Playground")
st.caption(
    "Learn optimization by experimenting with functions, "
    "gradients, and learning rates."
)


# =========================================================
# Learning introduction
# =========================================================
with st.expander("What is gradient descent?", expanded=True):
    st.markdown(
        """
        **Gradient descent** is an optimization algorithm that repeatedly
        adjusts a parameter to reduce a function's loss.

        Imagine standing on a hill and trying to reach its lowest point.
        The gradient tells you the local slope, and the learning rate
        controls the size of each step.

        ### The update rule

        """
    )

    st.latex(
        r"x_{\text{new}} = x_{\text{old}} "
        r"- \eta f'(x_{\text{old}})"
    )

    st.markdown(
        """
        - **Parameter (`x`)**: the value we want to optimize.
        - **Loss (`f(x)`)**: how much error the current parameter produces.
        - **Gradient (`f'(x)`)**: the slope of the function.
        - **Learning rate (`η`)**: how far each update moves.

        A positive gradient moves `x` downwards; a negative gradient
        moves it upwards. Both can reduce the loss when the step size
        is appropriate.
        """
    )

    with st.expander("See a numerical example"):
        st.markdown(
            """
            For `f(x) = x²`, the derivative is `f'(x) = 2x`.

            Start with `x = 10` and learning rate `0.1`:

            1. Gradient: `2 × 10 = 20`
            2. Update: `10 − (0.1 × 20) = 8`
            3. New loss: `8² = 64`

            The parameter moved from 10 to 8, and the loss fell from
            100 to 64.
            """
        )


# =========================================================
# Available objective functions
# =========================================================
function_options = {
    "Quadratic: x²": (quadratic, quadratic_gradient),
    "Shifted quadratic: (x - 5)²": (
        shifted_quadratic,
        shifted_quadratic_gradient,
    ),
    "Quartic: x⁴": (quartic, quartic_gradient),
}

function_names = list(function_options) + ["Custom expression"]


# =========================================================
# Sidebar: experiment configuration
# =========================================================
with st.sidebar:
    st.header("Experiment settings")

    selected_function = st.selectbox(
        "Objective function",
        function_names,
    )

    custom_expression = ""

    if selected_function == "Custom expression":
        custom_expression = st.text_input(
            "Enter f(x)",
            value="x**2 + 3*x + 2",
            help=(
                "Examples: x**2, (x-5)**2, sin(x)+x**2. "
                "Use ** for powers."
            ),
        )

    initial_x = st.slider(
        "Initial parameter (x)",
        min_value=-10.0,
        max_value=10.0,
        value=10.0,
        step=0.5,
    )

    learning_rate = st.select_slider(
        "Learning rate",
        options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9],
        value=0.1,
    )

    iterations = st.slider(
        "Maximum iterations",
        min_value=1,
        max_value=100,
        value=30,
    )

    animation_delay = st.select_slider(
        "Animation speed",
        options=[0.05, 0.15, 0.3, 0.5],
        value=0.15,
        format_func=lambda value: (
            "Fast" if value == 0.05
            else "Medium" if value == 0.15
            else "Slow" if value == 0.3
            else "Very slow"
        ),
    )

    st.caption(
        "Tip: begin with a learning rate of 0.1, then experiment."
    )


# =========================================================
# Resolve the selected function
# =========================================================
try:
    if selected_function == "Custom expression":
        function, gradient = build_custom_function(
            custom_expression
        )
    else:
        function, gradient = function_options[selected_function]

except (ValueError, TypeError, OverflowError) as error:
    st.error(f"Cannot use this expression: {error}")
    st.stop()


# =========================================================
# Utility: safely evaluate a curve for plotting
# =========================================================
def safe_curve(objective, points):
    """Return function values, using NaN for undefined points."""
    values = []

    for point in points:
        try:
            value = float(objective(float(point)))

            if not math.isfinite(value):
                value = np.nan

        except (ValueError, OverflowError, ZeroDivisionError):
            value = np.nan

        values.append(value)

    return np.asarray(values, dtype=float)


# =========================================================
# Build optimization history
# =========================================================
config = (
    selected_function,
    custom_expression,
    initial_x,
    learning_rate,
    iterations,
)

if st.session_state.get("config") != config:
    st.session_state["config"] = config
    st.session_state["current_step"] = 0

    try:
        st.session_state["history"] = gradient_descent(
            function=function,
            gradient=gradient,
            initial_x=initial_x,
            learning_rate=learning_rate,
            iterations=iterations,
        )
        st.session_state["optimization_error"] = None

    except (ValueError, TypeError, OverflowError) as error:
        st.session_state["history"] = []
        st.session_state["optimization_error"] = str(error)

if st.session_state.get("optimization_error"):
    st.error(st.session_state["optimization_error"])
    st.info(
        "Try a different starting value, a smaller learning rate, "
        "or another function."
    )
    st.stop()

history = st.session_state["history"]

if not history:
    st.warning("No optimization history is available.")
    st.stop()


# =========================================================
# Optimization controls
# =========================================================
st.divider()
st.header("Run your experiment")

st.write(
    "Advance one iteration at a time, complete the run, "
    "or watch the parameter update automatically."
)

control1, control2, control3, control4 = st.columns(4)

with control1:
    next_clicked = st.button(
        "Next Step",
        use_container_width=True,
        disabled=(
            st.session_state["current_step"] >= len(history) - 1
        ),
    )

with control2:
    run_clicked = st.button(
        "Run to End",
        use_container_width=True,
        type="primary",
    )

with control3:
    animate_clicked = st.button(
        "Animate",
        use_container_width=True,
        disabled=(
            st.session_state["current_step"] >= len(history) - 1
        ),
    )

with control4:
    reset_clicked = st.button(
        "Reset",
        use_container_width=True,
    )

if next_clicked:
    st.session_state["current_step"] += 1

if run_clicked:
    st.session_state["current_step"] = len(history) - 1

if reset_clicked:
    st.session_state["current_step"] = 0


# =========================================================
# Animated playback
# =========================================================
if animate_clicked:
    start_step = st.session_state["current_step"]

    animation_status = st.empty()
    animation_metrics = st.empty()
    animation_chart = st.empty()
    animation_progress = st.empty()

    # Plot range follows the trajectory while keeping a useful view.
    all_parameters = [row["x"] for row in history]
    plot_min = min(-10.0, min(all_parameters) - 1.0)
    plot_max = max(10.0, max(all_parameters) + 1.0)

    for step in range(start_step + 1, len(history)):
        st.session_state["current_step"] = step

        visible = history[:step + 1]
        current = visible[-1]

        animation_status.info(
            f"Applying update {step} of {iterations}"
        )

        with animation_metrics.container():
            a, b, c = st.columns(3)
            a.metric("Parameter x", f'{current["x"]:.6f}')
            b.metric("Gradient", f'{current["gradient"]:.6g}')
            c.metric("Loss", f'{current["loss"]:.6g}')

        with animation_chart.container():
            fig, ax = plt.subplots()

            points = np.linspace(plot_min, plot_max, 400)
            curve = safe_curve(function, points)

            ax.plot(points, curve, label="Objective function")
            ax.plot(
                [row["x"] for row in visible],
                [row["loss"] for row in visible],
                "o-",
                markersize=3,
                label="Optimization path",
            )
            ax.set_xlabel("Parameter x")
            ax.set_ylabel("Loss f(x)")
            ax.set_title("Gradient descent in motion")
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close(fig)

        animation_progress.progress(
            step / iterations,
            text=f"Iteration {step} / {iterations}",
        )

        time.sleep(animation_delay)

    animation_status.success("Animation completed.")


# =========================================================
# Current state
# =========================================================
current_step = st.session_state["current_step"]
visible_history = history[:current_step + 1]
current = visible_history[-1]

st.progress(
    current_step / iterations,
    text=f"Iteration {current_step} of {iterations}",
)

st.subheader("Current optimization state")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Iteration", current["iteration"])
m2.metric("Parameter x", f'{current["x"]:.6f}')
m3.metric("Gradient", f'{current["gradient"]:.6g}')
m4.metric("Loss", f'{current["loss"]:.6g}')


# =========================================================
# Explain the current update
# =========================================================
st.subheader("How did the parameter change?")

if current_step == 0:
    st.info(
        "This is the starting point. Click Next Step or Animate "
        "to see the first update."
    )
else:
    previous = history[current_step - 1]

    st.latex(
        r"x_{\mathrm{new}} = x_{\mathrm{old}} "
        r"- \eta f'(x_{\mathrm{old}})"
    )

    explanation1, explanation2 = st.columns(2)

    with explanation1:
        st.write(f"Previous parameter: `{previous['x']:.6f}`")
        st.write(f"Previous gradient: `{previous['gradient']:.6g}`")
        st.write(f"Learning rate: `{learning_rate}`")

    with explanation2:
        st.write(
            f"New parameter: **{current['x']:.6f}**"
        )
        st.write(
            f"Previous loss: `{previous['loss']:.6g}`"
        )
        st.write(
            f"New loss: `{current['loss']:.6g}`"
        )

    st.code(
        f"x_new = {previous['x']:.6f} "
        f"- ({learning_rate} * {previous['gradient']:.6g})\n"
        f"x_new = {current['x']:.6f}",
        language="text",
    )

    loss_change = current["loss"] - previous["loss"]

    if loss_change < 0:
        st.success("The loss decreased in this update.")
    elif loss_change > 0:
        st.warning(
            "The loss increased in this update. "
            "Try a smaller learning rate if this continues."
        )
    else:
        st.info("The loss did not change in this update.")


# =========================================================
# Visualizations
# =========================================================
st.divider()
st.header("Optimization visualizations")

left, right = st.columns(2)

with left:
    st.subheader("Loss vs. iteration")

    fig_loss, ax_loss = plt.subplots()

    ax_loss.plot(
        [row["iteration"] for row in visible_history],
        [row["loss"] for row in visible_history],
        marker="o",
        markersize=3,
    )

    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss convergence")
    ax_loss.set_yscale("symlog", linthresh=1e-8)
    ax_loss.grid(True, alpha=0.3)

    st.pyplot(fig_loss)
    plt.close(fig_loss)

with right:
    st.subheader("Objective function and path")

    all_x = [row["x"] for row in visible_history]

    lower = min(-10.0, min(all_x) - 1.0)
    upper = max(10.0, max(all_x) + 1.0)

    xs = np.linspace(lower, upper, 400)
    ys = safe_curve(function, xs)

    fig_path, ax_path = plt.subplots()

    ax_path.plot(xs, ys, label="Objective function")

    ax_path.plot(
        all_x,
        [row["loss"] for row in visible_history],
        "o-",
        markersize=4,
        label="Optimization path",
    )

    ax_path.scatter(
        [all_x[0]],
        [visible_history[0]["loss"]],
        label="Starting point",
        zorder=3,
    )

    ax_path.scatter(
        [all_x[-1]],
        [visible_history[-1]["loss"]],
        label="Current point",
        zorder=3,
    )

    ax_path.set_xlabel("Parameter x")
    ax_path.set_ylabel("Loss f(x)")
    ax_path.set_title("Parameter movement")
    ax_path.set_yscale("symlog", linthresh=1e-8)
    ax_path.legend()
    ax_path.grid(True, alpha=0.3)

    st.pyplot(fig_path)
    plt.close(fig_path)


# =========================================================
# Training history
# =========================================================
st.divider()
st.subheader("Iteration history")

st.dataframe(
    visible_history,
    use_container_width=True,
    hide_index=True,
)

if current_step == iterations:
    st.success("All iterations completed.")
else:
    st.info(
        "Use Next Step to inspect one update, or Animate to "
        "watch the optimization trajectory."
    )


# =========================================================
# Learning-rate comparison
# =========================================================
st.divider()
st.header("Compare learning rates")

st.write(
    "Compare how different learning rates affect convergence "
    "while keeping the function, initial parameter, and "
    "iteration count fixed."
)

comparison_rates = st.multiselect(
    "Choose learning rates",
    options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9],
    default=[0.01, 0.1, 0.5],
    key="comparison_rates",
)

if st.button("Compare", key="compare_button"):
    if not comparison_rates:
        st.warning("Select at least one learning rate.")
    else:
        fig_compare, ax_compare = plt.subplots()
        plotted_any = False

        for rate in comparison_rates:
            try:
                comparison_history = gradient_descent(
                    function=function,
                    gradient=gradient,
                    initial_x=initial_x,
                    learning_rate=rate,
                    iterations=iterations,
                )

                losses = [
                    row["loss"] for row in comparison_history
                ]

                ax_compare.plot(
                    range(len(losses)),
                    losses,
                    marker="o",
                    markersize=2,
                    label=f"Learning rate = {rate}",
                )
                plotted_any = True

            except (ValueError, TypeError, OverflowError) as error:
                st.warning(
                    f"Learning rate {rate} failed: {error}"
                )

        if plotted_any:
            ax_compare.set_xlabel("Iteration")
            ax_compare.set_ylabel("Loss")
            ax_compare.set_title("Learning-rate comparison")
            ax_compare.set_yscale("symlog", linthresh=1e-8)
            ax_compare.legend()
            ax_compare.grid(True, alpha=0.3)

            st.pyplot(fig_compare)
            plt.close(fig_compare)