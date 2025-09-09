from typing import Any, Optional
import time

import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.core.algorithm import Algorithm
from matplotlib import pyplot as plt
from pymoo.algorithms.soo.nonconvex.de import DE

# You can import more alogorithms from pymoo.algorithms.soo or .moo
# from pymoo.algorithms.soo.nonconvex.de import DE
# See: https://github.com/anyoptimization/pymoo/tree/main/pymoo/algorithms
# https://pymoo.org/algorithms/list.html#nb-algorithms-list (Docs are incomplete)


class HistoryCallback(Callback):
    """
    Callback to record the history of evaluations and best values during optimization.

    Docs: https://pymoo.org/interface/callback.html
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_evals: list[float] = []
        self.best_values: list[float] = []
        self.all_values: list[Any] = []

    def notify(self, algorithm: Algorithm) -> None:
        self.n_evals.append(float(algorithm.evaluator.n_eval))
        self.best_values.append(algorithm.opt.get("F").flatten()[0])
        self.all_values.append(algorithm.pop.get("F").flatten())


def plot_history_callback(history: HistoryCallback, title: str, filename: str):
    """
    Plot the history of evaluations and best values during optimization.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(
        history.n_evals,
        history.best_values,
        label="Best Value",
        color="blue",
        marker="o",
    )
    plt.xlabel("Number of Evaluations")
    plt.ylabel("Best Value")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig(filename)


def function_1d(x: float) -> float:
    """Simple 1 Dimensional test function."""
    return x**2 / 25 - x / 3 + np.sin(x)


def function_10d(
    x: list[float],
) -> float:
    """
    Simple 10 Dimensional test function.
    """
    a0 = (x[:, 0] - 2) ** 2 / 25
    a1 = (x[:, 1] + 1) ** 2 / 25
    a2 = (x[:, 2] - 3) ** 2 / 25
    a3 = (x[:, 3] + 4) ** 2 / 25
    a4 = (x[:, 4] - 5) ** 2 / 25
    a5 = (x[:, 5] + 6) ** 2 / 25
    a6 = (x[:, 6] - 8) ** 2 / 25
    a7 = (x[:, 7] + 7) ** 2 / 25
    a8 = (x[:, 8] - 9) ** 2 / 25
    a9 = (x[:, 9] + 10) ** 2 / 25
    return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + 1.2


# TODO: Implement the Problem classes for 1D and 10D functions.
# Docs: https://pymoo.org/interface/problem.html
# https://pymoo.org/problems/definition.html
# If you are being fancy, you could try to assign Function1D = FunctionalProblem (a higher level interface)
# Examples of problems:
# https://github.com/anyoptimization/pymoo/tree/main/pymoo/problems/single
# Problem class:
# https://github.com/anyoptimization/pymoo/blob/main/pymoo/core/problem.py

class Function1D(Problem):
    """1D Problem for testing purposes."""

    def __init__(self):
        super().__init__(n_var=1, n_obj=1, xl=-5, xu=5)

    def _evaluate(
        self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any
    ) -> None:
        out["F"] = function_1d(x).reshape(-1, 1)

class Function10D(Problem):
    """10D Problem for testing purposes."""

    def __init__(self):
        super().__init__(n_var=10, n_obj=1, xl=-15, xu=15)

    def _evaluate(
        self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any
    ) -> None:
        # out["F"] = function_1d(x[:, 0:10]).reshape(-1, 10)
        out["F"] = function_10d(x).reshape(-1, 1)
        


# Try not to change the function signature
# We can call this function to unit test your submission.
def get_optimal_1d(n_evals: int = 100, seed: int = 42, verbose: bool = False) -> float:
    """Get the optimal input, and optimal value for the 1D function."""

    my_problem = Function1D()
    
    de_algo = DE(pop_size=100)

    history_callback = HistoryCallback()

    res = minimize(my_problem, de_algo, seed=seed, verbose=verbose, terminate=("n_eval", n_evals), callback=history_callback)

    plot_history_callback(history_callback, "my plot", "myfile.png")

    return {"x_opt": res.X, "f_opt": res.F,}


# Try not to change the function signature
# We ca call this function to unit test your submission.
def get_optimal_10d(
    n_evals: int = 100, seed: int = 42, verbose: bool = False
) -> dict[str, float]:
    my_problem = Function10D()
    
    de_algo = DE(pop_size=100)

    history_callback = HistoryCallback()

    res = minimize(my_problem, de_algo, seed=seed, verbose=verbose, terminate=("n_eval", n_evals), callback=history_callback)

    plot_history_callback(history_callback, "Best Value vs Num Evaluations", "P0_History.png")

    return {"x_opt": res.X, "f_opt": res.F,}

optimal_x_1d = 4.67195
optimal_x_10d = np.array([[ 2.0, -1.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0 ]])

if __name__ == "__main__":
    result_1d = get_optimal_1d(n_evals=200, seed=42, verbose=True)
    print(f"Optimal 1D: x = {result_1d['x_opt']}, f(x) = {result_1d['f_opt']}")
    print("Ground Truth optimal_x_1d value:", function_1d(optimal_x_1d))
    print("========================")
    result_10d = get_optimal_10d(n_evals=5000, seed=42, verbose=True)
    print(f"Optimal 10D: x = {result_10d['x_opt']}, f(x) = {result_10d['f_opt']}")
    print("Ground Truth optimal_x_10d value:", function_10d(optimal_x_10d))
