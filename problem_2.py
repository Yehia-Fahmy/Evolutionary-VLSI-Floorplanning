import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from create_test_graphs import (
    create_layernorm_graph,
    create_dft_graph,
    create_lstm_graph,
    create_nas_lstm_graph,
    create_hadamard_transform_graph,
    create_parallel_linear_adds_graph,
    create_parallel_binary_tree_adds_graph,
    create_parallel_poly_graph,
)

from schedule import schedule, Context, asap_schedule
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.core.algorithm import Algorithm
from pymoo.termination import get_termination
from pymoo.core.termination import TerminateIfAny, Termination
from pymoo.util.hv import hv
import tqdm
import time
from collections import defaultdict
from pymoo.util.ref_dirs import get_reference_directions


class AreaDelayMinimizationProblem(Problem):
    """
    Pymoo Problem class for minimizing Area-Delay Product (ADP) by optimizing
    both operation priorities and operator counts. This is a multi-objective
    problem to minimize latency and area.
    """

    def __init__(self, config: "ProblemConfig"):
        self.config = config
        self.context = config.context
        self.op_keys = [key for key in self.context.operation_priority.keys()]
        self.n_ops = len(self.op_keys)
        self.opt_keys = sorted(list(self.context.operator_copies.keys()), key=lambda op: op.name)
        self.n_opts = len(self.opt_keys)
        xl = np.array([0.0] * self.n_ops + [1] * self.n_opts)
        xu = np.array([1.0] * self.n_ops + [20] * self.n_opts)
        super().__init__(n_var=self.n_ops+self.n_opts, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args, **kwargs) -> None:
        latency_results = []
        area_results = []
        for row in x:
            operation_priorities = row[:self.n_ops]
            for i, op in enumerate(self.context.operation_priority.keys()):
                self.context.operation_priority[op] = operation_priorities[i]

            total_area = 0.0
            operation_counts = np.round(row[self.n_ops:]).astype(int)
            for i, opt in enumerate(self.opt_keys):
                count = operation_counts[i]
                self.context.operator_copies[opt] = count
                total_area += count * self.config.operator_areas[opt.name]
            
            result = schedule(self.context, detailed_table=False)
            latency = result["latency"]

            latency_results.append(latency)
            area_results.append(total_area)
        out["F"] = np.column_stack([latency_results, area_results])

@dataclass
class ProblemConfig:
    """
    Configuration for a scheduling problem, including context,
    now including areas of operators.
    """

    name: str
    context: Context
    reference_point: tuple[float, float] # For hypervolume calculations.
    steps: int = 10000  # Default number of evals.
    max_time: float = 120.0  # Default maximum time in seconds.
    operator_areas: dict[str, float] = field(
        default_factory=lambda: {
            "add": 1.0,
            "mul": 4.0,
            "div": 8.0,
            "sqrt": 8.0,
            "tanh": 20.0,
            "sigmoid": 20.0,
            "max": 1.5,
            "load": 8.0,
        }
    )
    max_opt_count: int = 20  # Maximum count for each operator type.

    def termination(self) -> Termination:
        return TerminateIfAny(
            get_termination("n_evals", self.steps),
            get_termination("time", self.max_time),
        )


@dataclass
class AlgorithmConfig:
    """
    Configuration for an optimization algorithm.
    You should EITHER provide an algorithm instance OR if you want to be fancy,
    provide a function that takes a ProblemConfig and returns an Algorithm instance.
    This allows for adjusting parameters based on the problem configuration.
    """

    name: str
    algorithm: Optional[Algorithm] = None
    seed: int = 42
    lambda_constructor: Optional[Callable[[ProblemConfig], Algorithm]] = None
    verbose: bool = False
    debug: bool = False

    def get_algorithm(self, problem_config: ProblemConfig) -> Algorithm:
        assert (self.algorithm is None) != (
            self.lambda_constructor is None
        ), "You must provide either an algorithm instance or a lambda constructor function."
        if self.lambda_constructor:
            return self.lambda_constructor(problem_config)
        return self.algorithm


@dataclass
class AlgorithmResult:
    """
    Stores the result of running an optimization algorithm.
    Multi Objective this time.
    """

    raw: Any
    optimal_values: np.array  # (n_points, n_objectives)
    runtime: float
    hypervolume: float


problems: dict[str, ProblemConfig] = {
    "LinearAdds": ProblemConfig(
        name="LinearAdds",
        context=create_parallel_linear_adds_graph(depth=20, width=30),
        reference_point=(600, 25),
    ),
    "LayerNorm": ProblemConfig(
        name="LayerNorm",
        context=create_layernorm_graph(N=20),
        reference_point=(250, 250),
    ),
    "NASLSTM": ProblemConfig(
        name="NASLSTM",
        context=create_nas_lstm_graph(N=4, timesteps=2, load_weights=True),
        reference_point=(1400, 600),
    ),
    "Hadamard": ProblemConfig(
        name="Hadamard",
        context=create_hadamard_transform_graph(N=32),
        reference_point=(170, 25),
    ),
    "BinaryTreeAdds": ProblemConfig(
        name="BinaryTreeAdds",
        context=create_parallel_binary_tree_adds_graph(depth=20, width=30),
        reference_point=(600, 25),
    ),
    "DFT": ProblemConfig(
        name="DFT",
        context=create_dft_graph(N=16),
        reference_point=(500, 100),
    ),
    "LSTM": ProblemConfig(
        name="LSTM",
        context=create_lstm_graph(N=3, timesteps=2, load_weights=True),
        reference_point=(400, 500),
    ),
    "PolyEval": ProblemConfig(
        name="PolyEval",
        context=create_parallel_poly_graph(degree=6, width=20),
        reference_point=(700, 100),
    ),
}

algorithms: dict[str, AlgorithmConfig] = {
    "NSGA2": AlgorithmConfig(
        name="NSGA2",
        algorithm=NSGA2(pop_size=100),
    ),
    "NSGA3": AlgorithmConfig(
        name="NSGA3",
        lambda_constructor=lambda config: NSGA3(
            pop_size=100,
            ref_dirs=get_reference_directions("das-dennis", n_dim=2, n_points=100)
        )
    ),
    "SPEA2": AlgorithmConfig(
        name="SPEA2",
        algorithm=SPEA2(pop_size=100),
    ),
}


# Verify that names are consistent.
for problem_name, problem_config in problems.items():
    assert (
        problem_name == problem_config.name
    ), f"Problem name '{problem_name}' does not match config name '{problem_config.name}'."

for algo_name, algo_config in algorithms.items():
    assert (
        algo_name == algo_config.name
    ), f"Algorithm name '{algo_name}' does not match config name '{algo_config.name}'."


def create_results_directory():
    """
    Create a results directory tree: cwd/results_p2/YYYY-MM/DD/HH_MM_SS/
    Returns the full path as a string.
    """
    now = datetime.now()
    base = Path.cwd() / "results_p2"
    year_month = now.strftime("%Y-%m")
    day = now.strftime("%d")
    time_str = now.strftime("%H_%M_%S")
    results_dir = base / year_month / day / time_str
    results_dir.mkdir(parents=True, exist_ok=True)
    return str(results_dir)


def config_and_print_problem_info(problem: ProblemConfig) -> None:
    """
    Configure operator copies in the problem context and print summary information.
    Warns about missing or extra operators in the configuration.
    """
    print(f"=== Problem: {problem.name} ===")
    context = problem.context
    context_opts = [opt.name for opt in context.operator_copies.keys()]

    # Print operation counts
    opt_counts = defaultdict(int)
    for op in context.operation_priority.keys():
        opt_counts[op.operator] += 1
    print("Operation counts:")
    for opt, count in opt_counts.items():
        print(f"  {opt.name:10s}: {count}")
    print(f"Total operations: {sum(opt_counts.values())}")
    print(f"Operators in context: {context_opts}")


def get_pareto_frontier(points: np.ndarray) -> np.ndarray:
    """
    Find the Pareto-optimal points from a set of points.
    Assumes minimization for all objectives.
    """
    pareto_points = []
    for i, p1 in enumerate(points):
        is_dominated = False
        for j, p2 in enumerate(points):
            if i == j:
                continue
            # Check if p2 dominates p1
            if np.all(p2 <= p1) and np.any(p2 < p1):
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append(p1)

    pareto_points = np.array(pareto_points)
    # Sort by first objective (latency) for consistent plotting
    if pareto_points.shape[0] > 0:
        pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]
    return pareto_points


def run_random_sampler(
    problem_config: ProblemConfig, steps_override: Optional[int] = None, seed: int = 42
) -> dict[str, Any]:
    """
    Run random sampling of operation priorities and operator counts.
    Returns the Pareto frontier of (latency, area) points.
    """
    context = problem_config.context
    steps = steps_override if steps_override is not None else problem_config.steps
    rng = np.random.default_rng(seed)

    op_keys = list(context.operation_priority.keys())
    n_ops = len(op_keys)
    opt_keys = sorted(list(context.operator_copies.keys()), key=lambda op: op.name)
    n_opts = len(opt_keys)

    results = []
    for _ in range(steps):
        # Sample priorities
        random_priorities = rng.random(n_ops)
        for i, op in enumerate(op_keys):
            context.operation_priority[op] = random_priorities[i]

        # Sample operator counts
        random_op_counts = rng.integers(
            1, problem_config.max_opt_count + 1, size=n_opts
        )
        area = 0.0
        for i, opt in enumerate(opt_keys):
            count = random_op_counts[i]
            context.operator_copies[opt] = count
            area += count * problem_config.operator_areas[opt.name]

        result = schedule(context, detailed_table=False)
        latency = result["latency"]
        results.append([latency, area])

    results = np.array(results)
    pareto_frontier = get_pareto_frontier(results)

    return {
        "all_points": results,
        "pareto_frontier": pareto_frontier,
        "hypervolume": hv(F=pareto_frontier, ref_point=problem_config.reference_point),
    }


def run_algorithm(
    problem_config: ProblemConfig, algorithm_config: AlgorithmConfig
) -> AlgorithmResult:
    """
    Run the optimization algorithm on the given problem configuration.
    Returns an AlgorithmResult with the Pareto front and runtime.
    """
    problem = AreaDelayMinimizationProblem(problem_config)
    algorithm = algorithm_config.get_algorithm(problem_config)
    start = time.time()
    minimize_result = minimize(
        problem,
        algorithm,
        termination=problem_config.termination(),
        seed=algorithm_config.seed,
        save_history=False,
        verbose=algorithm_config.verbose,
    )
    total_time = time.time() - start

    pareto_front = minimize_result.F
    reference_point = problem_config.reference_point
    if pareto_front.shape[0] > 0:
        hypervolume_value = hv(F=pareto_front, ref_point=reference_point)
    else:
        hypervolume_value = 0.0

    return AlgorithmResult(
        raw=minimize_result,
        optimal_values=pareto_front,
        runtime=total_time,
        hypervolume=hypervolume_value,
    )


def plot_results_single(
    title: str,
    algorithm_config: AlgorithmConfig,
    problem_config: ProblemConfig,
    result: AlgorithmResult,
    results_dir: str,
) -> None:
    """
    Create a plot for a single algorithm on a single problem (Pareto front).

    Args:
        title: Title for the plot and filename
        algorithm_config: Configuration of the algorithm
        problem_config: Configuration of the problem
        result: Result of running the algorithm
        results_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))

    pareto_front = result.optimal_values
    if pareto_front.shape[0] > 0:
        pareto_front = pareto_front[np.argsort(pareto_front[:, 0])]
        plt.plot(
            pareto_front[:, 0],  # Latency
            pareto_front[:, 1],  # Area
            "o-",  # Line with markers
            label=f"{algorithm_config.name} Pareto Front",
        )

    plt.title(f"{title} - {algorithm_config.name} on {problem_config.name}")
    plt.xlabel("Latency (cycles)")
    plt.ylabel("Area")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = os.path.join(results_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {filename}")


def plot_results_comparing_algorithms(
    title: str,
    problem_config: ProblemConfig,
    algorithm_results: dict[str, AlgorithmResult],
    random_result: dict[str, Any] = None,
    results_dir: str = "results_p2",
) -> None:
    """
    Create a plot comparing multiple algorithms on a single problem.

    Args:
        title: Title for the plot and filename
        problem_config: Configuration of the problem
        algorithm_results: Dictionary mapping algorithm names to their results
        random_result: Results from random sampling
        results_dir: Directory to save the plot
    """
    plt.figure(figsize=(16, 10))

    # Plot random sampling pareto front if provided
    if random_result is not None:
        random_pareto = random_result["pareto_frontier"]
        if random_pareto.shape[0] > 0:
            plt.scatter(
                random_pareto[:, 0],
                random_pareto[:, 1],
                label="Random Sampling Pareto Front",
                alpha=0.7,
                marker="x",
            )

    # Plot algorithm results
    for algo_name, result in algorithm_results.items():
        pareto_front = result.optimal_values
        if pareto_front.shape[0] > 0:
            plt.scatter(
                pareto_front[:, 0],  # Latency
                pareto_front[:, 1],  # Area
                label=algo_name,
                alpha=0.8,
            )

    plt.title(f"{title} - Algorithm Comparison on {problem_config.name}")
    plt.xlabel("Latency (cycles)")
    plt.ylabel("Area")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    filename = os.path.join(results_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {filename}")


def create_comparison_tables(
    all_results: dict[str, dict[str, AlgorithmResult]],
    random_results: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create comparison tables for algorithms across problems.
    Reports Hypervolume, Ranks (higher is better), and Runtimes.

    Args:
        all_results: {problem_name: {algorithm_name: result}}
        random_results: {problem_name: {result_key: value}}

    Returns:
        Tuple of DataFrames: (hypervolume_table, rank_table, runtime_table)
    """
    problem_names = list(all_results.keys())
    if not problem_names:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    algorithm_names = list(all_results[problem_names[0]].keys())
    all_methods = algorithm_names + ["Random"]

    # --- Hypervolume Table ---
    hv_data = []
    for problem in problem_names:
        row = {"Problem": problem}
        for algo in algorithm_names:
            row[algo] = all_results[problem][algo].hypervolume
        row["Random"] = random_results[problem]["hypervolume"]
        hv_data.append(row)

    hv_df = pd.DataFrame(hv_data)
    hv_df = hv_df[["Problem"] + all_methods]

    # --- Rank Table (higher hypervolume is better) ---
    rank_data = []
    for problem in problem_names:
        values_to_rank = {}
        for algo in algorithm_names:
            values_to_rank[algo] = all_results[problem][algo].hypervolume
        values_to_rank["Random"] = random_results[problem]["hypervolume"]

        # Sort by value (descending) and assign ranks
        sorted_items = sorted(values_to_rank.items(), key=lambda x: x[1], reverse=True)
        ranks = {}

        # Group items by value to handle ties
        value_groups = {}
        for name, value in sorted_items:
            if value not in value_groups:
                value_groups[value] = []
            value_groups[value].append(name)

        # Assign average ranks for ties
        current_position = 1
        for value in sorted(value_groups.keys(), reverse=True):
            group = value_groups[value]
            group_size = len(group)
            avg_rank = current_position + (group_size - 1) / 2
            for name in group:
                ranks[name] = avg_rank
            current_position += group_size

        row = {"Problem": problem}
        for method in all_methods:
            row[method] = ranks[method]
        rank_data.append(row)

    rank_df = pd.DataFrame(rank_data)
    rank_df = rank_df[["Problem"] + all_methods]

    # Add average rank row
    avg_ranks = rank_df.iloc[:, 1:].mean().tolist()
    rank_df.loc[len(rank_df)] = ["Average Finish"] + avg_ranks

    # --- Runtime Table ---
    runtime_data = []
    for problem in problem_names:
        row = {"Problem": problem}
        for algo in algorithm_names:
            row[algo] = all_results[problem][algo].runtime
        runtime_data.append(row)

    runtime_df = pd.DataFrame(runtime_data)
    runtime_df = runtime_df[["Problem"] + algorithm_names]

    return hv_df, rank_df, runtime_df


def runtime_format(runtime: float) -> str:
    return f"{runtime:.2f}"


def hypervolume_format(hypervolume: float) -> str:
    return f"{hypervolume:.2f}".rstrip("0").rstrip(".")


def rank_format(rank: float) -> str:
    return f"{rank:.2f}".rstrip("0").rstrip(".")


def main():
    """
    Run all algorithms on all problems, generate plots and comparison tables.
    """
    print("Starting multi-objective optimization experiments...")

    # Create results directory tree
    results_dir = create_results_directory()

    # Dictionary to store all results
    all_results = {}
    all_random_results = {}

    # Run each problem with each algorithm
    for problem_name, problem_config in tqdm.tqdm(problems.items(), desc="Problems"):
        print(f"\nRunning problem: {problem_name}")

        # Configure the problem
        config_and_print_problem_info(problem_config)

        # Run random sampler
        random_result = run_random_sampler(problem_config)
        all_random_results[problem_name] = random_result
        if random_result["pareto_frontier"].shape[0] > 0:
            print(
                f"Random sampler found {len(random_result['pareto_frontier'])} Pareto points."
            )
        else:
            print("Random sampler found no Pareto points.")

        # Dictionary to store results for this problem
        problem_results = {}

        # Run each algorithm on this problem
        for algo_name, algo_config in tqdm.tqdm(algorithms.items(), desc="Algorithms"):
            print(f"  Running {algo_name}...")
            result = run_algorithm(problem_config, algo_config)
            problem_results[algo_name] = result
            print(f"    Found {len(result.optimal_values)} Pareto points.")
            print(f"    Runtime: {result.runtime:.2f}s")

            # Create individual plot
            plot_results_single(
                f"{problem_name}_{algo_name}",
                algo_config,
                problem_config,
                result,
                results_dir,
            )

        # Create comparison plot for this problem
        plot_results_comparing_algorithms(
            f"{problem_name}_Comparison",
            problem_config,
            problem_results,
            random_result,
            results_dir,
        )

        # Store results for this problem
        all_results[problem_name] = problem_results

    # Create comparison tables
    print("\nCreating comparison tables...")
    hv_table, rank_table, runtime_table = create_comparison_tables(
        all_results, all_random_results
    )

    # Save tables to CSV
    hv_table.to_csv(
        os.path.join(results_dir, "hypervolume_comparison.csv"), index=False
    )
    rank_table.to_csv(os.path.join(results_dir, "rank_comparison.csv"), index=False)
    runtime_table.to_csv(
        os.path.join(results_dir, "runtime_comparison.csv"), index=False
    )

    # Save tables as HTML in a Markdown file
    md_path = os.path.join(results_dir, "comparison_tables.md")
    with open(md_path, "w") as f:
        f.write("# Hypervolume Comparison\n\n")
        f.write(hv_table.to_html(index=False, border=1, float_format=hypervolume_format))
        f.write("\n\n# Rank Comparison (based on Hypervolume)\n\n")
        f.write(rank_table.to_html(index=False, border=1, float_format=rank_format))
        f.write("\n\n# Runtime Comparison (s)\n\n")
        f.write(
            runtime_table.to_html(index=False, border=1, float_format=runtime_format)
        )
    print(f"\nMarkdown file with HTML tables saved to {md_path}")

    # Print tables
    print("\n=== Hypervolume Comparison ===")
    print(hv_table.to_string())
    print("\n=== Rank Comparison (based on Hypervolume) ===")
    print(rank_table.to_string())
    print("\n=== Runtime Comparison (s) ===")
    print(runtime_table.to_string())

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
