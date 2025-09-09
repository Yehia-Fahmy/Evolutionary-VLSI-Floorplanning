import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Callable
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
from schedule import schedule, asap_schedule, alap_schedule, Context
# from test_scheduling import HistoryCallback, LatencyMinimizationProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
# from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.core.algorithm import Algorithm
from pymoo.termination import get_termination
from pymoo.core.termination import TerminateIfAny, Termination
import tqdm
import time
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime


class HistoryCallback(Callback):
    """
    Callback to record the history of evaluations and best values during optimization.
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


class LatencyMinimizationProblem(Problem):
    """
    Pymoo Problem class for minimizing latency by optimizing operation priorities.
    """

    def __init__(self, config: "ProblemConfig") -> None:
        self.config = config
        self.context = self.config.context
        super().__init__(n_var=len(self.context.operation_priority), n_obj=1, xl=0, xu=1)

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args, **kwargs) -> None:
        results = []
        for operation_priorities in x:
            for i, op in enumerate(self.context.operation_priority.keys()):
                self.context.operation_priority[op] = operation_priorities[i]
            schedule_result = schedule(self.context, detailed_table=False)
            results.append(schedule_result["latency"])
        out["F"] = np.array(results).reshape(-1, 1)

@dataclass
class ProblemConfig:
    """
    Configuration for a scheduling problem, including context and operator settings.
    """

    name: str
    context: Context
    operators: dict[str, int]
    steps: int = 5000  # Default number of evals.
    max_time: float = 120.0  # Default maximum time in seconds.

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
    """

    raw: Any
    optimal_value: float
    n_evals: list[float]
    best_values: list[float]
    all_values: list[Any]
    runtime: float


problems: dict[str, ProblemConfig] = {
    "LinearAdds": ProblemConfig(
        name="LinearAdds",
        context=create_parallel_linear_adds_graph(depth=20, width=30),
        operators={"add": 8},
    ),
    "LayerNorm": ProblemConfig(
        name="LayerNorm",
        context=create_layernorm_graph(N=20),
        operators={"add": 2, "mul": 2, "div": 1, "sqrt": 1},
    ),
    "NASLSTM": ProblemConfig(
        name="NASLSTM",
        context=create_nas_lstm_graph(N=4, timesteps=2, load_weights=True),
        operators={"add": 2, "mul": 2, "tanh": 1, "sigmoid": 1, "max": 1, "load": 2},
    ),
    "Hadamard": ProblemConfig(
        name="Hadamard",
        context=create_hadamard_transform_graph(N=32),
        operators={"add": 2},
    ),
    "BinaryTreeAdds": ProblemConfig(
        name="BinaryTreeAdds",
        context=create_parallel_binary_tree_adds_graph(depth=20, width=30),
        operators={"add": 8},
    ),
    "DFT": ProblemConfig(
        name="DFT",
        context=create_dft_graph(N=16),
        operators={"add": 2, "mul": 2},
    ),
    "LSTM": ProblemConfig(
        name="LSTM",
        context=create_lstm_graph(N=3, timesteps=2, load_weights=True),
        operators={"add": 2, "mul": 2, "tanh": 1, "sigmoid": 1, "load": 2},
    ),
    "PolyEval": ProblemConfig(
        name="PolyEval",
        context=create_parallel_poly_graph(degree=6, width=20),
        operators={"add": 2, "mul": 2},
    ),
    # You can add more problems if you want.
    # will just pull your "algorithms" dict for testing.
}

algorithms: dict[str, AlgorithmConfig] = {
    "DifferentialEvolution": AlgorithmConfig(
        name="DifferentialEvolution",
        algorithm=DE(),
    ),
    "CMAES": AlgorithmConfig(
        name="CMAES",
        algorithm=CMAES(),
    ),
    "GeneticAlgorithm": AlgorithmConfig(
        name="GeneticAlgorithm",
        algorithm=GA(),
    ),
    "NelderMead": AlgorithmConfig(
        name="NelderMead",
        algorithm=NelderMead(),
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


def config_and_print_problem_info(problem: ProblemConfig) -> None:
    """
    Configure operator copies in the problem context and print summary information.
    Warns about missing or extra operators in the configuration.
    """
    print(f"=== Problem: {problem.name} ===")
    context = problem.context
    operator_copies = problem.operators
    # Set operator copies, warn if operator is missing or extra
    context_ops = set(op.name for op in context.operator_copies.keys())
    config_ops = set(operator_copies.keys())
    missing_ops = context_ops - config_ops
    extra_ops = config_ops - context_ops
    # Set operator copies from config, warn if not present in context
    for op in operator_copies:
        found = False
        for op_obj in context.operator_copies.keys():
            if op_obj.name == op:
                context.operator_copies[op_obj] = operator_copies[op]
                found = True
        if not found:
            print(f"WARNING: Operator '{op}' in config not found in context.")
    # Set missing operator copies to 1
    for op in missing_ops:
        print(f"WARNING: Operator '{op}' missing from config, setting copies to 1.")
        for op_obj in context.operator_copies.keys():
            if op_obj.name == op:
                context.operator_copies[op_obj] = 1
    if extra_ops:
        print(f"WARNING: Operators in config but not in context: {extra_ops}")
    # Print operator counts
    print("Operator copies:")
    for op_obj, count in context.operator_copies.items():
        print(f"  {op_obj.name:10s}: {count}")
    # Print operation counts
    op_counts = defaultdict(int)
    for op in context.operation_priority.keys():
        op_counts[op.operator.name] += 1
    print("Operation counts:")
    for opname, count in op_counts.items():
        print(f"  {opname:10s}: {count}")
    print(f"Total operations: {sum(op_counts.values())}")
    print(f"Operators in context: {sorted(context_ops)}")
    print(f"Operators in config:  {sorted(config_ops)}")


def get_baselines(problem: ProblemConfig) -> dict[str, float]:
    """
    Compute baseline latencies for the problem using zero priorities,
    criticality-based priorities, and reversed criticality.
    Returns a dictionary of baseline latencies.
    """
    context = problem.context
    for op in context.operation_priority.keys():
        context.operation_priority[op] = 0.0
    baseline_zero = schedule(context, detailed_table=False)["latency"]

    asap_result = asap_schedule(context)
    alap_result = alap_schedule(context, end_time=asap_result["latency"])
    criticality = {}
    for op in context.operation_priority.keys():
        criticality[op] = (
            alap_result["operation_start_times"][op]
            - asap_result["operation_start_times"][op]
        )
    max_criticality = max(criticality.values()) if criticality.values() else 1.0
    for op in criticality:
        criticality[op] /= max_criticality if max_criticality else 1.0
    for op in context.operation_priority.keys():
        context.operation_priority[op] = criticality[op]
    baseline_crit = schedule(context, detailed_table=False)["latency"]
    for op in context.operation_priority.keys():
        context.operation_priority[op] = 1.0 - criticality[op]
    baseline_rev = schedule(context, detailed_table=False)["latency"]
    return {
        "zeros": baseline_zero,
        "criticality": baseline_crit,
        "reversed_criticality": baseline_rev,
    }


def run_random_sampler(
    problem_config: ProblemConfig, steps_override: Optional[int] = None, seed: int = 42
) -> dict[str, Any]:
    """
    Run random sampling of operation priorities for the given problem configuration.
    Returns statistics and running minimum latency.
    """
    context = problem_config.context
    steps = steps_override if steps_override is not None else problem_config.steps
    rng = np.random.default_rng(seed)
    latencies = []
    for _ in range(steps):
        random_priorities = rng.random(len(context.operation_priority))
        for i, op in enumerate(context.operation_priority.keys()):
            context.operation_priority[op] = random_priorities[i]
        result = schedule(context, detailed_table=False)
        latencies.append(result["latency"])

    latencies = np.array(latencies)
    running_min = np.minimum.accumulate(latencies)
    return {
        "latencies": latencies,
        "running_min": running_min,
        "n_evals": np.arange(1, steps + 1),
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "min": float(running_min[-1]),
    }


def run_algorithm(
    problem_config: ProblemConfig, algorithm_config: AlgorithmConfig
) -> AlgorithmResult:
    """
    Run the optimization algorithm on the given problem configuration using the specified algorithm configuration.
    Returns an AlgorithmResult with optimization history and statistics.
    """
    problem = LatencyMinimizationProblem(problem_config)
    algorithm = algorithm_config.get_algorithm(problem_config)
    callback = HistoryCallback()
    start = time.time()
    result = minimize(
        problem,
        algorithm,
        termination=problem_config.termination(),
        seed=algorithm_config.seed,
        callback=callback,
        save_history=False,
        verbose=algorithm_config.verbose,
    )
    total_time = time.time() - start
    return AlgorithmResult(
        raw=result,
        optimal_value=float(result.F),
        n_evals=callback.n_evals,
        best_values=callback.best_values,
        all_values=callback.all_values,
        runtime=total_time,
    )


def verify_problems() -> None:
    """
    Verify that all problems are correctly configured and can be scheduled.
    Runs baseline and random sampling for each problem.
    """
    for problem in problems.values():
        config_and_print_problem_info(problem)
        _ = get_baselines(problem)
        _ = run_random_sampler(problem, steps_override=100)
    print("All problems verified successfully.")


def verify_algorithms() -> None:
    """
    Verify that all algorithms can be instantiated and run on a sample problem.
    Runs each algorithm on the first problem in the problems dictionary.
    """
    sample_problem = ProblemConfig(
        name="SampleProblem",
        context=create_parallel_linear_adds_graph(depth=5, width=5),
        operators={"add": 2},
        steps=100,
    )
    for algo_name, algo_config in algorithms.items():
        print(f"Verifying algorithm: {algo_name}")
        algo = algo_config.get_algorithm(sample_problem)
        if not isinstance(algo, Algorithm):
            raise ValueError(
                f"Algorithm '{algo_name}' did not return a valid Algorithm instance."
            )
        try:
            result = run_algorithm(sample_problem, algo_config)
            print(
                f"  {algo_name} ran successfully with optimal value: {result.optimal_value}"
            )
        except Exception as e:
            print(f"  Error running {algo_name}: {e}")


def plot_results_single(
    title: str,
    algorithm_config: AlgorithmConfig,
    problem_config: ProblemConfig,
    result: AlgorithmResult,
    results_dir: str,
) -> None:
    """
    Create a plot for a single algorithm on a single problem.

    Args:
        title: Title for the plot and filename
        algorithm_config: Configuration of the algorithm
        problem_config: Configuration of the problem
        result: Result of running the algorithm
        results_dir: Directory to save the plot
    """
    plt.figure(figsize=(14, 8))

    # Plot best values over n_evals
    plt.plot(result.n_evals, result.best_values, "b-", label="Best Latency")

    # Plot all values as scatter plot
    all_evals = []
    all_values = []
    for i, values in enumerate(result.all_values):
        evals = [result.n_evals[i]] * len(values)
        all_evals.extend(evals)
        all_values.extend(values)
    plt.scatter(all_evals, all_values, alpha=0.2, color="gray", label="All Evaluations")

    plt.title(f"{title} - {algorithm_config.name} on {problem_config.name}")
    plt.xlabel("Number of Evaluations")
    plt.ylabel("Latency")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    filename = os.path.join(results_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {filename}")


def plot_results_comparing_algorithms(
    title: str,
    problem_config: ProblemConfig,
    algorithm_results: Dict[str, AlgorithmResult],
    baselines: Dict[str, float] = None,
    random_result: Dict[str, Any] = None,
    results_dir: str = "results",
) -> None:
    """
    Create a plot comparing multiple algorithms on a single problem.

    Args:
        title: Title for the plot and filename
        problem_config: Configuration of the problem
        algorithm_results: Dictionary mapping algorithm names to their results
        baselines: Dictionary of baseline latencies
        random_result: Results from random sampling
        results_dir: Directory to save the plot
    """
    plt.figure(figsize=(16, 10))

    # Plot algorithm results
    for algo_name, result in algorithm_results.items():
        plt.plot(result.n_evals, result.best_values, label=algo_name)

    # Plot random sampling if provided
    if random_result is not None:
        plt.plot(
            random_result["n_evals"],
            random_result["running_min"],
            label="Random Sampling",
            linestyle="--",
        )
        # Plot random mean as a solid red line
        plt.axhline(
            y=random_result["mean"],
            color="red",
            linestyle="-",
            label=f"Random Mean ({random_result['mean']:.2f})",
            linewidth=2,
        )
        # Plot random mean + std and mean - std as dotted, fainter red lines
        plt.axhline(
            y=random_result["mean"] + random_result["std"],
            color="red",
            linestyle="dotted",
            linewidth=1,
            alpha=0.5,
            label=f"Random Mean ± Std ({(random_result['mean'] + random_result['std']):.2f})",
        )
        plt.axhline(
            y=random_result["mean"] - random_result["std"],
            color="red",
            linestyle="dotted",
            linewidth=1,
            alpha=0.5,
        )

    # Plot baselines if provided
    if baselines is not None:
        for baseline_name, baseline_value in baselines.items():
            plt.axhline(
                y=baseline_value,
                linestyle="dashdot",
                label=f"Baseline: {baseline_name} ({baseline_value:.2f})",
            )

    plt.title(f"{title} - Algorithm Comparison on {problem_config.name}")
    plt.xlabel("Number of Evaluations")
    plt.ylabel("Latency")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    filename = os.path.join(results_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {filename}")


def create_comparison_tables(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    baselines: Dict[str, Dict[str, float]],
    random_results: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create comparison tables for algorithms across problems.

    Args:
        all_results: Dictionary mapping problem names to dictionaries of algorithm results
                     {problem_name: {algorithm_name: result}}
        baselines: Dictionary mapping problem names to baseline results
                   {problem_name: {baseline_name: value}}
        random_results: Dictionary mapping problem names to random sampling results
                       {problem_name: {result_key: value}}

    Returns:
        Tuple of DataFrames: (latency_table, rank_table, runtime_table)
    """
    # Extract problem names and algorithm names
    problem_names = list(all_results.keys())
    algorithm_names = list(all_results[problem_names[0]].keys())

    # Create DataFrame for best latencies
    latency_data = []
    for problem in problem_names:
        row = [problem]
        # Add algorithm results
        for algo in algorithm_names:
            row.append(all_results[problem][algo].optimal_value)

        # Add baseline results
        for baseline_name in ["zeros", "criticality", "reversed_criticality"]:
            row.append(baselines[problem][baseline_name])

        # Add random results
        row.append(random_results[problem]["min"])
        row.append(random_results[problem]["mean"])
        row.append(random_results[problem]["std"])

        latency_data.append(row)

    # Create column names
    algo_cols = algorithm_names
    baseline_cols = [
        "Baseline_Zeros",
        "Baseline_Criticality",
        "Baseline_RevCriticality",
    ]
    random_cols = ["Random_Min", "Random_Mean", "Random_Std"]
    all_cols = ["Problem"] + algo_cols + baseline_cols + random_cols

    latency_df = pd.DataFrame(latency_data, columns=all_cols)

    # Create DataFrame for rankings (lower latency is better)
    rank_data = []
    for problem in problem_names:
        values_to_rank = {}
        # Add algorithm results
        for algo in algorithm_names:
            values_to_rank[algo] = all_results[problem][algo].optimal_value

        # Add baseline results
        for baseline_name in ["zeros", "criticality", "reversed_criticality"]:
            values_to_rank[f"Baseline_{baseline_name.capitalize()}"] = baselines[
                problem
            ][baseline_name]

        # Add random min
        values_to_rank["Random_Min"] = random_results[problem]["min"]

        # Sort by value and assign ranks (handling ties with average ranking)
        sorted_items = sorted(values_to_rank.items(), key=lambda x: x[1])
        ranks = {}

        # Group items by value to handle ties
        value_groups = {}
        for name, value in sorted_items:
            if value not in value_groups:
                value_groups[value] = []
            value_groups[value].append(name)

        # Assign average ranks for ties
        current_position = 1
        for value in sorted(value_groups.keys()):
            group = value_groups[value]
            group_size = len(group)
            # Calculate average rank for this group
            avg_rank = current_position + (group_size - 1) / 2
            for name in group:
                ranks[name] = avg_rank
            current_position += group_size

        row = [problem]
        # Add algorithm rankings
        for algo in algorithm_names:
            row.append(ranks[algo])

        # Add baseline rankings
        for baseline_name in ["zeros", "criticality", "reversed_criticality"]:
            row.append(ranks[f"Baseline_{baseline_name.capitalize()}"])

        # Add random min ranking
        row.append(ranks["Random_Min"])

        rank_data.append(row)

    rank_cols = (
        ["Problem"] + algo_cols + [col for col in baseline_cols] + ["Random_Min"]
    )
    rank_df = pd.DataFrame(rank_data, columns=rank_cols)

    # Add average rank row
    avg_ranks = rank_df.iloc[:, 1:].mean().tolist()
    rank_df.loc[len(rank_df)] = ["Average Finish"] + avg_ranks

    # Create runtime DataFrame
    runtime_data = []
    for problem in problem_names:
        row = [problem]
        # Add algorithm runtimes
        for algo in algorithm_names:
            row.append(all_results[problem][algo].runtime)
        runtime_data.append(row)

    runtime_cols = ["Problem"] + algo_cols
    runtime_df = pd.DataFrame(runtime_data, columns=runtime_cols)
    return latency_df, rank_df, runtime_df


def create_results_directory():
    """
    Create a results directory tree: cwd/results/YYYY-MM/DD/HH_MM_SS/
    Returns the full path as a string.
    """

    now = datetime.now()
    base = Path.cwd() / "results"
    year_month = now.strftime("%Y-%m")
    day = now.strftime("%d")
    time_str = now.strftime("%H_%M_%S")
    results_dir = base / year_month / day / time_str
    results_dir.mkdir(parents=True, exist_ok=True)
    return str(results_dir)


def runtime_format(runtime: float) -> str:
    return f"{runtime:.2f}"


def latency_format(latency: float) -> str:
    return f"{latency:.2f}".rstrip("0").rstrip(".")


def rank_format(rank: float) -> str:
    return f"{rank:.2f}".rstrip("0").rstrip(".")


def main():
    """
    Run all algorithms on all problems, generate plots and comparison tables.
    """
    print("Starting optimization experiments...")

    # Create results directory tree
    results_dir = create_results_directory()

    # Dictionary to store all results
    all_results = {}
    all_baselines = {}
    all_random_results = {}

    # Run each problem with each algorithm
    for problem_name, problem_config in tqdm.tqdm(problems.items(), desc="Problems"):
        print(f"\nRunning problem: {problem_name}")

        # Configure the problem
        config_and_print_problem_info(problem_config)

        # Get baselines
        baselines = get_baselines(problem_config)
        print(f"Baselines: {baselines}")
        all_baselines[problem_name] = baselines

        # Run random sampler
        random_result = run_random_sampler(problem_config)
        print(f"Random sampler min latency: {random_result['min']}")
        all_random_results[problem_name] = random_result

        # Dictionary to store results for this problem
        problem_results = {}

        # Run each algorithm on this problem
        for algo_name, algo_config in tqdm.tqdm(algorithms.items(), desc="Algorithms"):
            print(f"  Running {algo_name}...")
            result = run_algorithm(problem_config, algo_config)
            problem_results[algo_name] = result
            print(f"    Optimal latency: {result.optimal_value}")
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
            baselines,
            random_result,
            results_dir,
        )

        # Store results for this problem
        all_results[problem_name] = problem_results

    # Create comparison tables
    print("\nCreating comparison tables...")
    latency_table, rank_table, runtime_table = create_comparison_tables(
        all_results, all_baselines, all_random_results
    )

    # Save tables to CSV
    latency_table.to_csv(
        os.path.join(results_dir, "latency_comparison.csv"), index=False
    )
    rank_table.to_csv(os.path.join(results_dir, "rank_comparison.csv"), index=False)
    runtime_table.to_csv(
        os.path.join(results_dir, "runtime_comparison.csv"), index=False
    )
    # Save tables as HTML in a Markdown file
    md_path = os.path.join(results_dir, "comparison_tables.md")
    with open(md_path, "w") as f:
        f.write("# Latency Comparison Table (cycles)\n\n")
        f.write(
            latency_table.to_html(index=False, border=1, float_format=latency_format)
        )
        f.write("\n\n# Rank Comparison Table\n\n")
        f.write(rank_table.to_html(index=False, border=1, float_format=rank_format))
        f.write("\n\n# Runtime Comparison Table (s)\n\n")
        f.write(
            runtime_table.to_html(index=False, border=1, float_format=runtime_format)
        )
    print(f"\nMarkdown file with HTML tables saved to {md_path}")

    # Print tables
    print("\n=== Latency Comparison Table (cycles) ===")
    print(latency_table)
    print("\n=== Rank Comparison Table ===")
    print(rank_table)
    print("\n=== Runtime Comparison Table (s) ===")
    print(runtime_table)

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    # Debugging line to verify problems
    # verify_problems()
    # Debugging line to verify algorithms
    # verify_algorithms()

    main()
