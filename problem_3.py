"""
Problem 3: Evolutionary Floorplanning Experiments

This script evaluates an evolutionary floorplanning approach on:
- Synthetic benchmarks (random rectangles with optional structured nets)
- MCNC benchmarks `ami33` and `ami49` (YAL format included in the repo)

Objectives:
- Minimize area (bounding rectangle area)
- Minimize HPWL (wirelength) when nets are known
- Minimize overlap (should be zero for valid placements)

Notes:
- For MCNC YAL files, only hard block dimensions are parsed. Pin-level netlists
  in YAL are non-trivial to parse robustly, so wirelength on MCNC defaults to 0.
  This matches the common practice in older literature focusing on area for
  these two benchmarks (e.g., Tang & Yao). You can enhance the parser later to
  compute HPWL from nets if desired.

Outputs:
- A results tree under `results_p3/YYYY-MM/DD/HH_MM_SS/` containing CSV tables
  with per-benchmark statistics (best/mean/std) and algorithm comparisons.

Dependencies: numpy, pandas, scipy, pymoo, matplotlib (already in requirements)
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import kruskal

from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination import get_termination
from pymoo.core.termination import TerminateIfAny, Termination
from pymoo.core.evaluator import Evaluator
from pymoo.core.sampling import Sampling
from pymoo.core.callback import Callback
import matplotlib.pyplot as plt


# ---------------------------- Data structures -----------------------------


@dataclass
class Module:
    name: str
    width: float
    height: float


@dataclass
class Net:
    modules: List[int]  # indices into the modules list


@dataclass
class Floorplan:
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    h: np.ndarray

    def bbox(self) -> Tuple[float, float]:
        total_w = float(np.max(self.x + self.w)) if len(self.x) else 0.0
        total_h = float(np.max(self.y + self.h)) if len(self.y) else 0.0
        return total_w, total_h

    def area(self) -> float:
        total_w, total_h = self.bbox()
        return total_w * total_h

    def deadspace(self) -> float:
        area_bbox = self.area()
        area_blocks = float(np.sum(self.w * self.h))
        return max(0.0, area_bbox - area_blocks)

    def hpwl(self, nets: List[Net]) -> float:
        if not nets:
            return 0.0
        centers_x = self.x + 0.5 * self.w
        centers_y = self.y + 0.5 * self.h
        wl = 0.0
        for net in nets:
            idx = net.modules
            xs = centers_x[idx]
            ys = centers_y[idx]
            wl += (np.max(xs) - np.min(xs)) + (np.max(ys) - np.min(ys))
        return float(wl)


# ----------------------------- YAL parser ---------------------------------


def parse_yal_modules(yal_path: str) -> List[Module]:
    """
    Parse the YAL file and extract module names and dimensions.

    Observed DIMENSIONS format in provided files:
      DIMENSIONS <W> 0 <W> <H> 0 <H> 0 0;
    We take width=W, height=H. Skip the `bound` parent module.
    """
    modules: List[Module] = []
    current_name: Optional[str] = None
    in_module = False
    skip_module = False

    with open(yal_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("MODULE "):
                # Close previous
                in_module = True
                parts = line.split()
                current_name = parts[1].strip(";")
                skip_module = current_name.lower() == "bound"
            elif in_module and line.startswith("TYPE "):
                # If type is not GENERAL we still allow, but in these files GENERAL is fine.
                pass
            elif in_module and line.startswith("DIMENSIONS ") and not skip_module:
                tokens = line.replace("DIMENSIONS", "").replace(";", "").split()
                try:
                    nums = [float(t) for t in tokens]
                    # Defensive: prefer 1st as W and 4th as H if present
                    width = nums[0]
                    height = nums[3] if len(nums) >= 4 else nums[1]
                    assert width > 0 and height > 0
                    assert current_name is not None
                    modules.append(Module(current_name, width, height))
                except Exception:
                    # Skip malformed entries
                    pass
            elif in_module and line.startswith("ENDMODULE"):
                in_module = False
                current_name = None
                skip_module = False

    return modules


# Extract net connectivity from a YAL file by co-occurrence of module instances
# on the same signal in the NETWORK section. Power/ground and obvious pin labels
# are filtered. Returns a list of Net objects where each net contains indices
# into the modules list (by module type name).
def parse_yal_nets(yal_path: str, modules: List[Module]) -> List[Net]:
    name_to_idx: Dict[str, int] = {m.name: i for i, m in enumerate(modules)}
    nets_map: Dict[str, set] = {}
    in_network = False
    with open(yal_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("NETWORK"):
                in_network = True
                continue
            if in_network and line.startswith("ENDNETWORK"):
                break
            if not in_network:
                continue
            if not line:
                continue
            # Lines are like: C_0 bk9d GND P2G P2F ... ; possibly spanning lines
            # Remove trailing semicolon and split
            line_no_semicolon = line.replace(";", "")
            tokens = line_no_semicolon.split()
            if len(tokens) < 2:
                continue
            # tokens[0] is instance name (e.g., C_0), tokens[1] is module type
            mod_type = tokens[1]
            if mod_type not in name_to_idx:
                continue
            mod_idx = name_to_idx[mod_type]
            # Remaining tokens are signal labels; filter out known power/pad labels
            for t in tokens[2:]:
                tu = t.upper()
                if tu in {"GND", "POW", "VDD", "VSS"}:
                    continue
                if tu.startswith("P") or tu.startswith("C"):
                    # filter pad/pin indices like P12, C30, etc.
                    continue
                if t.isdigit():
                    # numeric pin ids
                    continue
                # keep as net label
                if t not in nets_map:
                    nets_map[t] = set()
                nets_map[t].add(mod_idx)

    nets: List[Net] = []
    for net, mods in nets_map.items():
        if len(mods) >= 2 and len(mods) <= 10:
            nets.append(Net(sorted(list(mods))))
    return nets


def parse_yal_outline(yal_path: str) -> Optional[Tuple[float, float]]:
    """Return (W, H) of the parent bound module if present, else None."""
    in_module = False
    current_name = None
    with open(yal_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("MODULE "):
                in_module = True
                current_name = line.split()[1].strip(";")
            elif in_module and line.startswith("DIMENSIONS ") and current_name and current_name.lower() == "bound":
                tokens = line.replace("DIMENSIONS", "").replace(";", "").split()
                try:
                    nums = [float(t) for t in tokens]
                    width = nums[0]
                    height = nums[3] if len(nums) >= 4 else nums[1]
                    return (width, height)
                except Exception:
                    return None
            elif in_module and line.startswith("ENDMODULE"):
                in_module = False
                current_name = None
    return None

# ------------------------ Sequence Pair Placement -------------------------


def longest_paths(weights: np.ndarray, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Compute longest path distances in a DAG given by edges i->j with weight=weights[i].
    Topological order will be assumed from node indices in inputs of `edges`.
    """
    n = len(weights)
    dist = np.zeros(n, dtype=float)
    # Simple relaxation in topological order by source index
    # For robustness, build adjacency list and indegrees and do Kahn's algo.
    adj: List[List[int]] = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=int)
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    # Kahn queue
    queue = [i for i in range(n) if indeg[i] == 0]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u]:
            cand = dist[u] + weights[u]
            if cand > dist[v]:
                dist[v] = cand
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return dist


def place_sequence_pair(
    widths: np.ndarray,
    heights: np.ndarray,
    seq_a: np.ndarray,
    seq_b: np.ndarray,
) -> Floorplan:
    """
    Compute a placement using the sequence-pair representation.
    seq_a and seq_b are permutations (arrays of indices 0..N-1).
    """
    n = len(widths)
    # Positions in sequences for constant-time precedence checks
    pos_a = np.empty(n, dtype=int)
    pos_b = np.empty(n, dtype=int)
    pos_a[seq_a] = np.arange(n)
    pos_b[seq_b] = np.arange(n)

    # Edges for x and y constraint graphs
    x_edges: List[Tuple[int, int]] = []
    y_edges: List[Tuple[int, int]] = []
    # O(N^2) construction
    for i in range(n):
        for j in range(i + 1, n):
            a_i, a_j = seq_a[i], seq_a[j]
            # If a_i precedes a_j in A, compare positions in B
            if pos_b[a_i] < pos_b[a_j]:
                # a_i left of a_j
                x_edges.append((a_i, a_j))
            else:
                # a_i above a_j
                y_edges.append((a_i, a_j))

    x = longest_paths(widths, x_edges)
    y = longest_paths(heights, y_edges)
    return Floorplan(x=x, y=y, w=widths.copy(), h=heights.copy())


def decode_random_keys(keys: np.ndarray) -> np.ndarray:
    """
    Given a vector of real keys, return a permutation by argsort.
    """
    # Stable argsort to be deterministic for ties
    return np.argsort(keys, kind="mergesort")


def decode_solution(
    x: np.ndarray, modules: List[Module]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    From a decision vector x (3N real numbers in [0,1]), produce widths, heights
    after rotation, and the two permutations (sequence pair A and B).
    """
    n = len(modules)
    assert len(x) == 3 * n
    keys_a = x[:n]
    keys_b = x[n : 2 * n]
    rot_bits = x[2 * n : 3 * n] > 0.5

    seq_a = decode_random_keys(keys_a)
    seq_b = decode_random_keys(keys_b)

    base_w = np.array([m.width for m in modules], dtype=float)
    base_h = np.array([m.height for m in modules], dtype=float)
    widths = np.where(rot_bits, base_h, base_w)
    heights = np.where(rot_bits, base_w, base_h)
    return widths, heights, seq_a, seq_b


def permutations_to_keys(perm: np.ndarray) -> np.ndarray:
    """Map a permutation into evenly spaced keys in [0, 1]."""
    n = len(perm)
    keys = np.empty(n, dtype=float)
    # Assign ranks 0..n-1 to items in the order of perm
    for rank, item in enumerate(perm):
        keys[item] = rank / max(1, n - 1)
    return keys


def encode_from_seqs_and_rot(
    seq_a: np.ndarray, seq_b: np.ndarray, rot_bits: np.ndarray
) -> np.ndarray:
    keys_a = permutations_to_keys(seq_a)
    keys_b = permutations_to_keys(seq_b)
    rot = np.where(rot_bits, 1.0, 0.0)
    return np.concatenate([keys_a, keys_b, rot]).astype(float)


# ------------------------------- Problems ---------------------------------


class FloorplanMOProblem(ElementwiseProblem):
    """
    Multi-objective problem: minimize [area, hpwl, overlap].
    If nets is empty, hpwl returns 0 and problem effectively optimizes area and overlap.
    """

    def __init__(self, modules: List[Module], nets: List[Net] | None = None, outline: Optional[Tuple[float, float]] = None):
        self.modules = modules
        self.nets = nets or []
        self.outline = outline
        n = len(modules)
        super().__init__(n_var=3 * n, n_obj=3, xl=0.0, xu=1.0)

    def _evaluate(self, x: np.ndarray, out: Dict[str, Any], *args, **kwargs) -> None:
        widths, heights, seq_a, seq_b = decode_solution(x, self.modules)
        placement = place_sequence_pair(widths, heights, seq_a, seq_b)
        area = placement.area()
        wl = placement.hpwl(self.nets)
        dead = placement.deadspace()
        # Outline overflow as a soft constraint if outline is provided
        overflow = 0.0
        if self.outline is not None:
            total_w, total_h = placement.bbox()
            ow = max(0.0, total_w - self.outline[0])
            oh = max(0.0, total_h - self.outline[1])
            overflow = ow + oh
        out["F"] = np.array([area, wl, dead + overflow], dtype=float)


class FloorplanAreaProblem(ElementwiseProblem):
    """
    Single-objective area-only floorplanning. Useful for MCNC comparisons.
    """

    def __init__(self, modules: List[Module]):
        self.modules = modules
        n = len(modules)
        super().__init__(n_var=3 * n, n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, x: np.ndarray, out: Dict[str, Any], *args, **kwargs) -> None:
        widths, heights, seq_a, seq_b = decode_solution(x, self.modules)
        placement = place_sequence_pair(widths, heights, seq_a, seq_b)
        out["F"] = float(placement.area())


# ------------------------------ Benchmarks --------------------------------


def synthetic_benchmark(num_modules: int, rng: np.random.Generator) -> List[Module]:
    """
    Create a synthetic set of rectangles with a wide range of sizes.
    """
    widths = rng.integers(10, 80, size=num_modules)
    heights = rng.integers(10, 80, size=num_modules)
    return [Module(f"b{i}", float(w), float(h)) for i, (w, h) in enumerate(zip(widths, heights))]


def create_results_directory(base: str = "results_p3") -> str:
    now = datetime.now()
    base_path = Path.cwd() / base
    year_month = now.strftime("%Y-%m")
    day = now.strftime("%d")
    time_str = now.strftime("%H_%M_%S")
    results_dir = base_path / year_month / day / time_str
    results_dir.mkdir(parents=True, exist_ok=True)
    return str(results_dir)


@dataclass
class RunConfig:
    steps: int = 20000
    max_time_s: float = 60.0
    pop_size: int = 200
    repeats: int = 10  # Number of independent runs per benchmark
    seed: int = 42
    n_jobs: Optional[int] = None  # CPU parallel workers; None -> os.cpu_count()
    # Memetic GA parameters
    memetic_top_k_rate: float = 0.1  # fraction of population to hill-climb each generation
    memetic_steps: int = 20          # local steps per individual
    seed_rate: float = 0.1           # fraction of initial pop seeded by heuristics
    plot: bool = True                # enable plotting
    # Early stopping if no improvement in best area for a while
    no_improve_patience_evals: Optional[int] = 5000
    no_improve_min_delta: float = 0.0

    def termination(self) -> Termination:
        ts = [
            get_termination("n_evals", self.steps),
            get_termination("time", self.max_time_s),
        ]
        if self.no_improve_patience_evals and self.no_improve_patience_evals > 0:
            ts.append(NoImprovementTermination(
                patience_evals=int(self.no_improve_patience_evals),
                min_delta=float(self.no_improve_min_delta),
            ))
        return TerminateIfAny(*ts)


class NoImprovementTermination(Termination):
    """
    Stop if best area does not improve by at least `min_delta` for `patience_evals` evaluations.
    Works for both SOO and MOO (uses first objective as area).
    """

    def __init__(self, patience_evals: int = 5000, min_delta: float = 0.0) -> None:
        super().__init__()
        self.patience = patience_evals
        self.min_delta = min_delta
        self.best = None
        self.last_improve_eval = 0

    def _update_best(self, current_best: float, n_eval: int) -> None:
        if self.best is None or (self.best - current_best) > self.min_delta:
            self.best = current_best
            self.last_improve_eval = n_eval

    def _update(self, algorithm) -> float:
        # number of evaluations so far
        n_eval = int(algorithm.evaluator.n_eval)
        # extract current best area from algorithm.opt
        F = algorithm.opt.get("F")
        if F.ndim == 1:
            current_best = float(F[0])
        else:
            current_best = float(F[0, 0])
        # update best and last improvement
        self._update_best(current_best, n_eval)
        # progress grows with stagnation; 1.0 means stop
        stagnation = n_eval - self.last_improve_eval
        progress = stagnation / max(1, self.patience)
        # clamp to [0, 1]
        if progress < 0.0:
            progress = 0.0
        elif progress > 1.0:
            progress = 1.0
        return float(progress)


def run_area_only(modules: List[Module], config: RunConfig, algo_name: str) -> Tuple[List[float], Dict[str, Any]]:
    problem = FloorplanAreaProblem(modules)
    rng = np.random.default_rng(config.seed)
    results: List[float] = []
    extras: Dict[str, Any] = {}
    n_jobs = config.n_jobs if (config.n_jobs or 0) > 0 else (os.cpu_count() or 2)

    def make_evaluator(n_workers: int) -> Evaluator:
        try:
            return Evaluator(n_threads=n_workers)  # newer pymoo
        except TypeError:
            e = Evaluator()
            try:
                setattr(e, "n_threads", n_workers)
            except Exception:
                pass
            return e

    histories: List["HistoryCallback"] = []
    for r in range(config.repeats):
        seed = int(rng.integers(0, 1_000_000))
        sampling = HeuristicSampling(seed_rate=config.seed_rate)
        if algo_name == "GA":
            algo = GA(pop_size=config.pop_size, sampling=sampling)
        elif algo_name == "NSGA2":
            # NSGA-II also works for single objective
            algo = NSGA2(pop_size=config.pop_size, sampling=sampling)
        else:
            raise ValueError("Unsupported algorithm for area-only run")

        history = HistoryCallback()
        histories.append(history)
        res = minimize(
            problem,
            algo,
            termination=config.termination(),
            seed=seed,
            verbose=False,
            evaluator=make_evaluator(n_jobs),
            callback=CompositeCallback([
                MemeticCallback(top_k_rate=config.memetic_top_k_rate, steps=config.memetic_steps),
                history,
            ]),
        )
        results.append(float(np.asarray(res.F).reshape(-1)[0]))

    extras["runs"] = results
    extras["best"] = float(np.min(results)) if results else np.nan
    extras["mean"] = float(np.mean(results)) if results else np.nan
    extras["std"] = float(np.std(results)) if results else np.nan
    extras["histories"] = histories
    return results, extras


def run_multi_objective(modules: List[Module], nets: List[Net], config: RunConfig, outline: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    problem = FloorplanMOProblem(modules, nets, outline=outline)
    sampling = HeuristicSampling(seed_rate=config.seed_rate)
    algo = NSGA2(pop_size=config.pop_size, sampling=sampling)
    n_jobs = config.n_jobs if (config.n_jobs or 0) > 0 else (os.cpu_count() or 2)
    def make_evaluator(n_workers: int) -> Evaluator:
        try:
            return Evaluator(n_threads=n_workers)
        except TypeError:
            e = Evaluator()
            try:
                setattr(e, "n_threads", n_workers)
            except Exception:
                pass
            return e
    history = HistoryCallback()
    res = minimize(
        problem,
        algo,
        termination=config.termination(),
        seed=config.seed,
        verbose=False,
        evaluator=make_evaluator(n_jobs),
        callback=CompositeCallback([
            MemeticCallback(top_k_rate=config.memetic_top_k_rate, steps=config.memetic_steps),
            history,
        ]),
    )
    pareto = res.F  # (k, 3)
    # Pick minimum area point for reporting
    best_idx = int(np.argmin(pareto[:, 0])) if pareto.size else 0
    summary = {
        "pareto": pareto,
        "best_area": float(pareto[best_idx, 0]) if pareto.size else np.nan,
        "best_hpwl": float(pareto[best_idx, 1]) if pareto.size else np.nan,
        "best_dead_or_overflow": float(pareto[best_idx, 2]) if pareto.size else np.nan,
    }
    return summary


# -------------------------- Memetic components ----------------------------


def area_from_x(x: np.ndarray, modules: List[Module]) -> float:
    widths, heights, seq_a, seq_b = decode_solution(x, modules)
    placement = place_sequence_pair(widths, heights, seq_a, seq_b)
    return placement.area()


def hill_climb_area(
    x: np.ndarray,
    modules: List[Module],
    steps: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    n = len(modules)
    # Decode to discrete representation
    _, _, seq_a, seq_b = decode_solution(x, modules)
    rot_bits = x[2 * n : 3 * n] > 0.5
    best_area = area_from_x(x, modules)
    best_seq_a = seq_a.copy()
    best_seq_b = seq_b.copy()
    best_rot = rot_bits.copy()

    for _ in range(max(1, steps)):
        move_type = rng.integers(0, 3)
        if move_type == 0 and n > 1:
            # swap in A
            i = int(rng.integers(0, n - 1))
            j = i + 1
            seq_new_a = best_seq_a.copy()
            seq_new_a[i], seq_new_a[j] = seq_new_a[j], seq_new_a[i]
            x_new = encode_from_seqs_and_rot(seq_new_a, best_seq_b, best_rot)
        elif move_type == 1 and n > 1:
            # swap in B
            i = int(rng.integers(0, n - 1))
            j = i + 1
            seq_new_b = best_seq_b.copy()
            seq_new_b[i], seq_new_b[j] = seq_new_b[j], seq_new_b[i]
            x_new = encode_from_seqs_and_rot(best_seq_a, seq_new_b, best_rot)
        else:
            # flip a rotation bit
            k = int(rng.integers(0, n))
            rot_new = best_rot.copy()
            rot_new[k] = ~rot_new[k]
            x_new = encode_from_seqs_and_rot(best_seq_a, best_seq_b, rot_new)

        new_area = area_from_x(x_new, modules)
        if new_area < best_area:
            best_area = new_area
            # Update best components to reflect accepted move
            if move_type == 0 and n > 1:
                best_seq_a = seq_new_a
            elif move_type == 1 and n > 1:
                best_seq_b = seq_new_b
            else:
                best_rot = rot_new
            x = x_new

    return x


class MemeticCallback(Callback):
    def __init__(self, top_k_rate: float = 0.1, steps: int = 20) -> None:
        super().__init__()
        self.top_k_rate = max(0.0, min(1.0, top_k_rate))
        self.steps = max(1, int(steps))

    def notify(self, algorithm) -> None:
        pop = algorithm.pop
        if pop is None or len(pop) == 0:
            return
        problem = algorithm.problem
        # Determine objective for sorting (area is F or F[:,0])
        F = pop.get("F")
        if F.ndim == 1:
            scores = F
        else:
            scores = F[:, 0]
        order = np.argsort(scores)
        k = max(1, int(np.ceil(self.top_k_rate * len(pop))))
        idxs = order[:k]
        rng = np.random.default_rng()
        for idx in idxs:
            x = pop[idx].X.copy()
            # modules are available on problem
            if hasattr(problem, "modules"):
                modules = problem.modules
            else:
                # Fallback: infer from n_var
                n = problem.n_var // 3
                modules = [Module(str(i), 1.0, 1.0) for i in range(n)]
            x_improved = hill_climb_area(x, modules, steps=self.steps, rng=rng)
            # Evaluate improved solution robustly across pymoo versions
            def eval_F(prob, xvec):
                try:
                    res = prob.evaluate(xvec, return_values_of="F")
                    if isinstance(res, dict):
                        return np.asarray(res["F"])  # older dict-like
                    return np.asarray(res)
                except Exception:
                    return np.asarray(prob.evaluate(xvec))

            f_new = eval_F(problem, x_improved)
            f_old = np.asarray(pop[idx].F)
            # Compare on first objective (area)
            new_area = float(np.asarray(f_new).reshape(-1)[0])
            old_area = float(np.asarray(f_old).reshape(-1)[0])
            if new_area < old_area:
                pop[idx].set("X", x_improved)
                pop[idx].set("F", np.asarray(f_new).reshape(f_old.shape))


class HeuristicSampling(Sampling):
    """
    Seed a fraction of the initial population with heuristic permutations/rotations:
    - A-order = by height desc, B-order = by width desc
    - A-order = by area desc,   B-order = by aspect ratio desc
    - random rotations with bias toward making tall modules rotate to reduce height
    Remaining individuals are random in [0,1].
    """

    def __init__(self, seed_rate: float = 0.1):
        super().__init__()
        self.seed_rate = max(0.0, min(1.0, float(seed_rate)))

    def _do(self, problem: Problem, n_samples: int, **kwargs) -> np.ndarray:
        n = problem.n_var // 3
        assert n * 3 == problem.n_var
        rng = np.random.default_rng()

        # Access modules for heuristics when available
        modules = getattr(problem, "modules", [Module(str(i), 1.0, 1.0) for i in range(n)])
        widths = np.array([m.width for m in modules], dtype=float)
        heights = np.array([m.height for m in modules], dtype=float)
        areas = widths * heights
        aspects = widths / np.maximum(1.0, heights)

        candidates: List[np.ndarray] = []

        # Heuristic 1: sort by height desc / width desc
        order_h = np.argsort(-heights)
        order_w = np.argsort(-widths)
        rot1 = heights > widths
        candidates.append(encode_from_seqs_and_rot(order_h, order_w, rot1))

        # Heuristic 2: sort by area desc / aspect desc
        order_a = np.argsort(-areas)
        order_ar = np.argsort(-aspects)
        rot2 = heights > widths
        candidates.append(encode_from_seqs_and_rot(order_a, order_ar, rot2))

        # Heuristic 3: interleave tall and wide
        tall = list(np.argsort(-(heights - widths)))
        wide = list(np.argsort(-(widths - heights)))
        inter = []
        for i in range(n):
            if i < len(tall):
                inter.append(tall[i])
            if i < len(wide):
                inter.append(wide[i])
        inter = np.array(list(dict.fromkeys(inter))[:n])
        order_b = inter[::-1]
        rot3 = heights > widths
        candidates.append(encode_from_seqs_and_rot(inter, order_b, rot3))

        num_seed = int(np.ceil(self.seed_rate * n_samples))
        X = np.empty((n_samples, 3 * n), dtype=float)

        # Place heuristic seeds
        for i in range(min(num_seed, len(candidates))):
            X[i, :] = candidates[i]

        # Fill the rest randomly
        if num_seed < n_samples:
            X[num_seed:, :] = rng.random((n_samples - num_seed, 3 * n))

        return X


class HistoryCallback(Callback):
    """Track best area vs number of evaluations."""

    def __init__(self) -> None:
        super().__init__()
        self.n_evals: List[float] = []
        self.best_values: List[float] = []

    def notify(self, algorithm) -> None:
        self.n_evals.append(float(algorithm.evaluator.n_eval))
        # area is F or first col of F
        F = algorithm.opt.get("F")
        if F.ndim == 1:
            self.best_values.append(float(F[0]))
        else:
            self.best_values.append(float(F[0, 0]))


class CompositeCallback(Callback):
    def __init__(self, callbacks: List[Callback]):
        super().__init__()
        self.callbacks = callbacks

    def notify(self, algorithm) -> None:
        for cb in self.callbacks:
            cb.notify(algorithm)


def plot_history(history: HistoryCallback, title: str, filename: str) -> None:
    if not history.n_evals:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(history.n_evals, history.best_values, label="Best area", color="blue")
    plt.xlabel("Evaluations")
    plt.ylabel("Best area")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_sota_comparison(stats_df: pd.DataFrame, results_dir: str) -> None:
    # Paper best values (Tang & Yao MA) extracted from the provided table image
    paper_best = {"ami33": 1190896.0, "ami49": 37487940.0}
    for bench in ["ami33", "ami49"]:
        dfb = stats_df[stats_df["benchmark"] == bench]
        if dfb.empty:
            continue
        ga_best = float(dfb[dfb["algorithm"] == "GA"]["best"].min()) if not dfb[dfb["algorithm"] == "GA"].empty else np.nan
        nsga_best = float(dfb[dfb["algorithm"] == "NSGA2"]["best"].min()) if not dfb[dfb["algorithm"] == "NSGA2"].empty else np.nan
        labels = ["Paper MA", "GA best", "NSGA2 best"]
        values = [paper_best[bench], ga_best, nsga_best]
        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values, color=["gray", "#1f77b4", "#ff7f0e"]) 
        for b, v in zip(bars, values):
            plt.text(b.get_x() + b.get_width()/2, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9, rotation=0)
        plt.ylabel("Area")
        plt.title(f"State-of-the-art vs Our Results: {bench}")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"sota_comparison_{bench}.png"), bbox_inches="tight")
        plt.close()


# --------------------------------- Main -----------------------------------


def main():
    results_dir = create_results_directory()

    # Benchmarks to run
    rng = np.random.default_rng(7)
    synthetic_sets = {
        "synthetic_50": synthetic_benchmark(50, rng),
        "synthetic_100": synthetic_benchmark(100, rng),
    }

    # MCNC: parse modules only (area-focused comparison)
    repo_root = Path.cwd()
    ami33_path = str(repo_root / "ami33.yal.txt")
    ami49_path = str(repo_root / "ami49.yal.txt")
    if os.path.exists(ami33_path):
        synthetic_sets["ami33"] = parse_yal_modules(ami33_path)
    if os.path.exists(ami49_path):
        synthetic_sets["ami49"] = parse_yal_modules(ami49_path)

    config = RunConfig()

    # Announce CPU core usage for parallel evaluation
    effective_n_jobs = config.n_jobs if (config.n_jobs or 0) > 0 else (os.cpu_count() or 2)
    print(f"Using {effective_n_jobs} CPU thread(s) for parallel evaluations.")

    # Area-only stats table for GA and NSGA2
    rows = []
    runs_by_algo: Dict[str, Dict[str, List[float]]] = {"GA": {}, "NSGA2": {}}

    for name, modules in synthetic_sets.items():
        for algo_name in ["GA", "NSGA2"]:
            runs, stats = run_area_only(modules, config, algo_name)
            runs_by_algo[algo_name][name] = runs
            rows.append(
                {
                    "benchmark": name,
                    "algorithm": algo_name,
                    "best": stats["best"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                }
            )

    stats_df = pd.DataFrame(rows)
    stats_csv = os.path.join(results_dir, "area_stats.csv")
    stats_df.to_csv(stats_csv, index=False)
    # Optional SOTA comparison plots
    if config.plot:
        try:
            plot_sota_comparison(stats_df, results_dir)
        except Exception as e:
            print(f"SOTA plot error: {e}")

    # Kruskal–Wallis tests per benchmark between GA and NSGA2 runs
    kw_rows = []
    for name in synthetic_sets.keys():
        ga_vals = runs_by_algo["GA"].get(name, [])
        nsga_vals = runs_by_algo["NSGA2"].get(name, [])
        if ga_vals and nsga_vals:
            H, p = kruskal(ga_vals, nsga_vals)
            # Mean ranks not directly from scipy; report means instead
            kw_rows.append({"benchmark": name, "H": float(H), "p_value": float(p)})
    kw_df = pd.DataFrame(kw_rows)
    kw_csv = os.path.join(results_dir, "kruskal_wallis_ga_vs_nsga2.csv")
    kw_df.to_csv(kw_csv, index=False)

    # Run multi-objective example on synthetic_50 (no nets) to produce Pareto front
    # and on ami33/ami49 focusing on area only; MO run still executed but wl=0.
    mo_rows = []
    for name in ["synthetic_50", "ami33", "ami49"]:
        if name in synthetic_sets:
            modules = synthetic_sets[name]
            # If MCNC, parse nets and outline
            nets = []
            outline = None
            yal_path = None
            if name == "ami33":
                yal_path = ami33_path if os.path.exists(ami33_path) else None
            elif name == "ami49":
                yal_path = ami49_path if os.path.exists(ami49_path) else None
            if yal_path:
                nets = parse_yal_nets(yal_path, modules)
                outline = parse_yal_outline(yal_path)
            mo_summary = run_multi_objective(modules, nets=nets, config=config, outline=outline)
            mo_rows.append({
                "benchmark": name,
                "best_area": mo_summary["best_area"],
                "best_hpwl": mo_summary["best_hpwl"],
                "best_dead_or_overflow": mo_summary["best_dead_or_overflow"],
                "pareto_size": 0 if np.isnan(mo_summary["best_area"]) else len(mo_summary["pareto"]),
            })
            # Save Pareto front if available
            if mo_summary["pareto"] is not None and len(mo_summary["pareto"]):
                np.savetxt(os.path.join(results_dir, f"pareto_{name}.csv"), mo_summary["pareto"], delimiter=",", header="area,hpwl,dead_or_overflow", comments="")

    mo_df = pd.DataFrame(mo_rows)
    mo_csv = os.path.join(results_dir, "multi_objective_summary.csv")
    mo_df.to_csv(mo_csv, index=False)

    # Friendly console prints
    print("=== Area statistics (best / mean / std) over repeats ===")
    print(stats_df.to_string(index=False))
    if not kw_df.empty:
        print("\n=== Kruskal–Wallis: GA vs NSGA2 (per benchmark) ===")
        print(kw_df.to_string(index=False))
    if not mo_df.empty:
        print("\n=== Multi-objective summary (min-area point from Pareto front) ===")
        print(mo_df.to_string(index=False))
    print(f"\nResults saved to: {results_dir}")

    # Progress plots per run for area-only experiments (first few for brevity)
    if config.plot:
        try:
            # Reconstruct a minimal mapping benchmark->histories by re-running lightweight histories collection
            # (We already have histories within run_area_only per call, but not returned here per benchmark.)
            # Plot per-benchmark for NSGA2 as representative.
            for name, modules in list(synthetic_sets.items()):
                runs, stats = run_area_only(modules, config, "NSGA2")
                histories = stats.get("histories", [])
                for i, hist in enumerate(histories[:2]):
                    plot_history(hist, f"{name} NSGA2 Run {i+1}", os.path.join(results_dir, f"history_{name}_nsga2_{i+1}.png"))
        except Exception as e:
            print(f"History plot error: {e}")


if __name__ == "__main__":
    main()


