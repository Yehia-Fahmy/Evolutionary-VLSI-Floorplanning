# Evolutionary VLSI Floorplanning & High-Level Synthesis Optimization

> **Advanced Multi-Objective Optimization for VLSI Circuit Design**

A comprehensive machine learning and evolutionary optimization system that solves critical VLSI circuit design problems using state-of-the-art algorithms including NSGA-II, CMA-ES, and Differential Evolution.

## 🎯 What This Project Does

This project implements **evolutionary optimization algorithms** to solve two fundamental VLSI design challenges:

1. **High-Level Synthesis (HLS) Scheduling**: Optimizes operation scheduling to minimize latency in data flow graphs
2. **Multi-Objective VLSI Floorplanning**: Balances area and delay optimization using Pareto-optimal solutions

## 🚀 Key Technical Achievements

- **Multi-Objective Optimization**: Implements NSGA-II, NSGA-III, and SPEA2 for Pareto-optimal solutions
- **Advanced Algorithms**: Features CMA-ES, Differential Evolution, and Genetic Algorithms
- **Real VLSI Benchmarks**: Tested on industry-standard AMI33 and AMI49 floorplanning benchmarks
- **Comprehensive Evaluation**: Statistical analysis with Kruskal-Wallis tests and hypervolume metrics
- **Production-Ready**: Generates detailed performance reports and visualization

## 📊 Performance Highlights

- **Benchmark Results**: Achieves competitive results on AMI33 (33 modules) and AMI49 (49 modules) benchmarks
- **Statistical Validation**: Comprehensive statistical analysis comparing algorithm performance
- **Pareto Front Analysis**: Multi-objective optimization with hypervolume-based evaluation
- **Scalable Architecture**: Handles complex data flow graphs with 100+ operations

## 🛠️ Quick Start

### Prerequisites
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Optimization

#### Problem 1: High-Level Synthesis Scheduling
```bash
python problem_1.py
```
- Optimizes operation priorities using evolutionary algorithms
- Minimizes latency in data flow graphs
- Supports multiple algorithms: NSGA-II, CMA-ES, DE, GA

#### Problem 2: Multi-Objective VLSI Floorplanning
```bash
python problem_2.py
```
- Optimizes both area and delay simultaneously
- Uses Pareto-optimal solutions
- Generates comprehensive performance reports

#### Problem 0: Algorithm Benchmarking
```bash
python problem_0.py
```
- Compares optimization algorithms on standard test functions
- Demonstrates algorithm performance and convergence

## 📈 Key Results

The project includes comprehensive results in the `latest_results/` directory:

- **Statistical Analysis**: Kruskal-Wallis tests comparing algorithm performance
- **Pareto Fronts**: Multi-objective optimization results for different benchmarks
- **Performance Plots**: Convergence history and algorithm comparison visualizations
- **Area Statistics**: Detailed area optimization results across benchmarks

## 🎯 Technical Skills Demonstrated

- **Evolutionary Algorithms**: NSGA-II, NSGA-III, SPEA2, CMA-ES, Differential Evolution
- **Multi-Objective Optimization**: Pareto-optimal solutions and hypervolume metrics
- **VLSI Design**: High-level synthesis, floorplanning, and circuit optimization
- **Statistical Analysis**: Kruskal-Wallis tests, performance benchmarking
- **Python Ecosystem**: NumPy, Matplotlib, PyMOO, SciPy, Pandas

## 📁 Project Structure

```
├── problem_0.py              # Algorithm benchmarking and comparison
├── problem_1.py              # High-level synthesis scheduling optimization
├── problem_2.py              # Multi-objective VLSI floorplanning
├── schedule.py               # Core scheduling algorithms and data structures
├── create_test_graphs.py     # Test data flow graph generation
├── latest_results/           # Comprehensive performance analysis
│   ├── area_stats.csv       # Area optimization statistics
│   ├── pareto_*.csv         # Pareto front results
│   └── *.png                # Performance visualizations
└── requirements.txt          # Python dependencies
```

## 🔧 Technical Stack

- **Optimization**: PyMOO (NSGA-II, NSGA-III, SPEA2, CMA-ES, DE, GA)
- **Data Analysis**: NumPy, Pandas, SciPy
- **Visualization**: Matplotlib
- **VLSI Design**: Custom scheduling and floorplanning algorithms
- **Statistical Analysis**: Kruskal-Wallis tests, hypervolume metrics

## 💼 Business Impact

This solution addresses critical optimization challenges in:
- **Semiconductor Industry**: VLSI circuit design and floorplanning
- **High-Performance Computing**: Operation scheduling and resource allocation
- **Embedded Systems**: Real-time scheduling and area optimization

The evolutionary approach provides significant advantages over traditional heuristic methods through automated optimization and multi-objective trade-off analysis.

## 📋 Usage Examples

### Basic Single-Objective Optimization
```python
# Optimize operation scheduling for minimum latency
python problem_1.py
```

### Multi-Objective Optimization
```python
# Optimize both area and delay simultaneously
python problem_2.py
```

### Algorithm Comparison
```python
# Compare different evolutionary algorithms
python problem_0.py
```

## 📊 Understanding the Results

### Key Metrics
- **Latency**: Total execution time of the scheduled operations (minimize)
- **Area**: Total hardware area required (minimize)
- **Hypervolume**: Quality of Pareto-optimal solutions (maximize)
- **Convergence**: Algorithm performance over iterations

### Output Files
- **CSV Reports**: Detailed statistical analysis and performance metrics
- **PNG Visualizations**: Convergence plots, Pareto fronts, and comparisons
- **Partition Files**: VLSI floorplanning results in standard format

## 🏗️ Advanced Features

- **Statistical Validation**: Kruskal-Wallis tests for algorithm comparison
- **Pareto Front Analysis**: Multi-objective optimization with hypervolume metrics
- **Benchmark Integration**: Industry-standard AMI33/AMI49 test cases
- **Comprehensive Reporting**: Automated generation of performance reports
- **Algorithm Comparison**: Side-by-side evaluation of multiple optimization methods
