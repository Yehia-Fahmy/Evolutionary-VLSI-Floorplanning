# LAB 2: Evolutionary Optimization
# Deadline: June 28th 2025

## Setup

Clone the repo from Gitlab and create and set up a Python virtual environment.

```zsh
mkdir -p $HOME/ece493t32-s25_ml-chip-design/labs
cd $HOME/ece493t32-s25_ml-chip-design/labs
git clone ist-git@git.uwaterloo.ca:ece493t32-s25_ml-chip-design/labs/y2fahmy-lab2.git
cd y2fahmy-lab2
```

### On Lab Machines

If you are using the lab machines we have a Python virtual environment already set up for you. Use this command to activate it.

```zsh
source /zfsspare/ml-playground-env/bin/activate
```

### On Other Machines

```zsh
git clone ist-git@git.uwaterloo.ca:ece493t32-s25_ml-chip-design/labs/y2fahmy-lab2.git
python3 -m venv ~/ml-playground
source ~/ml-playground/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

**Every time you create a new terminal, you will need to activate the virtual environment.**

Check using `which python` to make sure it points to `ml-playground/bin/python`.

## Lab Objectives

This lab is broken down into several different problems.

1. `problem_0.py`
2. `problem_1.py`
3. Other problems added soon.


## Problem 0

In problem 0 you are just whetting your appetite with pymoo. It involves optimizing the same 1d and 10d functions as in the previous lab, but this time using the `pymoo` library. You might be surprised at how much faster it is.

Just complete the todos and make sure the two `get_optimal` functions work. Links to the pymoo documentation are left in the code for you to look at if you need help.


## Problem 1

In problem 1 you will need to understand a little bit about lab infrastructure. As discussed in the tutorial, we will be trying to use evoluationary optimization to optimize the operation priorities for list scheduling. The goal is to minimize the final latency of the schedule.

### `schedule.py`

In this file you will find the bulk of the code for scheduling and the basic classes used for defining data flow graphs (DFGs). The main three classes for DFGs are:

- `Value`: Represents a value in the DFG, instantiated without a source implies that the value is a constant or an input.
- `Operator`: Represents an operator. An operator has a name, number of inputs and ouputs, and timing characteristics. Operators are callable classes, and when you call them they create an operation and return values.
- `Operation`: Represents an operation in the DFG. An operation is the application of an operator to a particular value.
- `Context`: Context is a class that holds the whole state of the DFG, it is instantiated with a list of values, and chases back to find the source of each value recursively to analyze the whole DFG. Context has 3 main properties:
    - `values`: A set of all values in the DFG.
    - `operation_priority`: A dictionary mapping each operation to its priority in the schedule.
    - `operator_copies`: A dictionary mapping each operator to the number of copies of that operator available for scheduling.

The keys of the `operation_priority` and `operator_copies` dictionaries are the set of all operations and operators in the DFG, respectively.

There are other things to look at in the file, but the main other important one is the `schedule` function.

```python
def schedule(
    context: Context,
    constants_as_intermediates=False,
    detailed_table=False,
):
    """
    Schedule the operations in the context using list scheduling.
    """
    ...
    return {
        "schedule_table": schedule_table,
        "operation_start_times": operation_start_times,
        "latency": time,
        "max_number_intermediates": max_number_intermediates,
    }
```

Scheduling runs list scheduling as described in the tutorial. It returns the latency, and a the schedule table if `detailed_table` is set to `True`.

With this in mind you should now be able to understand the scheduling problem, you will be given a context, and you will need to set all of the operation priorities in the context to minimize the latency of the schedule. Something along this lines:

```python
for operation in context.operation_priority:
    context.operation_priority[operation] = cleverly_chosen_prioritys

latency = schedule(context)["latency"]
```

### `create_test_graphs.py`

Create test graphs is a script that has some interesting DFGs that can be created. It has a set of generic functions that apply operations, and it also has a list of functions with the names `create_*_graph` that create specific DFGs. For example:

```python
def create_parallel_linear_adds_graph(
    depth: int, width: int, operators=basic_operators
):
    ...
```

Will return a DFG (a context) that is a `width` parallel linear addition chains of `depth` depth. For `width=4, depth=5` it looks something like this:
```
out_0 = ((((in_0_0 + in_0_1) + in_0_2) + in_0_3) + in_0_4)
out_1 = ((((in_1_0 + in_1_1) + in_1_2) + in_1_3) + in_1_4)
out_2 = ((((in_2_0 + in_2_1) + in_2_2) + in_2_3) + in_2_4)
out_3 = ((((in_3_0 + in_3_1) + in_3_2) + in_3_3) + in_3_4)
```

### TODOS

In `problem_1.py` you will find a set of todos that you need to complete. The actual goal will be for students to try to get the best configurations of algorithms as you have many dials and things to play with.

## Problem 2

Problem 2 involves optimizing both the scheduling and the operator copies. This involves a more complicated genome representation. Your goals are to minimize latency, and minimize area. Area will be calculated using `operator_areas` which is a dictionary giving the cost of each operator.

As the optimization problem is now multi-objective. The comparisions are based on Hypervolume of the Pareto front. The reference points are already added to the problem configurations.

Not all of the algorithms in `pymoo` support multi-objective optimization, so you will need to use the ones that do.

## Problem 3

Cancelled, might be rolled into Mini project ideas.

## Submission

Make sure to commit your changes frequently and push them to the remote repository.

```zsh
git add .
git commit -m "Completed lab"
git push
```
