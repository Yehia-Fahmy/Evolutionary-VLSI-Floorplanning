import heapq
from collections import defaultdict, deque
from enum import Enum
from itertools import count
from typing import Optional


class Value:
    """Represents a data value in the computation graph."""
    __slots__ = ("source", "name")
    source: Optional["Operation"]
    name: Optional[str]

    def __init__(
        self, name: Optional[str] = None, source: Optional["Operation"] = None
    ):
        self.source = source
        self.name = name

    def __repr__(self):
        return f"Value({self.name})"


class Operation:
    """Represents an operation that takes inputs and produces outputs."""
    __slots__ = ("operator", "args", "outs")
    operator: "Operator"
    args: list[Value]
    outs: list[Value]

    def __init__(self, operator: "Operator", args: list[Value]):
        self.operator = operator
        self.args = args
        self.outs = []

    def __repr__(self):
        return f"({self.operator.name}, {self.args})->({self.outs})"


class Operator:
    """Defines the properties of an operation type."""
    __slots__ = ("name", "num_inputs", "num_outputs", "latency", "ii")
    name: str
    num_inputs: int
    num_outputs: int
    latency: int
    ii: int # Initiation Interval

    def __init__(
        self,
        name: str,
        num_inputs: int,
        num_outputs: int,
        latency: int = 1,
        ii: int = None,
    ):
        self.name = name
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.latency = latency
        self.ii = ii if ii is not None else latency

    def __call__(self, *args):
        """Creates an Operation instance when the Operator is called."""
        assert len(args) == self.num_inputs
        operation = Operation(self, list(args))
        for _ in range(self.num_outputs):
            operation.outs.append(Value(source=operation))

        if self.num_outputs == 1:
            return operation.outs[0]
        else:
            return operation.outs

    def __repr__(self):
        return f"Operator({self.name})"


class Context:
    """Holds the entire computation graph."""
    values: set[Value]
    operation_priority: dict[Operation, float]
    operator_copies: dict[Operator, int]
    cached_dependency_graph: Optional[dict]

    def __init__(self, *values):
        # Recursively find all values, operations, and operators.
        self.values = set()
        self.operation_priority = {}
        self.operator_copies = {}
        self.cached_dependency_graph = None

        counter = count()

        def find_all(value):
            if value in self.values:
                return

            self.values.add(value)
            if value.name is None:
                value.name = f"v{next(counter)}"

            if value.source is not None:
                if value.source not in self.operation_priority:
                    self.operation_priority[value.source] = 0.0
                    self.operator_copies[value.source.operator] = 1
                    for arg in value.source.args:
                        find_all(arg)

        for value in values:
            find_all(value)

        # Sort operators alphabetically by name and remake operator_copies
        sorted_operators = sorted(self.operator_copies.items(), key=lambda x: x[0].name)
        self.operator_copies = {op: num for op, num in sorted_operators}

    def dependency_graph(self):
        """
        Returns useful information for scheduling.
        Cached for performance.
        """
        if self.cached_dependency_graph is not None:
            return self.cached_dependency_graph

        op_dependants = {op: [] for op in self.operation_priority.keys()}
        op_num_waiting = {op: 0 for op in self.operation_priority.keys()}
        val_num_dependants = {val: 0 for val in self.values}

        for op in self.operation_priority.keys():
            for arg in op.args:
                val_num_dependants[arg] += 1
                if arg.source is not None:
                    if op not in op_dependants[arg.source]:
                        op_dependants[arg.source].append(op)
                        op_num_waiting[op] += 1

        op_ready = []
        for op in self.operation_priority.keys():
            if op_num_waiting[op] == 0:
                op_ready.append(op)

        num_constants = 0
        for val in self.values:
            if val.source is None:
                num_constants += 1

        self.cached_dependency_graph = {
            # Operation -> List of dependent operations
            "op_dependants": op_dependants,
            # Operation -> Number of values still waiting for
            "op_num_waiting": op_num_waiting,
            # Value -> Number of dependent operations
            "val_num_dependants": val_num_dependants,
            # List of operations with no waiting dependencies
            "op_ready": op_ready,
            # Number of constants (values with no source)
            "num_constants": num_constants,
        }

        return self.cached_dependency_graph


class EventHeap:
    """A min-heap for scheduling events, ensuring stable sort."""
    def __init__(self):
        self.i = 0  # Unique ID for stable sorting
        self.heap = []

    def push(self, time, event):
        heapq.heappush(self.heap, (time, self.i, event))
        self.i += 1

    def pop(self):
        time, _, event = heapq.heappop(self.heap)
        return (time, event)

    def is_empty(self):
        return len(self.heap) == 0

    def next_time(self):
        if self.is_empty():
            return None
        return self.heap[0][0]


class Event(Enum):
    """Types of events in the list scheduler."""
    OPERATOR_AVAILABLE = 1
    VALUES_READY = 2


def schedule(
    context: Context,
    constants_as_intermediates=False,
    detailed_table=False,
):
    """
    Schedule the operations in the context using list scheduling.
    """
    schedule_table = defaultdict(list)  # Time -> List[Operation]
    operation_start_times = {}  # Operation -> Start time

    cached_dependency_graph = context.dependency_graph()
    op_dependants = cached_dependency_graph["op_dependants"]
    op_num_waiting = cached_dependency_graph["op_num_waiting"].copy()
    val_num_dependants = cached_dependency_graph["val_num_dependants"].copy()

    for arg in context.values:
        if arg.source is None and not constants_as_intermediates:
            val_num_dependants[arg] = -1

    unique_id = count()
    events = EventHeap()
    topo_heap = []

    operator_queues = defaultdict(
        list
    )  # Operator -> List of operations ready to be scheduled

    # Phase 1: Do topological sort and prepare the operator queues.
    for op in cached_dependency_graph["op_ready"]:
        heapq.heappush(
            topo_heap,
            # Heapq uses a min-heap.
            (context.operation_priority[op], -next(unique_id), op),
        )

    while topo_heap:
        _, _, op = heapq.heappop(topo_heap)
        operator_queues[op.operator].append(op)
        for dep in op_dependants[op]:
            op_num_waiting[dep] -= 1
            if op_num_waiting[dep] == 0:
                heapq.heappush(
                    topo_heap,
                    (context.operation_priority[dep], -next(unique_id), dep),
                )

    # Reverse all operator queues so that we can pop from the end.
    for op in operator_queues:
        operator_queues[op].reverse()

    # Phase 2: Schedule given resource constraints.
    opt_available = {
        opt: 0 for opt in context.operator_copies.keys()
    }
    op_num_waiting = cached_dependency_graph["op_num_waiting"].copy()

    for opt, num in context.operator_copies.items():
        # All operators are available at time 0.
        for _ in range(num):
            events.push(0, (Event.OPERATOR_AVAILABLE, opt))

    op_booked = {op: False for op in context.operation_priority.keys()}

    time = 0
    number_intermediates = 0
    max_number_intermediates = 0
    if constants_as_intermediates:
        number_intermediates = cached_dependency_graph["num_constants"]

    while not events.is_empty():
        newtime, event = events.pop()
        if newtime != time:
            time = newtime
            # Update max intermediates count
            max_number_intermediates = max(
                number_intermediates, max_number_intermediates
            )

        if event[0] == Event.OPERATOR_AVAILABLE:
            opt = event[1]
            opt_available[opt] += 1
            if operator_queues[opt]:
                op = operator_queues[opt].pop()
                # The operation is either scheduled immediately or booked for later.
                # Either way it reserves an operator.
                opt_available[opt] -= 1
                if op_num_waiting[op] == 0:
                    # Schedule this operation
                    if detailed_table:
                        schedule_table[time].append(op)
                        operation_start_times[op] = time

                    # Update intermediate value counts
                    for arg in op.args:
                        val_num_dependants[arg] -= 1
                        if val_num_dependants[arg] == 0:
                            number_intermediates -= 1

                    # Add the outputs as new intermediates
                    number_intermediates += len(op.outs)

                    # Push events for when the values will be ready
                    events.push(time + op.operator.latency, (Event.VALUES_READY, op))
                    events.push(time + op.operator.ii, (Event.OPERATOR_AVAILABLE, opt))
                else:
                    op_booked[op] = True

        elif event[0] == Event.VALUES_READY:
            op = event[1]
            for dep in op_dependants[op]:
                op_num_waiting[dep] -= 1
                if op_num_waiting[dep] == 0 and op_booked[dep]:
                    # If this operation was booked, we can now schedule it.
                    if detailed_table:
                        schedule_table[time].append(dep)
                        operation_start_times[dep] = time

                    # Update intermediate value counts
                    for arg in dep.args:
                        val_num_dependants[arg] -= 1
                        if val_num_dependants[arg] == 0:
                            number_intermediates -= 1

                    # Add the outputs as new intermediates
                    number_intermediates += len(dep.outs)
                    
                    # Push events for when the values will be ready
                    events.push(time + dep.operator.latency, (Event.VALUES_READY, dep))
                    events.push(
                        time + dep.operator.ii, (Event.OPERATOR_AVAILABLE, dep.operator)
                    )

    return {
        "schedule_table": schedule_table,
        "operation_start_times": operation_start_times,
        "latency": time,
        "max_number_intermediates": max_number_intermediates,
    }


def asap_schedule(context: Context, constants_as_intermediates=False):
    """
    Perform an ASAP scheduling of the operations in the context.
    This is a simplified version that does not consider operator availability.
    """
    schedule_table = defaultdict(list)
    operation_start_times = {}

    cached_dependency_graph = context.dependency_graph()
    op_dependants = cached_dependency_graph["op_dependants"]
    op_num_waiting = cached_dependency_graph["op_num_waiting"].copy()
    
    value_ready_time = {}  # Maps a Value to the time it becomes available
    ready_queue = deque(cached_dependency_graph["op_ready"])

    # Graph inputs (constants) are available at time 0
    for val in context.values:
        if val.source is None:
            value_ready_time[val] = 0

    latency = 0

    # Process operations in topological order
    while ready_queue:
        op = ready_queue.popleft()
        
        # An operation can start as soon as all its arguments are ready.
        start_time = 0
        if op.args:
            start_time = max(value_ready_time.get(arg, 0) for arg in op.args)
        
        operation_start_times[op] = start_time
        
        # Its outputs will be ready after its latency.
        completion_time = start_time + op.operator.latency
        latency = max(latency, completion_time)
        for out_val in op.outs:
            value_ready_time[out_val] = completion_time
            
        # Decrement the dependency counter for all succeeding operations.
        # If a successor is ready, add it to the queue.
        for dep in op_dependants[op]:
            op_num_waiting[dep] -= 1
            if op_num_waiting[dep] == 0:
                ready_queue.append(dep)

    for op, start in operation_start_times.items():
        schedule_table[start].append(op)
        
    return {
        "schedule_table": schedule_table,
        "operation_start_times": operation_start_times,
        "latency": latency,
        "max_number_intermediates": 0, # Not calculated in this simple version
    }


def alap_schedule(
    context: Context, end_time: int, constants_as_intermediates=False
):
    """
    Perform an ALAP scheduling of the operations in the context.
    This is a simplified version that does not consider operator availability.
    An `end_time` (latency constraint) must be provided. It's often the
    latency from an ASAP schedule.
    """
    schedule_table = defaultdict(list)
    operation_start_times = {}

    cached_dependency_graph = context.dependency_graph()
    op_dependants = cached_dependency_graph["op_dependants"]
    
    op_producers = {op: [] for op in context.operation_priority}
    for op, deps in op_dependants.items():
        for dep in deps:
            op_producers[dep].append(op)

    # Initialize data structures for reverse topological traversal
    # Stores the latest an op's outputs must be ready
    op_latest_finish_time = {} 
    # Counter for scheduling an op's producers
    num_dependants_left = {op: len(deps) for op, deps in op_dependants.items()}
    
    ready_queue = deque()
    # Start with terminal operations (those with no dependants)
    for op, num in num_dependants_left.items():
        if num == 0:
            ready_queue.append(op)
            # A terminal operation must finish by the end_time
            op_latest_finish_time[op] = end_time

    # Process operations in reverse topological order
    while ready_queue:
        op = ready_queue.popleft()
        
        # The start time is determined by the latest finish time minus latency
        start_time = op_latest_finish_time[op] - op.operator.latency
        operation_start_times[op] = start_time
        
        # For each producer of this operation's inputs...
        for producer in op_producers[op]:
            # The producer must finish before this operation can start.
            # We take the minimum of constraints from all of the producer's children.
            current_latest_finish = op_latest_finish_time.get(producer, end_time)
            op_latest_finish_time[producer] = min(current_latest_finish, start_time)
            
            # If we have scheduled all dependants of the producer, it's ready.
            num_dependants_left[producer] -= 1
            if num_dependants_left[producer] == 0:
                ready_queue.append(producer)

    for op, start in operation_start_times.items():
        schedule_table[start].append(op)
        
    return {
        "schedule_table": schedule_table,
        "operation_start_times": operation_start_times,
        "latency": end_time,
        "max_number_intermediates": 0, # Not calculated in this simple version
    }


def write_dot(context: Context, filename: str):
    """
    Write a dot file representing the dataflow graph of the context.
    """
    with open(filename, "w") as f:
        f.write("digraph G {\n")
        f.write("  rankdir=TB;\n")
        # Write nodes for operations
        for op in context.operation_priority.keys():
            op_label = f"{op.operator.name}"
            f.write(f'  "{id(op)}" [label="{op_label}", shape=box];\n')
        # Write nodes for values
        for val in context.values:
            val_label = val.name if val.name else ""
            f.write(f'  "{id(val)}" [label="{val_label}", shape=ellipse];\n')
        # Write edges: op -> value (outputs)
        for op in context.operation_priority.keys():
            for out in op.outs:
                f.write(f'  "{id(op)}" -> "{id(out)}";\n')
        # Write edges: value -> op (inputs)
        for op in context.operation_priority.keys():
            for arg in op.args:
                f.write(f'  "{id(arg)}" -> "{id(op)}";\n')
        f.write("}\n")


__all__ = [
    "Value",
    "Context",
    "Operator",
    "schedule",
    "asap_schedule",
    "alap_schedule",
    "write_dot",
]
