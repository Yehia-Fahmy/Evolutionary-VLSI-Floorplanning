from schedule import *
from functools import cache


def flatten(lst):
    """
    Flatten nested lists.
    """
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


basic_operators = {
    "add": Operator("add", 2, 1, latency=1),
    "mul": Operator("mul", 2, 1, latency=3),
    "sqrt": Operator("sqrt", 1, 1, latency=32),
    "div": Operator("div", 2, 1, latency=12),
    "relu": Operator("relu", 1, 1, latency=1),
    "tanh": Operator("tanh", 1, 1, latency=10),
    "sigmoid": Operator("sigmoid", 1, 1, latency=8),
    "max": Operator("max", 2, 1, latency=1),
    "load": Operator("load", 0, 1, latency=1),
}

# Use the same operator for subtraction as addition.
basic_operators["sub"] = basic_operators["add"]


ZERO = Value("0")
ONE = Value("1")


def uniary_vector_operation(op: Operator, x: list[Value]) -> list[Value]:
    """
    Perform a unary operation on a vector.
    """
    return [op(v) for v in x]


def binary_vector_operation(
    op: Operator, x: list[Value], y: list[Value]
) -> list[Value]:
    """
    Perform a binary operation on two vectors of the same length.
    """
    if len(x) != len(y):
        raise ValueError("Vectors must be of the same length.")

    return [op(x[i], y[i]) for i in range(len(x))]


def binary_vector_linear_reduce(op: Operator, x: list[Value]) -> Value:
    """
    Perform a binary reduction on a vector, reducing it to a single value.
    """
    if len(x) == 0:
        raise ValueError("Vector must not be empty.")

    result = x[0]
    for i in range(1, len(x)):
        result = op(result, x[i])

    return result


def binary_vector_tree_reduce(op: Operator, x: list[Value]) -> Value:
    """
    Perform a binary reduction on a vector, reducing it to a single value using a tree structure.
    """
    if len(x) == 0:
        raise ValueError("Vector must not be empty.")

    nodes = x.copy()
    while len(nodes) > 1:
        next_nodes = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                next_nodes.append(op(nodes[i], nodes[i + 1]))
            else:
                next_nodes.append(nodes[i])
        nodes = next_nodes

    return nodes[0]


def load_matrix(M: int, N: int, operators) -> list[list[Value]]:
    """
    Load a matrix of size MxN, returning a list of lists.
    """
    LOAD = operators["load"]
    matrix = []
    for i in range(M):
        row = []
        for j in range(N):
            value = LOAD()
            value.name = f"m_{i}_{j}"
            row.append(value)
        matrix.append(row)
    return matrix


def load_vector(N: int, operators) -> list[Value]:
    """
    Load a vector of length N, returning a list of Values.
    """
    LOAD = operators["load"]
    vector = []
    for i in range(N):
        value = LOAD()
        value.name = f"v_{i}"
        vector.append(value)
    return vector


def const_matrix(M: int, N: int) -> list[list[Value]]:
    """
    Create a constant matrix of size MxN, returning a list of lists.
    """
    return [[Value(f"c_{i}_{j}") for j in range(N)] for i in range(M)]


def const_vector(N: int) -> list[Value]:
    """
    Create a constant vector of length N, returning a list of Values.
    """
    return [Value(f"c_{i}") for i in range(N)]


def mat_mul(A: list[list[Value]], B: list[list[Value]], operators) -> list[list[Value]]:
    """
    Perform matrix multiplication on two matrices A and B, returning the result as a new matrix.
    """
    if len(A[0]) != len(B):
        raise ValueError("Number of columns in A must match number of rows in B.")

    M = len(A)
    N = len(B[0])
    P = len(B)

    MUL = operators["mul"]
    ADD = operators["add"]

    result = [[None for j in range(N)] for i in range(M)]

    for i in range(M):
        for j in range(N):
            for k in range(P):
                if k == 0:
                    sum_value = MUL(A[i][k], B[k][j])
                else:
                    sum_value = ADD(sum_value, MUL(A[i][k], B[k][j]))
            result[i][j] = sum_value
    return result


def mat_vec_mul(A: list[list[Value]], x: list[Value], operators) -> list[Value]:
    """
    Perform matrix-vector multiplication on matrix A and vector x, returning the result as a new vector.
    """
    if len(A[0]) != len(x):
        raise ValueError("Number of columns in A must match length of vector x.")

    # Convert x to a column matrix for mat_mul
    xt = [[v] for v in x]
    mm = mat_mul(A, xt, operators)
    result = [mm[i][0] for i in range(len(mm))]
    return result


def vec_mat_mul(x: list[Value], B: list[list[Value]], operators) -> list[Value]:
    """
    Perform vector-matrix multiplication on vector x and matrix B, returning the result as a new vector.
    """
    if len(x) != len(B):
        raise ValueError("Length of vector x must match number of rows in matrix B.")

    # Treat x as a 1xN row matrix
    mm = mat_mul([x], B, operators)
    result = [mm[0][j] for j in range(len(mm[0]))]
    return result


def poly(coeffs: list[Value], x: Value, operators) -> Value:
    """
    Evaluate a polynomial with given coefficients at a point x.
    Coefficients are in the form [c0, c1, c2, ..., cn] for c0 + c1*x + c2*x^2 + ... + cn*x^n.
    """
    ADD = operators["add"]
    MUL = operators["mul"]

    result = coeffs[0]
    x_power = x  # Start with x^1
    for i in range(1, len(coeffs)):
        result = ADD(result, MUL(coeffs[i], x_power))
        if i < len(coeffs) - 1:  # Don't compute x_power for the last iteration
            x_power = MUL(x_power, x)  # Multiply by x for the next term
    return result


# Constants for DFT.
@cache
def exp_value(k: int, N: int) -> tuple[Value, Value]:
    return Value(f"exp(-2*pi*i*{k}/{N})_r"), Value(f"exp(-2*pi*i*{k}/{N})_i")


def cadd(
    a: tuple[Value, Value], b: tuple[Value, Value], operators
) -> tuple[Value, Value]:
    """
    Add two complex numbers represented as tuples (real, imag).
    """
    ADD = operators["add"]
    return ADD(a[0], b[0]), ADD(a[1], b[1])


def csub(
    a: tuple[Value, Value], b: tuple[Value, Value], operators
) -> tuple[Value, Value]:
    """
    Subtract two complex numbers represented as tuples (real, imag).
    """
    SUB = operators["sub"]
    return SUB(a[0], b[0]), SUB(a[1], b[1])


def cmul(
    a: tuple[Value, Value], b: tuple[Value, Value], operators
) -> tuple[Value, Value]:
    """
    Multiply two complex numbers represented as tuples (real, imag).
    """
    ADD = operators["add"]
    SUB = operators["sub"]
    MUL = operators["mul"]
    return (
        SUB(MUL(a[0], b[0]), MUL(a[1], b[1])),
        ADD(MUL(a[0], b[1]), MUL(a[1], b[0])),
    )


def discrete_fourier_transform(
    x: list[Value], operators
) -> tuple[list[Value], list[Value]]:
    """
    Compute the discrete Fourier transform of a vector x
    Returns two lists: real and imaginary parts of the DFT.
    """

    def dft_recursive(inputs: list[tuple[Value, Value]]) -> list[tuple[Value, Value]]:
        """
        Recursive function to compute the DFT.
        """
        N = len(inputs)
        if N == 1:
            return [inputs[0]]

        even = dft_recursive(inputs[0::2])
        odd = dft_recursive(inputs[1::2])
        assert len(even) == len(odd), "DFT not divisible by 2."
        odd = [cmul(o, exp_value(k, N), operators) for k, o in enumerate(odd)]

        outputs = []
        for i in range(len(even)):
            outputs.append(cadd(even[i], odd[i], operators))
        for i in range(len(even)):
            outputs.append(csub(even[i], odd[i], operators))

        return outputs

    inputs = [(xi, ZERO) for xi in x]
    outputs = dft_recursive(inputs)
    real_outputs = [o[0] for o in outputs]
    imag_outputs = [o[1] for o in outputs]
    return real_outputs, imag_outputs


def hadamard_transform(x: list[Value], operators) -> list[Value]:
    """
    Compute the Hadamard transform of a vector x.
    """
    N = len(x)
    if N == 1:
        return [x[0]]

    even = hadamard_transform(x[0::2], operators)
    odd = hadamard_transform(x[1::2], operators)
    assert len(even) == len(odd), "Hadamard transform not divisible by 2."

    ADD = operators["add"]
    SUB = operators["sub"]

    outputs = []
    for i in range(len(even)):
        outputs.append(ADD(even[i], odd[i]))
    for i in range(len(even)):
        outputs.append(SUB(even[i], odd[i]))

    return outputs


def layer_norm(
    input_vals: list[Value],
    gamma_val: Value,  # This is 'beta' in the standard y = scale * norm(x) + shift
    epsilon_val: Value,
    N_val: Value,  # Symbolic Value for N, e.g., Value("N") from the caller
    operators,
) -> list[Value]:
    """
    Perform layer normalization computation using Value objects and operators.
    This function encapsulates the logic previously in create_layernorm_graph.
    """
    ADD = operators["add"]
    MUL = operators["mul"]
    DIV = operators["div"]
    SQRT = operators["sqrt"]
    SUB = operators["sub"]

    # Mean calculation: E[x]
    # sum_val = sum(input_vals) using tree reduction
    sum_val = binary_vector_tree_reduce(ADD, input_vals)
    mean = DIV(sum_val, N_val)

    # Variance calculation: Var(x) = E[x^2] - (E[x])^2
    # E[x^2]
    squared_inputs = [MUL(x, x) for x in input_vals]
    # sum_squared_val = sum(squared_inputs) using tree reduction
    sum_squared_val = binary_vector_tree_reduce(ADD, squared_inputs)
    square_mean = DIV(sum_squared_val, N_val)  # This is E[x^2]

    # (E[x])^2
    mean_square = MUL(mean, mean)  # This is (E[x])^2

    # Var(x) = E[x^2] - (E[x])^2
    variance = SUB(square_mean, mean_square)

    variance_plus_epsilon = ADD(variance, epsilon_val)  # variance + epsilon
    stddev = SQRT(variance_plus_epsilon)

    # reciprocal_stddev = 1.0 / stddev
    reciprocal_stddev = DIV(ONE, stddev)

    outputs = []
    for x_i in input_vals:
        # term1 = x_i - mean
        term1 = SUB(x_i, mean)

        # term2 = (x_i - mean) / stddev
        term2 = MUL(term1, reciprocal_stddev)

        # result_i = gamma_val + term2
        # Here, gamma_val is the shift parameter (beta in standard formula).
        # The scale parameter (gamma in standard formula) is implicitly 1.0.
        output_val = ADD(gamma_val, term2)
        outputs.append(output_val)

    return outputs


def lstm_cell_forget_gate(
    x_t: list[Value],
    h_t_1: list[Value],
    c_t_1: list[Value],
    weights: dict,
    operators=basic_operators,
) -> tuple[list[Value], list[Value]]:
    """
    Create a single LSTM cell computation graph.
    """
    ADD = operators["add"]
    MUL = operators["mul"]
    TANH = operators["tanh"]
    SIGMOID = operators["sigmoid"]

    w_f = weights["w_f"]
    u_f = weights["u_f"]
    b_f = weights["b_f"]
    w_i = weights["w_i"]
    u_i = weights["u_i"]
    b_i = weights["b_i"]
    w_o = weights["w_o"]
    u_o = weights["u_o"]
    b_o = weights["b_o"]
    w_c = weights["w_c"]
    u_c = weights["u_c"]
    b_c = weights["b_c"]

    f_t = uniary_vector_operation(
        SIGMOID,
        binary_vector_operation(
            ADD,
            binary_vector_operation(
                ADD,
                vec_mat_mul(x_t, w_f, operators),
                vec_mat_mul(h_t_1, u_f, operators),
            ),
            b_f,
        ),
    )

    i_t = uniary_vector_operation(
        SIGMOID,
        binary_vector_operation(
            ADD,
            binary_vector_operation(
                ADD,
                vec_mat_mul(x_t, w_i, operators),
                vec_mat_mul(h_t_1, u_i, operators),
            ),
            b_i,
        ),
    )

    o_t = uniary_vector_operation(
        SIGMOID,
        binary_vector_operation(
            ADD,
            binary_vector_operation(
                ADD,
                vec_mat_mul(x_t, w_o, operators),
                vec_mat_mul(h_t_1, u_o, operators),
            ),
            b_o,
        ),
    )

    c_tilde = uniary_vector_operation(
        TANH,
        binary_vector_operation(
            ADD,
            binary_vector_operation(
                ADD,
                vec_mat_mul(x_t, w_c, operators),
                vec_mat_mul(h_t_1, u_c, operators),
            ),
            b_c,
        ),
    )

    c_t = binary_vector_operation(
        ADD,
        binary_vector_operation(
            MUL,
            f_t,
            c_t_1,
        ),
        binary_vector_operation(
            MUL,
            i_t,
            c_tilde,
        ),
    )
    h_t = binary_vector_operation(
        MUL,
        o_t,
        uniary_vector_operation(TANH, c_t),
    )
    return h_t, c_t


def relu(x: list[Value], operators=basic_operators) -> list[Value]:
    """
    Apply the ReLU activation function to a vector of Values.
    """
    return binary_vector_operation(operators["max"], x, [ZERO] * len(x))


def nas_lstm_cell(
    x_t: list[Value],
    h_t_1: list[Value],
    c_t_1: list[Value],
    weights: dict,
    operators=basic_operators,
) -> tuple[list[Value], list[Value]]:
    """
    Create a single LSTM cell computation graph using the NAS LSTM structure.
    """
    ADD = operators["add"]
    MUL = operators["mul"]
    TANH = operators["tanh"]
    SIGMOID = operators["sigmoid"]
    MAX = operators["max"]

    i1_1 = vec_mat_mul(x_t, weights["i1_1"], operators)
    i1_2 = vec_mat_mul(h_t_1, weights["i1_2"], operators)
    i1 = binary_vector_operation(ADD, i1_1, i1_2)

    i2_1 = vec_mat_mul(x_t, weights["i2_1"], operators)
    i2_2 = vec_mat_mul(h_t_1, weights["i2_2"], operators)
    i2 = binary_vector_operation(ADD, i2_1, i2_2)

    i3_1 = vec_mat_mul(x_t, weights["i3_1"], operators)
    i3_2 = vec_mat_mul(h_t_1, weights["i3_2"], operators)
    i3 = binary_vector_operation(ADD, i3_1, i3_2)

    i4_1 = vec_mat_mul(x_t, weights["i4_1"], operators)
    i4_2 = vec_mat_mul(h_t_1, weights["i4_2"], operators)
    i4 = binary_vector_operation(ADD, i4_1, i4_2)

    i5_1 = vec_mat_mul(x_t, weights["i5_1"], operators)
    i5_2 = vec_mat_mul(h_t_1, weights["i5_2"], operators)
    i5 = binary_vector_operation(ADD, i5_1, i5_2)

    i6_1 = vec_mat_mul(x_t, weights["i6_1"], operators)
    i6_2 = vec_mat_mul(h_t_1, weights["i6_2"], operators)
    i6 = binary_vector_operation(MAX, i6_1, i6_2)

    i7_1 = vec_mat_mul(x_t, weights["i7_1"], operators)
    i7_2 = vec_mat_mul(h_t_1, weights["i7_2"], operators)
    i7 = binary_vector_operation(MAX, i7_1, i7_2)

    i8_1 = vec_mat_mul(x_t, weights["i8_1"], operators)
    i8_2 = vec_mat_mul(h_t_1, weights["i8_2"], operators)
    i8 = binary_vector_operation(MAX, i8_1, i8_2)

    i9 = uniary_vector_operation(TANH, i1)
    i10 = uniary_vector_operation(TANH, i2)
    i11 = [x for x in i3]  # identity
    i12 = relu(i4, operators)
    i13 = uniary_vector_operation(TANH, i5)
    i14 = uniary_vector_operation(TANH, i6)
    i15 = [x for x in i7]  # identity
    i32 = uniary_vector_operation(TANH, i8)

    i16_1 = vec_mat_mul(i32, weights["i16_1"], operators)
    i16_2 = vec_mat_mul(c_t_1, weights["i16_2"], operators)
    i16 = binary_vector_operation(ADD, i16_1, i16_2)

    i17_1 = vec_mat_mul(i10, weights["i17_1"], operators)
    i17_2 = vec_mat_mul(i11, weights["i17_2"], operators)
    i17 = binary_vector_operation(ADD, i17_1, i17_2)

    i18_1 = vec_mat_mul(i9, weights["i18_1"], operators)
    i18_2 = vec_mat_mul(i12, weights["i18_2"], operators)
    i18 = binary_vector_operation(ADD, i18_1, i18_2)

    i19_1 = vec_mat_mul(i13, weights["i19_1"], operators)
    i19_2 = vec_mat_mul(i14, weights["i19_2"], operators)
    i19 = binary_vector_operation(ADD, i19_1, i19_2)

    i20 = [x for x in i16]  # identity

    i21_1 = vec_mat_mul(i15, weights["i21_1"], operators)
    i21_2 = vec_mat_mul(i20, weights["i21_2"], operators)
    i21 = binary_vector_operation(ADD, i21_1, i21_2)

    i22 = uniary_vector_operation(SIGMOID, i17)
    i23 = uniary_vector_operation(SIGMOID, i19)
    i24 = uniary_vector_operation(SIGMOID, i18)
    i25 = [x for x in i21]  # identity

    i26 = binary_vector_operation(MUL, i22, i23)
    i27 = uniary_vector_operation(SIGMOID, i26)

    i28 = binary_vector_operation(MUL, i24, i25)
    i29 = [x for x in i28]  # identity

    i30 = binary_vector_operation(MUL, i27, i29)
    i31 = uniary_vector_operation(TANH, i30)

    h_t = [x for x in i31]  # identity
    c_t = [x for x in i28]  # identity

    return h_t, c_t


def create_layernorm_graph(N: int, operators=basic_operators) -> Context:
    """
    Create a graph for layer normalization of a vector of length N, returning a context.
    """
    # Define input Value nodes. Names match those implicitly used in the original version.
    input_x_vals = [Value(f"x{i}") for i in range(N)]
    # gamma_val is used as the 'beta' (shift) parameter in the layernorm formula.
    gamma_param_node = Value("gamma")
    epsilon_param_node = Value("epsilon")
    # N_param_node is a symbolic Value node representing N, used in division.
    N_param_node = Value("N")

    # Delegate the graph construction logic to the layer_norm function.
    output_vals = layer_norm(
        input_x_vals, gamma_param_node, epsilon_param_node, N_param_node, operators
    )

    return Context(*output_vals)


def create_dft_graph(N: int, operators=basic_operators) -> Context:
    """
    Create a graph for DFT, returning a context.
    """
    inputs = [Value(f"x_{i}") for i in range(N)]
    real_outputs, imag_outputs = discrete_fourier_transform(inputs, operators)
    outputs = real_outputs + imag_outputs
    return Context(*outputs)


def create_lstm_graph(
    N: int,
    timesteps: int,
    load_weights: bool = True,
    operators=basic_operators,
) -> Context:
    """
    Create a graph for LSTM with N features and a given number of timesteps.
    If load_weights is True, it will load the weights using load operators,
    otherwise weights will be created as constant matrices.
    """
    h_t_0 = [Value(f"h_{i}") for i in range(N)]
    c_t_0 = [Value(f"c_{i}") for i in range(N)]

    if load_weights:
        weights = {
            "w_f": load_matrix(N, N, operators),
            "u_f": load_matrix(N, N, operators),
            "b_f": load_vector(N, operators),
            "w_i": load_matrix(N, N, operators),
            "u_i": load_matrix(N, N, operators),
            "b_i": load_vector(N, operators),
            "w_o": load_matrix(N, N, operators),
            "u_o": load_matrix(N, N, operators),
            "b_o": load_vector(N, operators),
            "w_c": load_matrix(N, N, operators),
            "u_c": load_matrix(N, N, operators),
            "b_c": load_vector(N, operators),
        }
    else:
        weights = {
            "w_f": const_matrix(N, N),
            "u_f": const_matrix(N, N),
            "b_f": const_vector(N),
            "w_i": const_matrix(N, N),
            "u_i": const_matrix(N, N),
            "b_i": const_vector(N),
            "w_o": const_matrix(N, N),
            "u_o": const_matrix(N, N),
            "b_o": const_vector(N),
            "w_c": const_matrix(N, N),
            "u_c": const_matrix(N, N),
            "b_c": const_vector(N),
        }

    h_t = h_t_0
    c_t = c_t_0
    for t in range(timesteps):
        x_t = [Value(f"x_{i}_{t}") for i in range(N)]
        h_t, c_t = lstm_cell_forget_gate(x_t, h_t, c_t, weights, operators)

    outputs = h_t + c_t
    return Context(*outputs)


def create_nas_lstm_graph(
    N: int,
    timesteps: int,
    load_weights: bool = True,
    operators=basic_operators,
) -> Context:
    """
    Create a graph for NAS LSTM with N features and a given number of timesteps.
    If load_weights is True, it will load the weights using load operators,
    otherwise weights will be created as constant matrices.
    """
    h_t_0 = [Value(f"h_{i}") for i in range(N)]
    c_t_0 = [Value(f"c_{i}") for i in range(N)]

    def make_weights(prefixes):
        if load_weights:
            return {k: load_matrix(N, N, operators) for k in prefixes}
        else:
            return {k: const_matrix(N, N) for k in prefixes}

    # All weight keys used in nas_lstm_cell
    weight_keys = [
        "i1_1",
        "i1_2",
        "i2_1",
        "i2_2",
        "i3_1",
        "i3_2",
        "i4_1",
        "i4_2",
        "i5_1",
        "i5_2",
        "i6_1",
        "i6_2",
        "i7_1",
        "i7_2",
        "i8_1",
        "i8_2",
        "i16_1",
        "i16_2",
        "i17_1",
        "i17_2",
        "i18_1",
        "i18_2",
        "i19_1",
        "i19_2",
        "i21_1",
        "i21_2",
    ]
    weights = make_weights(weight_keys)

    h_t = h_t_0
    c_t = c_t_0
    for t in range(timesteps):
        x_t = [Value(f"x_{i}_{t}") for i in range(N)]
        h_t, c_t = nas_lstm_cell(x_t, h_t, c_t, weights, operators)

    outputs = h_t + c_t
    return Context(*outputs)


def create_hadamard_transform_graph(N: int, operators=basic_operators) -> Context:
    """
    Create a graph for the Hadamard transform of a vector of length N, returning a context.
    """
    inputs = [Value(f"x_{i}") for i in range(N)]
    outputs = hadamard_transform(inputs, operators)
    return Context(*outputs)


def create_parallel_linear_adds_graph(
    depth: int, width: int, operators=basic_operators
):
    """
    Create a graph for parallel linear adds, returning a context.
    """
    outputs = []
    for i in range(width):
        chain = [Value(f"x_{j}_{i}") for j in range(depth)]
        sum = binary_vector_linear_reduce(operators["add"], chain)
        outputs.append(sum)

    return Context(*outputs)


def create_parallel_binary_tree_adds_graph(
    depth: int, width: int, operators=basic_operators
):
    """
    Create a graph for parallel binary tree adds, returning a context.
    """
    outputs = []
    for i in range(width):
        chain = [Value(f"x_{j}_{i}") for j in range(depth)]
        sum = binary_vector_tree_reduce(operators["add"], chain)
        outputs.append(sum)

    return Context(*outputs)

def create_parallel_poly_graph(
        degree: int, width: int, operators=basic_operators
):
    """
    Create a graph for parallel polynomial evaluation, returning a context.
    """
    outputs = []
    for i in range(width):
        coeffs = [Value(f"c_{j}_{i}") for j in range(degree + 1)]
        x = Value(f"x_{i}")
        poly_val = poly(coeffs, x, operators)
        outputs.append(poly_val)

    return Context(*outputs)


def create_bagged_graph(N: int, operators=basic_operators) -> Context:
    """
    Create a graph that schedules both a DFT and a Hadamard transform on the same input vector.
    Returns a context containing all outputs.
    """
    # Shared input vector
    inputs = [Value(f"x_{i}") for i in range(N)]

    # DFT outputs
    real_outputs, imag_outputs = discrete_fourier_transform(inputs, operators)

    # Hadamard outputs
    hadamard_outputs = hadamard_transform(inputs, operators)

    # Combine all outputs into a single context
    outputs = real_outputs + imag_outputs + hadamard_outputs
    return Context(*outputs)


__all__ = [
    "create_layernorm_graph",
    "create_dft_graph",
    "create_lstm_graph",
    "create_nas_lstm_graph",
    "create_hadamard_transform_graph",
    "create_parallel_linear_adds_graph",
    "create_parallel_binary_tree_adds_graph",
    "create_parallel_poly_graph",
]
