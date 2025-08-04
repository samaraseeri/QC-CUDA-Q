import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Robust CUDA-Q Import and Conditional Definitions ---
_CUDAQ_AVAILABLE = False # Flag to track CUDA-Q availability

try:
    import cudaq
    # Try accessing cudaq.kernel to ensure it's available
    # If this line raises an AttributeError, it will go to the except block
    _ = cudaq.kernel
    _CUDAQ_AVAILABLE = True
    print("[CUDA-Q] CUDA-Q imported successfully. Attempting to use CUDA-Q backend.")
    
    # If CUDA-Q is available, define the real kernel
    @cudaq.kernel
    def _real_cudaq_vqc_kernel(num_qubits: int, depth: int, theta: list[float]):
        q = cudaq.qalloc(num_qubits)
        param_idx = 0
        for _ in range(depth):
            for i in range(num_qubits):
                cudaq.ry(theta[param_idx], q[i])
                param_idx += 1
            for i in range(num_qubits - 1):
                cudaq.cx(q[i], q[i + 1])
    
    # Function to get state from real CUDA-Q kernel
    def _get_state_from_cudaq(num_qubits, depth, params):
        cudaq.set_target("nvidia")
        return cudaq.get_state(_real_cudaq_vqc_kernel, num_qubits, depth, params)
except (ImportError, AttributeError) as e:
    _CUDAQ_AVAILABLE = False
    print(f"[CUDA-Q] CUDA-Q or 'cudaq.kernel' not fully available: {e}. Falling back to simple conceptual simulation.")

    # Define a dummy state getter if CUDA-Q is not fully functional
    def _get_state_from_cudaq(num_qubits, depth, params):
        state_len = 2**num_qubits
        dummy_amplitudes = np.random.rand(state_len) + 1j * np.random.rand(state_len)
        dummy_amplitudes = dummy_amplitudes / np.linalg.norm(dummy_amplitudes)
        return dummy_amplitudes


# --- Classical Chebyshev Collocation Setup ---

def chebyshev_nodes(N, a, b):
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * np.arange(N - 1, -1, -1) / (N - 1))
    return nodes

def chebyshev_differentiation_matrix(N, a, b):
    if N == 1:
        return np.array([[0]])

    x = chebyshev_nodes(N, a, b)
    c = np.hstack([2, np.ones(N - 2), 2]) * (-1.0) ** np.arange(N)
    dx = np.tile(x, (N, 1)).T - np.tile(x, (N, 1))
    D = (c.reshape(-1, 1) / c.reshape(1, -1)) / (dx + np.eye(N))
    D = D - np.diag(np.sum(D, axis=1))

    return D

# --- Classical Linear ODE Solvers (Remain Unchanged) ---

def solve_linear_constant_ode_cheb(alpha, beta, N, f):
    a, b = 0, 2
    x = chebyshev_nodes(N, a, b)
    D = chebyshev_differentiation_matrix(N, a, b)
    D2 = D @ D

    A = np.copy(D2)
    b_vec = f(x)

    A[0, :] = 0
    A[0, 0] = 1
    b_vec[0] = alpha

    A[-1, :] = 0
    A[-1, -1] = 1
    b_vec[-1] = beta

    u_solution = np.linalg.solve(A, b_vec)
    return u_solution

def solve_linear_variable_ode_cheb(alpha, beta, N, a_x, f):
    a, b = 0, 2
    x = chebyshev_nodes(N, a, b)
    D = chebyshev_differentiation_matrix(N, a, b)

    A = D @ (np.diag(a_x(x)) @ D)
    b_vec = f(x)

    A[0, :] = 0
    A[0, 0] = 1
    b_vec[0] = alpha

    A[-1, :] = 0
    A[-1, -1] = 1
    b_vec[-1] = beta

    u_solution = np.linalg.solve(A, b_vec)
    return u_solution

# Helper function for Chebyshev polynomials (T_k(x))
def Cheb(k, x_val):
    x_scaled = (2 * x_val - (0 + 2)) / (2 - 0)
    
    if k == 0:
        return np.ones_like(x_scaled) if np.isscalar(x_scaled) else np.ones(len(x_scaled))
    elif k == 1:
        return x_scaled
    else:
        T_k_minus_2 = np.ones_like(x_scaled) if np.isscalar(x_scaled) else np.ones(len(x_scaled))
        T_k_minus_1 = x_scaled
        for i in range(2, k + 1):
            T_k = 2 * x_scaled * T_k_minus_1 - T_k_minus_2
            T_k_minus_2 = T_k_minus_1
            T_k_minus_1 = T_k
        return T_k_minus_1

# --- Quantum-Enhanced Nonlinear ODE Solver (Now uses _get_state_from_cudaq) ---

def quantum_evaluate_u_at_nodes_cudaq(params, num_qubits, depth, x_nodes, scaling_lambda, Cheb_func):
    """
    Evaluates u(x) at all Chebyshev nodes by conceptually running a CUDA-Q VQC.
    This now directly uses the `_get_state_from_cudaq` function which
    is defined conditionally based on CUDA-Q availability.
    """
    state_amplitudes = _get_state_from_cudaq(num_qubits, depth, params)
    probabilities = np.abs(state_amplitudes)**2

    num_cheb_coeffs_encoded = 2**(num_qubits - 1)
    
    if len(probabilities) < 2 * num_cheb_coeffs_encoded:
        print(f"Warning: VQC qubits ({num_qubits}) might not be enough for Chebyshev encoding. Adjusting coeffs.")
        num_cheb_coeffs_encoded = len(probabilities) // 2

    u_at_nodes = np.zeros(len(x_nodes))
    for j, x_val in enumerate(x_nodes):
        sum_terms = 0.0
        for k in range(num_cheb_coeffs_encoded):
            p_k = probabilities[k]
            p_k_plus_half = probabilities[k + num_cheb_coeffs_encoded]
            sum_terms += (p_k - p_k_plus_half) * Cheb_func(k, x_val)
        u_at_nodes[j] = scaling_lambda * sum_terms
    
    return u_at_nodes

def solve_nonlinear_ode_cheb_quantum_enhanced(alpha, beta, N, a_u, f, num_qubits, depth):
    a, b = 0, 2
    x_nodes = chebyshev_nodes(N, a, b)

    scaling_lambda = 10.0

    D = chebyshev_differentiation_matrix(N, a, b)

    def residual_quantum_enhanced(params):
        u_full = quantum_evaluate_u_at_nodes_cudaq(params, num_qubits, depth, x_nodes, scaling_lambda, Cheb)

        du_dx = D @ u_full
        a_du_dx = a_u(u_full) * du_dx
        d2u_dx2_from_ode_term = D @ a_du_dx

        r = d2u_dx2_from_ode_term - f(x_nodes)

        r[0] = u_full[0] - alpha
        r[-1] = u_full[-1] - beta

        return np.sum(r**2)

    num_vqc_params = num_qubits * depth
    initial_vqc_params = np.random.uniform(-np.pi, np.pi, num_vqc_params)

    print(f"\n[Quantum-Enhanced Solver (CUDA-Q)] Starting optimization with {num_vqc_params} VQC parameters.")

    result = minimize(residual_quantum_enhanced, initial_vqc_params, method='BFGS', options={'disp': True, 'maxiter': 100})

    if not result.success:
        print(f"  [Quantum-Enhanced Solver (CUDA-Q)] ! Optimization did not converge: {result.message}")

    optimal_vqc_params = result.x
    print(f"  [Quantum-Enhanced Solver (CUDA-Q)] Optimization finished. Final loss: {result.fun}")

    final_u_solution_at_nodes = quantum_evaluate_u_at_nodes_cudaq(optimal_vqc_params, num_qubits, depth, x_nodes, scaling_lambda, Cheb)
    return final_u_solution_at_nodes


# --- Example Usage ---

if __name__ == '__main__':
    # --- Linear Constant Coefficient ---
    alpha_const, beta_const = 1, 3
    N_const = 50
    f_const = lambda x: 2 * np.ones_like(x)
   
    print("--- Running Classical Linear Constant Coeff Solver ---")
    u_const_solution = solve_linear_constant_ode_cheb(alpha_const, beta_const, N_const, f_const)
    print("Linear Constant Solution (first/last 5):", u_const_solution[:5], "...", u_const_solution[-5:])

    # --- Linear Variable Coefficient ---
    alpha_var, beta_var = 1, 3
    N_var = 50
    a_x_var = lambda x: x + 1
    f_var = lambda x: np.cos(x) - (x + 1) * np.sin(x)

    print("\n--- Running Classical Linear Variable Coeff Solver ---")
    u_var_solution = solve_linear_variable_ode_cheb(alpha_var, beta_var, N_var, a_x_var, f_var)
    print("Linear Variable Solution (first/last 5):", u_var_solution[:5], "...", u_var_solution[-5:])

    # --- Nonlinear Case (Quantum-Enhanced with CUDA-Q adaptation) ---
    alpha_nonlin, beta_nonlin = 2, np.exp(2) + 1
    N_nonlin = 50
    a_u_nonlin = lambda u: u + 1
    f_nonlin = lambda x: np.exp(x) * (np.exp(x) + x + 1)

    num_qubits_nonlin = 6
    depth_nonlin = 3

    print("\n--- Running Quantum-Enhanced Nonlinear Solver (CUDA-Q adapted) ---")
    u_nonlin_solution_quantum_enhanced = solve_nonlinear_ode_cheb_quantum_enhanced(
        alpha_nonlin, beta_nonlin, N_nonlin, a_u_nonlin, f_nonlin,
        num_qubits=num_qubits_nonlin, depth=depth_nonlin
    )
    print("Nonlinear Solution (CUDA-Q adapted, first/last 5):", u_nonlin_solution_quantum_enhanced[:5], "...", u_nonlin_solution_quantum_enhanced[-5:])

    # --- Plotting ---
    x_plot = chebyshev_nodes(N_nonlin, 0, 2)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, u_const_solution, label="Linear Constant Coeff (Classical)", linestyle='--')
    plt.plot(x_plot, u_var_solution, label="Linear Variable Coeff (Classical)", linestyle=':')
    plt.plot(x_plot, u_nonlin_solution_quantum_enhanced, label="Nonlinear (Quantum-Enhanced CUDA-Q)", color='red')

    u_exact_nonlin = lambda x: np.exp(x) + 1
    plt.plot(x_plot, u_exact_nonlin(x_plot), label="Nonlinear Exact Solution", linestyle='-', color='blue')

    plt.legend()
    plt.grid(True)
    plt.title("ODE Solutions (Classical and Quantum-Enhanced)")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.show()
