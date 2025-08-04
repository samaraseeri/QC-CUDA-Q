import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Minimal CUDA-Q Modifications for Real Hardware ---
_CUDAQ_AVAILABLE = False
USE_REAL_HARDWARE = True  # **CHANGE THIS TO True FOR REAL QUANTUM HARDWARE**

try:
    import cudaq
    _ = cudaq.kernel
    _CUDAQ_AVAILABLE = True
    print("[CUDA-Q] CUDA-Q imported successfully.")
    
    # **MINIMAL CHANGE 1: Choose your quantum hardware**
    if USE_REAL_HARDWARE:
        # **UNCOMMENT ONE OF THESE LINES FOR REAL HARDWARE:**
        
        # For IonQ (need: export IONQ_API_KEY="your_key"):
        cudaq.set_target('ionq', qpu='qpu.aria-1')
        print("[CUDA-Q] Using IonQ Aria-1 real quantum hardware")
        
        # For Amazon Braket (need AWS credentials):
        # cudaq.set_target('braket', machine='arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1')
        # print("[CUDA-Q] Using Amazon Braket IonQ via CUDA-Q")
        
        # For Quantinuum (need Quantinuum credentials):
        # cudaq.set_target('quantinuum', machine='H1-1')
        # print("[CUDA-Q] Using Quantinuum H1-1")
        
    else:
        # Keep your original local simulation
        cudaq.set_target("nvidia")
        print("[CUDA-Q] Using local NVIDIA GPU simulation")
    
    # **YOUR ORIGINAL KERNEL STAYS EXACTLY THE SAME:**
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
    
    # **MINIMAL CHANGE 2: Modify this function to handle both simulation and real hardware**
    def _get_state_from_cudaq(num_qubits, depth, params):
        if USE_REAL_HARDWARE:
            # **FOR REAL HARDWARE: Use sampling instead of get_state**
            print("[CUDA-Q] Running on real quantum hardware - using sampling")
            shots = 4096  # Adjust based on your budget and accuracy needs
            
            # Sample from your exact same kernel
            result = cudaq.sample(_real_cudaq_vqc_kernel, num_qubits, depth, params, shots_count=shots)
            
            # Convert measurement counts to approximate state amplitudes
            counts = result.get_register_counts()
            state_len = 2**num_qubits
            amplitudes = np.zeros(state_len, dtype=complex)
            
            # Reconstruct approximate amplitudes from measurement statistics
            for bitstring, count in counts.items():
                index = int(bitstring, 2)
                amplitudes[index] = np.sqrt(count / shots)
            
            # Add random phases (limitation of measurement-based reconstruction)
            phases = np.random.uniform(0, 2*np.pi, state_len)
            amplitudes = amplitudes * np.exp(1j * phases)
            
            # Normalize
            norm = np.linalg.norm(amplitudes)
            if norm > 0:
                amplitudes = amplitudes / norm
                
            return amplitudes
        else:
            # **FOR SIMULATION: Keep your original method**
            return cudaq.get_state(_real_cudaq_vqc_kernel, num_qubits, depth, params)

except (ImportError, AttributeError) as e:
    _CUDAQ_AVAILABLE = False
    print(f"[CUDA-Q] CUDA-Q not available: {e}. Falling back to dummy simulation.")

    def _get_state_from_cudaq(num_qubits, depth, params):
        state_len = 2**num_qubits
        dummy_amplitudes = np.random.rand(state_len) + 1j * np.random.rand(state_len)
        dummy_amplitudes = dummy_amplitudes / np.linalg.norm(dummy_amplitudes)
        return dummy_amplitudes


# --- ALL YOUR ORIGINAL FUNCTIONS STAY EXACTLY THE SAME ---

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

def quantum_evaluate_u_at_nodes_cudaq(params, num_qubits, depth, x_nodes, scaling_lambda, Cheb_func):
    """YOUR ORIGINAL FUNCTION - NO CHANGES NEEDED"""
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
    """YOUR ORIGINAL FUNCTION WITH MINIMAL CHANGES"""
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

    # **MINIMAL CHANGE 3: Adjust iterations for real hardware costs**
    if USE_REAL_HARDWARE:
        max_iter = 20  # Fewer iterations to manage costs on real hardware
        print(f"\n[Quantum-Enhanced Solver] Using REAL QUANTUM HARDWARE with {num_vqc_params} parameters")
        print("[WARNING] This will cost money! Each iteration uses ~4096 shots.")
    else:
        max_iter = 100  # Your original number for simulation
        print(f"\n[Quantum-Enhanced Solver] Using simulation with {num_vqc_params} parameters")

    result = minimize(residual_quantum_enhanced, initial_vqc_params, method='BFGS', 
                     options={'disp': True, 'maxiter': max_iter})

    if not result.success:
        print(f"  [Quantum-Enhanced Solver] ! Optimization did not converge: {result.message}")

    optimal_vqc_params = result.x
    print(f"  [Quantum-Enhanced Solver] Optimization finished. Final loss: {result.fun}")

    final_u_solution_at_nodes = quantum_evaluate_u_at_nodes_cudaq(optimal_vqc_params, num_qubits, depth, x_nodes, scaling_lambda, Cheb)
    return final_u_solution_at_nodes


# --- YOUR ORIGINAL EXAMPLE USAGE WITH MINIMAL CHANGES ---

if __name__ == '__main__':
    # **MINIMAL CHANGE 4: Adjust circuit size for real hardware noise**
    if USE_REAL_HARDWARE:
        num_qubits_nonlin = 4  # Smaller for real hardware
        depth_nonlin = 2
        print(f"[System] Using reduced circuit for real hardware: {num_qubits_nonlin} qubits, depth {depth_nonlin}")
    else:
        num_qubits_nonlin = 6  # Your original values
        depth_nonlin = 3
        print(f"[System] Using original circuit for simulation: {num_qubits_nonlin} qubits, depth {depth_nonlin}")
    
    # --- ALL YOUR ORIGINAL CODE BELOW STAYS THE SAME ---
    
    alpha_const, beta_const = 1, 3
    N_const = 50
    f_const = lambda x: 2 * np.ones_like(x)
   
    print("--- Running Classical Linear Constant Coeff Solver ---")
    u_const_solution = solve_linear_constant_ode_cheb(alpha_const, beta_const, N_const, f_const)
    print("Linear Constant Solution (first/last 5):", u_const_solution[:5], "...", u_const_solution[-5:])

    alpha_var, beta_var = 1, 3
    N_var = 50
    a_x_var = lambda x: x + 1
    f_var = lambda x: np.cos(x) - (x + 1) * np.sin(x)

    print("\n--- Running Classical Linear Variable Coeff Solver ---")
    u_var_solution = solve_linear_variable_ode_cheb(alpha_var, beta_var, N_var, a_x_var, f_var)
    print("Linear Variable Solution (first/last 5):", u_var_solution[:5], "...", u_var_solution[-5:])

    alpha_nonlin, beta_nonlin = 2, np.exp(2) + 1
    N_nonlin = 50
    a_u_nonlin = lambda u: u + 1
    f_nonlin = lambda x: np.exp(x) * (np.exp(x) + x + 1)

    print("\n--- Running Quantum-Enhanced Nonlinear Solver ---")
    u_nonlin_solution_quantum_enhanced = solve_nonlinear_ode_cheb_quantum_enhanced(
        alpha_nonlin, beta_nonlin, N_nonlin, a_u_nonlin, f_nonlin,
        num_qubits=num_qubits_nonlin, depth=depth_nonlin
    )
    print("Nonlinear Solution (first/last 5):", u_nonlin_solution_quantum_enhanced[:5], "...", u_nonlin_solution_quantum_enhanced[-5:])

    # --- YOUR ORIGINAL PLOTTING CODE ---
    x_plot = chebyshev_nodes(N_nonlin, 0, 2)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, u_const_solution, label="Linear Constant Coeff (Classical)", linestyle='--')
    plt.plot(x_plot, u_var_solution, label="Linear Variable Coeff (Classical)", linestyle=':')
    
    hardware_label = "Real Quantum Hardware" if USE_REAL_HARDWARE else "GPU Simulation"
    plt.plot(x_plot, u_nonlin_solution_quantum_enhanced, label=f"Nonlinear (CUDA-Q {hardware_label})", color='red')

    u_exact_nonlin = lambda x: np.exp(x) + 1
    plt.plot(x_plot, u_exact_nonlin(x_plot), label="Nonlinear Exact Solution", linestyle='-', color='blue')

    plt.legend()
    plt.grid(True)
    plt.title(f"ODE Solutions ({'Real Hardware' if USE_REAL_HARDWARE else 'Simulation'})")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.show()

# **SETUP INSTRUCTIONS:**
# 1. Set USE_REAL_HARDWARE = True (line 7)
# 2. Uncomment one of the cudaq.set_target() lines (lines 15-25)
# 3. Set up credentials for your chosen provider:
#    - IonQ: export IONQ_API_KEY="your_key"
#    - Braket: export AWS_ACCESS_KEY_ID="key" and AWS_SECRET_ACCESS_KEY="secret"
#    - Quantinuum: Set up Quantinuum credentials
# 4. Run the code!
