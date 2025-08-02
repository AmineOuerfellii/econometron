import numpy as np
from scipy.linalg import ordqz
from scipy.linalg import qz
from sympy import Symbol

__all__ = ['klein_solver_func']

def klein_solver_func(linearized_system, n_states,variables):
    n_vars = len(variables)
    A = np.zeros((n_vars, n_vars))
    B = np.zeros((n_vars, n_vars))
    for i, eq in enumerate(linearized_system):
        for var in variables:
            tp1_term = eq.get(Symbol(f"hat_{var}_tp1"), 0)
            j = variables.index(var)
            A[i, j] = float(tp1_term)
            t_term = eq.get(Symbol(f"hat_{var}_t"), 0)
            B[i, j] = float(t_term)
    s, t, _, _, q, z = ordqz(A, B, sort='ouc', output='complex')
    z21 = z[n_states:, :n_states]
    z11 = z[:n_states, :n_states]
    if np.linalg.matrix_rank(z11) < n_states:
        raise ValueError("Invertibility condition not satisfied.")
    z11i = np.linalg.inv(z11)
    s11 = s[:n_states, :n_states]
    t11 = t[:n_states, :n_states]
    dyn = np.linalg.solve(s11, t11)
    f = np.real(z21 @ z11i)  # Shape: (n_costates, n_states)
    p = np.real(z11 @ dyn @ z11i)  # Shape: (n_states, n_states)
    return f, p

# def sims_solver_func(linearized_system, n_states, n_exo_states, variables, states, shock_names):
#     n_vars = len(variables)
#     n_costates = n_vars - n_states

#     Gamma_0 = np.zeros((n_vars, n_vars))
#     Gamma_1 = np.zeros((n_vars, n_vars))
#     for i, eq in enumerate(linearized_system):
#         for var in variables:
#             t_term = eq.get(Symbol(f"hat_{var}_t"), 0)
#             j = variables.index(var)
#             Gamma_0[i, j] = float(t_term)
#             tp1_term = eq.get(Symbol(f"hat_{var}_tp1"), 0)
#             Gamma_1[i, j] = float(tp1_term)

#     S, T, Q, Z = qz(Gamma_0, Gamma_1, output='real')  # Use real output
#     eigvals = np.abs(np.diag(T) / (np.diag(S) + 1e-10))
#     stable_idx = eigvals < 1
#     n_stable = np.sum(stable_idx)

#     if n_stable != n_states:
#         raise ValueError(f"Solvability condition not satisfied: {n_stable} stable roots, expected {n_states}")

#     Z11 = Z[:n_states, :n_states]
#     Z21 = Z[n_states:, :n_states]
#     S11 = S[:n_states, :n_states]
#     T11 = T[:n_states, :n_states]

#     # Compute p using stable dynamics
#     p = np.real(np.linalg.solve(S11, T11))  # Shape: (n_states, n_states)
#     # Compute f using stable manifold
#     f = np.real(Z21 @ np.linalg.inv(Z11))  # Shape: (n_costates, n_states)

#     return f, p

# def blanchard_kahn_solver_func(linearized_system, n_states, n_exo_states, variables, states, shock_names):
#     """Solve DSGE model using Blanchard-Kahn method, returning f and p matrices."""
#     n_vars = len(variables)
#     n_costates = n_vars - n_states

#     # Construct A and B matrices from linearized_system
#     A = np.zeros((n_vars, n_vars))
#     B = np.zeros((n_vars, n_vars))
#     for i, eq in enumerate(linearized_system):
#         for var in variables:
#             # Coefficients for t+1 terms (A matrix)
#             tp1_term = eq.get(Symbol(f"hat_{var}_tp1"), 0)
#             j = variables.index(var)
#             A[i, j] = float(tp1_term)
#             # Coefficients for t terms (B matrix)
#             t_term = eq.get(Symbol(f"hat_{var}_t"), 0)
#             B[i, j] = float(t_term)

#     # Compute eigenvalues and eigenvectors
#     eigvals, eigvecs = np.linalg.eig(np.linalg.inv(A + 1e-10) @ B)  # Avoid singular A
#     n_stable = np.sum(np.abs(eigvals) < 1)

#     if n_stable != n_states:
#         raise ValueError("Saddle-path condition not satisfied")

#     stable_idx = np.abs(eigvals) < 1
#     V_s = eigvecs[:, stable_idx]

#     # Compute p and f matrices
#     p = np.real(V_s[:n_states, :n_states])  # Shape: (n_states, n_states)
#     f = np.zeros((n_costates, n_states))  
#     # Approximate f using the relationship between controls and states
#     if n_costates > 0:
#         f = -np.real(np.linalg.pinv(A[n_states:, :n_states]) @ B[n_states:, :n_states])

#     return f, p