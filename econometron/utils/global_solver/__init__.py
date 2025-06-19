from itertools import product
import numpy as np
from scipy.special import chebyt
from scipy.optimize import root


def cd_jac(f, x, n, eps=None):
    """Compute central difference Jacobian."""
    if eps is None:
        eps = np.finfo(float).eps
    m = len(x)
    df = np.zeros((n, m))
    h0 = eps ** (1/3)
    x1 = x.copy()
    x2 = x.copy()
    for i in range(m):
        h = h0 * max(abs(x[i]), 1.0) * (-1 if x[i] < 0 else 1)
        x1[i] = x[i] + h
        x2[i] = x[i] - h
        h = x1[i] - x[i]
        f1 = f(x1)
        f2 = f(x2)
        for j in range(n):
            df[j, i] = (f1[j] - f2[j]) / (2 * h)
        x1[i] = x[i]
        x2[i] = x[i]
    return df

def nr_step(x0, dx0, dg, f, smult=1e-4, smin=0.1, smax=0.5, stol=1e-11):
    """Line search for Newton-Raphson."""
    g0 = 0.5 * np.sum(f(x0) ** 2)
    dgdx = dg @ dx0
    s1 = 1.0
    g1 = 0.5 * np.sum(f(x0 + dx0) ** 2)
    if g1 <= g0 + smult * dgdx:
        return s1
    s = -dgdx / (2 * (g1 - g0 - dgdx))
    s = min(max(s, smin), smax)
    x1 = x0 + s * dx0
    g2 = 0.5 * np.sum(f(x1) ** 2)
    s2 = s
    while g2 > g0 + smult * s2 * dgdx:
        amat = np.array([[1/s2**2, -1/s1**2], [-s1/s2**2, s2/s1**2]])
        bvec = np.array([g2 - s2 * dgdx - g0, g1 - s1 * dgdx - g0])
        ab = np.linalg.solve(amat, bvec) / (s2 - s1)
        if ab[0] == 0:
            s = -dgdx / (2 * ab[1])
        else:
            disc = ab[1]**2 - 3 * ab[0] * dgdx
            if disc < 0:
                s = s2 * smax
            elif ab[1] <= 0:
                s = (-ab[1] + np.sqrt(disc)) / (3 * ab[0])
            else:
                s = -dgdx / (ab[1] + np.sqrt(disc))
        s = min(max(s, s2 * smin), s2 * smax)
        tol = np.sqrt(np.sum((s * dx0)**2)) / (1 + np.sqrt(np.sum(x0**2)))
        if tol < stol:
            return -1.0
        s1, s2, g1 = s2, s, g2
        x1 = x0 + s2 * dx0
        g2 = 0.5 * np.sum(f(x1) ** 2)
    return s2

def fixv_mn1(x0, f, maxit=5000, stopc=1e-8, use_global=True, use_qr=False, verbose=False):
    """Modified Newton-Raphson solver."""
    x1 = x0.copy()
    crit = np.ones(5)
    crit[0] = 0
    critold = 2
    itn = 0
    lam = 1e-6
    lam_max = 1e6
    lam_mult = 10.0
    min_step = 1e-8
    while itn < maxit and crit[1] >= stopc:
        print(f"[Newton] Iteration {itn}, coeffs[:5]: {x1[:5]}")
        fx = f(x1)
        df = cd_jac(f, x1, len(fx))
        if np.any(np.isnan(df)):
            crit[0] = 1
            print("Jacobian contains NaN. Aborting.")
            return x1, crit
        jac_cond = np.linalg.cond(df)
        obj_val = 0.5 * np.sum(fx ** 2)
        if verbose:
            print(f"Step {itn}: Convergence = {crit[1]:.2e}, Objective = {obj_val:.2e}, Cond = {jac_cond:.2e}")
        reg = lam * np.eye(df.shape[1])
        try:
            if use_qr:
                q, r = np.linalg.qr(df)
                dx = np.linalg.solve(r + lam * np.eye(r.shape[0]), q.T @ (-fx))
            else:
                JTJ = df.T @ df
                JTF = df.T @ fx
                dx = np.linalg.solve(JTJ + reg, -JTF)
        except np.linalg.LinAlgError:
            print("LinAlgError in Newton step. Increasing regularization.")
            lam = min(lam * lam_mult, lam_max)
            continue
        step_norm = np.linalg.norm(dx)
        if verbose:
            print(f"  Newton step norm = {step_norm:.2e}, lambda = {lam:.1e}")
        step = 1.0
        x_trial = x1 + step * dx
        f_trial = f(x_trial)
        obj_trial = 0.5 * np.sum(f_trial ** 2)
        while (not np.all(np.isfinite(f_trial)) or obj_trial > obj_val) and step > min_step:
            step *= 0.5
            x_trial = x1 + step * dx
            f_trial = f(x_trial)
            obj_trial = 0.5 * np.sum(f_trial ** 2)
            if verbose:
                print(f"    Backtracking: step = {step:.2e}, obj = {obj_trial:.2e}")
        if step <= min_step:
            lam = min(lam * lam_mult, lam_max)
            if verbose:
                print(f"    Step too small, increasing lambda to {lam:.1e}")
            continue
        else:
            lam = max(lam / lam_mult, 1e-12)
        x2 = x1 + step * dx
        crit[1] = np.max(np.abs(f(x2)))
        crit[2] = np.max(np.abs(step * dx) / np.maximum(np.abs(x2), 1.0))
        critold = crit[3]
        crit[3] = 0.5 * np.sum(f(x2) ** 2)
        x1 = x2
        itn += 1
        crit[4] = itn
    if itn >= maxit:
        crit[0] = 3
    return x1, crit

def grad_test(df, x, fx):
    """Compute relative gradient norm."""
    crit = np.abs(df) * np.maximum(np.abs(x), 1.0) / np.maximum(np.abs(fx), 1.0)
    return np.max(crit)

def par_test(x, dx):
    """Compute relative parameter change."""
    return np.max(np.abs(dx) / np.maximum(np.abs(x), 1.0))

def qn_step(x0, dx0, f0, df, f, smult=1e-4, smin=0.1, smax=0.5, ptol=1e-12):
    """Line search for Quasi-Newton."""
    dfdx = df @ dx0
    s1 = 1.0
    f1 = f(x0 + dx0)
    if f1 <= f0 + smult * dfdx:
        return s1, 0
    s = -dfdx / (2 * (f1 - f0 - dfdx))
    s = min(max(s, smin), smax)
    x1 = x0 + s * dx0
    f2 = f(x1)
    s2 = s
    while f2 > f0 + smult * s2 * dfdx:
        amat = np.array([[1/s2**2, -1/s1**2], [-s1/s2**2, s2/s1**2]])
        bvec = np.array([f2 - s2 * dfdx - f0, f1 - s1 * dfdx - f0])
        ab = np.linalg.solve(amat, bvec) / (s2 - s1)
        if ab[0] == 0:
            s = -dfdx / (2 * ab[1])
        else:
            disc = ab[1]**2 - 3 * ab[0] * dfdx
            if disc < 0:
                s = s2 * smax
            elif ab[1] <= 0:
                s = (-ab[1] + np.sqrt(disc)) / (3 * ab[0])
            else:
                s = -dfdx / (ab[1] + np.sqrt(disc))
        s = min(max(s, s2 * smin), s2 * smax)
        if s < ptol:
            return s, 1
        s1, s2, f1 = s2, s, f2
        x1 = x0 + s2 * dx0
        f2 = f(x1)
    return s2, 0

def quasi_newton(x0, f, maxit=500, gtol=None, ptol=1e-7, verbose=False):
    """Quasi-Newton minimizer with BFGS update."""
    if gtol is None:
        gtol = np.finfo(float).eps ** (1/3)
    crit = np.zeros(5)
    h = np.eye(len(x0))
    x1 = x0.copy()
    f1 = f(x1)
    df1 = cd_jac(lambda x: np.array([f(x)]), x1, 1).flatten()
    crit[1] = grad_test(df1, x1, f1)
    if crit[1] < 1e-3 * gtol:
        crit[0], crit[3], crit[4] = 0, f1, 0
        return x1, crit
    itn = 1
    crit[2] = 1
    while itn < maxit:
        if verbose:
            print(f"Iteration {itn}: gTol = {crit[1]:.2e}, pTol = {crit[2]:.2e}, f(x) = {crit[3]:.2e}")
        dx = np.linalg.solve(h, -df1)
        step1 = 1.0
        while np.isnan(f(x1 + step1 * dx)):
            step1 /= 2
            if step1 < 1e-16:
                crit[0] = 1
                return x1, crit
        dx = step1 * dx
        step2, rc = qn_step(x1, dx, f1, df1, f)
        dx = step2 * dx
        x2 = x1 + dx
        f2 = f(x2)
        crit[3] = f2
        df2 = cd_jac(lambda x: np.array([f(x)]), x2, 1).flatten()
        crit[1] = grad_test(df2, x2, f2)
        crit[2] = par_test(x2, dx)
        if crit[1] > gtol and rc:
            crit[0] = 2
            return x2, crit
        if crit[1] < gtol or crit[2] < ptol:
            crit[0] = 0
            return x2, crit
        dgrad = df2 - df1
        h -= np.outer(h @ dx, dx @ h) / (dx @ h @ dx) + np.outer(dgrad, dgrad) / (dgrad @ dx)
        df1, x1, f1 = df2, x2, f2
        itn += 1
        crit[4] = itn
    crit[0] = 2
    return x1, crit

class ChebyshevBasis:
    def __init__(self, order, node_number, domain=None):
        self.order = [order] if isinstance(order, int) else order
        self.node_number = [node_number] if isinstance(node_number, int) else node_number
        self.ndim = len(self.node_number)
        self.domain = domain if domain is not None else [[-1, 1] for _ in range(self.ndim)]
        if len(self.order) != self.ndim or len(self.node_number) != self.ndim or len(self.domain) != self.ndim:
            raise ValueError("Order, node_number, and domain must have the same length.")
        for i, (a, b) in enumerate(self.domain):
            if a >= b:
                raise ValueError(f"Domain bounds for dimension {i} must satisfy a < b.")
        self.nodes = self._generate_nodes()

    def _chebyshev_poly(self, x, n):
        return chebyt(n)(x)

    def _generate_nodes(self):
        nodes = []
        for i in range(self.ndim):
            n = self.node_number[i]
            z = np.cos(np.pi * (2 * np.arange(1, n + 1) - 1) / (2 * n))
            a, b = self.domain[i]
            x = 0.5 * (b - a) * z + 0.5 * (a + b)
            nodes.append(x)
        return nodes

    def get_tensor_grid(self):
        return np.array(list(product(*self.nodes)))

    def evaluate_basis_1d(self, x, dim, max_order=None):
        max_order = self.order[dim] if max_order is None else max_order
        a, b = self.domain[dim]
        z = (2 * x - (a + b)) / (b - a)
        return np.array([self._chebyshev_poly(z, n) for n in range(max_order + 1)])

    def evaluate_basis(self, x, max_order=None):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        for i in range(self.ndim):
            a, b = self.domain[i]
            if np.any(x[:, i] < a) or np.any(x[:, i] > b):
                raise ValueError(f"Input values outside domain in dimension {i}")
        max_order = self.order if max_order is None else ([max_order] * self.ndim if isinstance(max_order, int) else max_order)
        basis = []
        for xi in x:
            basis_i = np.array([1.0], dtype=np.float64)
            for d in range(self.ndim):
                try:
                    basis_1d = self.evaluate_basis_1d(xi[d], d, max_order[d])
                    if not np.all(np.isfinite(basis_1d)):
                        scale = np.max(np.abs(basis_1d[np.isfinite(basis_1d)]))
                        if scale > 0:
                            basis_1d = np.where(np.isfinite(basis_1d), basis_1d/scale, 0)
                    if len(basis_1d) > 1:
                        q, r = np.linalg.qr(basis_1d.reshape(-1, 1))
                        basis_1d = q.flatten() * np.sign(r[0, 0])
                    basis_i_new = np.zeros(len(basis_i) * len(basis_1d))
                    for j, b1 in enumerate(basis_1d):
                        basis_i_new[j::len(basis_1d)] = basis_i * b1
                    basis_i = basis_i_new
                except Exception as e:
                    print(f"Warning: Basis evaluation failed for dimension {d}: {str(e)}")
                    return np.zeros((len(x), np.prod([o+1 for o in max_order])))
            basis.append(basis_i)
        result = np.array(basis)
        if not np.all(np.isfinite(result)):
            result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
        return result

    def compute_weights(self, values, dim=None):
        if dim is not None:
            nodes = self.nodes[dim]
            basis = self.evaluate_basis_1d(nodes, dim)
            return np.linalg.lstsq(basis.T, values, rcond=None)[0]
        else:
            grid = self.get_tensor_grid()
            basis = self.evaluate_basis(grid)
            return np.linalg.lstsq(basis, values, rcond=None)[0]

    def approximate(self, x, weights, dim=None):
        if dim is not None:
            basis = self.evaluate_basis_1d(x, dim)
            return np.dot(basis.T, weights)
        else:
            basis = self.evaluate_basis(x)
            return np.dot(basis, weights)

    def compute_diff_matrix(self, dim):
        n = self.node_number[dim]
        theta = np.pi * (2 * np.arange(1, n + 1) - 1) / (2 * n)
        z = np.cos(theta)
        D = np.zeros((n, n))
        c = np.ones(n)
        c[0] = c[-1] = 2
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = c[i] / c[j] * (-1)**(i + j) / (z[i] - z[j])
        for i in range(n):
            D[i, i] = -np.sum(D[i, :])
        a, b = self.domain[dim]
        D_scaled = D * (2 / (b - a))
        return D_scaled

    def apply_diff(self, u, dim):
        return self.compute_diff_matrix(dim) @ u

    def integrate_via_coeffs(self, weights, dim):
        max_order = len(weights) - 1
        integral = 0
        for k in range(0, max_order + 1, 2):
            if k == 0:
                integral += weights[k] * np.pi / 2
            else:
                integral += weights[k] * 2 / (1 - k**2)
        a, b = self.domain[dim]
        integral *= (b - a) / 2
        return integral

    def gc_integral(self, f, dims, n=None):
        if isinstance(dims, int):
            dim = dims
            n = n or self.node_number[dim]
            x = np.cos(np.pi * (2 * np.arange(1, n + 1) - 1) / (2 * n))
            a, b = self.domain[dim]
            z = (x + 1) * (b - a) * 0.5 + a
            sum_ = sum(f(z[i]) * np.sqrt(1 - x[i]**2) for i in range(n))
            return np.pi * (b - a) * sum_ / (2 * n)
        elif isinstance(dims, (tuple, list)) and len(dims) == 2:
            dim1, dim2 = dims
            n = n or (self.node_number[dim1], self.node_number[dim2])
            n1, n2 = n if isinstance(n, (tuple, list)) else (n, n)
            x1 = np.cos(np.pi * (2 * np.arange(1, n1 + 1) - 1) / (2 * n1))
            x2 = np.cos(np.pi * (2 * np.arange(1, n2 + 1) - 1) / (2 * n2))
            a1, b1 = self.domain[dim1]
            a2, b2 = self.domain[dim2]
            z1 = (x1 + 1) * (b1 - a1) * 0.5 + a1
            z2 = (x2 + 1) * (b2 - a2) * 0.5 + a2
            sum_ = 0
            for i in range(n1):
                for j in range(n2):
                    y = f(z1[i], z2[j])
                    if np.isnan(y):
                        return np.nan
                    sum_ += y * np.sqrt(1 - x1[i]**2) * np.sqrt(1 - x2[j]**2)
            sum_ *= np.pi * (b1 - a1) * np.pi * (b2 - a2) / (2 * n1 * 2 * n2)
            return sum_
        else:
            raise ValueError("dims must be an integer or tuple/list of two integers.")


class ProjectionSolver:
    def __init__(self, model, basis):
        self.model = model
        self.basis = basis
        self.grid = self.basis.get_tensor_grid()
        # Compute basis function indices for Galerkin
        self.basis_indices = list(product(*[range(o + 1) for o in self.basis.order]))
        self.n_coeffs = len(self.basis_indices)

    def policy_func(self, state, coeffs):
        val = self.basis.approximate(state, coeffs)
        return float(np.exp(val))

    def residual_vector(self, coeffs):
        residuals = np.zeros(len(self.grid))
        for i, state in enumerate(self.grid):
            residuals[i] = self.model.residual(coeffs, state, self.basis, self.policy_func)
        return residuals

    def least_squares_objective(self, coeffs):
        return 0.5 * np.mean(self.residual_vector(coeffs) ** 2)

    def galerkin_residual(self, coeffs):
        """Compute Galerkin residuals: ∫ R(x; θ) * φ_j(x) w(x) dx = 0."""
        residuals = np.zeros(self.n_coeffs)
        for j, idx in enumerate(self.basis_indices):
            print(f"[Galerkin] Computing for basis index {j} (multi-index {idx})...")
            def integrand(k, z):
                state = np.array([k, z])
                residual = self.model.residual(coeffs, state, self.basis, self.policy_func)
                if np.isnan(residual):
                    print(f"[Galerkin] NaN residual at state (k={k}, z={z}) for basis index {j}")
                    return np.nan
                # Compute basis function φ_j(k, z)
                basis_vec = self.basis.evaluate_basis(state)[0]
                basis_j = basis_vec[j]
                return residual * basis_j
            integral = self.basis.gc_integral(integrand, dims=(0, 1), n=(10, 10))
            if not np.isfinite(integral):
                print(f"[Galerkin] Integral is not finite for basis index {j} (multi-index {idx}): {integral}")
            else:
                print(f"[Galerkin] Integral for basis index {j} (multi-index {idx}): {integral:.4e}")
            residuals[j] = integral if np.isfinite(integral) else np.nan
        print(f"[Galerkin] Residual vector: {residuals}")
        return residuals

    def solve(self, method='newton', maxit=5000, tol=1e-4, verbose=False, use_global=True, use_qr=True):
        try:
            coeffs_init = self.model.initial_guess(self.basis)
            if not np.all(np.isfinite(coeffs_init)):
                raise ValueError("Initial guess contains invalid values")
            grid = self.basis.get_tensor_grid()
            basis_matrix = self.basis.evaluate_basis(grid)
            rcond = np.linalg.cond(basis_matrix)
            if rcond > 1e12:
                print(f"Warning: Poorly conditioned basis matrix (condition number = {rcond:.2e}")
            if method == 'newton':
                try:
                    coeffs, crit = fixv_mn1(coeffs_init, self.residual_vector, maxit, tol, use_global, use_qr, verbose)
                except Exception as e:
                    print("Newton failed, retrying with QR factorization...")
                    coeffs, crit = fixv_mn1(coeffs_init, self.residual_vector, maxit, tol, True, True, verbose)
                max_res = crit[1]
                if not np.isfinite(max_res) or max_res > 1e3:
                    raise ValueError(f"Solution diverged (max residual = {max_res:.2e})")
                if crit[0] != 0:
                    raise ValueError(f"Failed to converge (return code = {crit[1]})")
                print(f"Newton: Converged, Max residual: {max_res:.2e}")
                return coeffs
            elif method == 'quasi_newton':
                coeffs, crit = quasi_newton(coeffs_init, self.least_squares_objective, maxit, tol, ptol=1e-4, verbose=verbose)
                max_res = np.max(np.abs(self.residual_vector(coeffs)))
                print(f"Quasi-Newton: {'Converged' if crit[0] == 0 else 'Failed'}, Max residual: {max_res:.2e}, RC: {crit[0]}")
                return coeffs
            elif method == 'scipy_hybr':
                solution = root(self.residual_vector, coeffs_init, method='hybr', options={'xtol': tol, 'maxfev': maxit})
                max_res = np.max(np.abs(solution.fun))
                print(f"SciPy Hybr: {'Converged' if solution.success else 'Failed'}, Max residual: {max_res:.2e}")
                return solution.x
            elif method == 'galerkin':
                try:
                    print("Galerkin Newton failed. Retrying with SciPy hybr...")
                    solution = root(self.galerkin_residual, coeffs_init, method='hybr', options={'xtol': tol, 'maxfev': maxit})
                    max_res = np.max(np.abs(self.galerkin_residual(solution.x)))
                    print(f"Galerkin SciPy Hybr: {'Converged' if solution.success else 'Failed'}, Max residual: {max_res:.2e}")
                    return solution.x
                except Exception as e:
                  coeffs, crit = fixv_mn1(coeffs_init, self.galerkin_residual, maxit, tol, use_global, use_qr, verbose)
                max_res = crit[1]
                if not np.isfinite(max_res) or max_res > 1e3:
                    raise ValueError(f"Galerkin solution diverged (max residual = {max_res:.2e})")
                if crit[0] != 0:
                    print(f"Galerkin failed to converge (return code = {crit[0]})")
                print(f"Galerkin: {'Converged' if crit[0] == 0 else 'Failed'}, Max residual: {max_res:.2e}")
                # Validate solution
                test_points = self.basis.get_tensor_grid()
                for point in test_points:
                    c = self.policy_func(point, coeffs)
                    if c <= 0 or not np.isfinite(c):
                        raise ValueError("Galerkin solution produces invalid consumption values")
                return coeffs
            else:
                raise ValueError("Method must be 'newton', 'quasi_newton', 'scipy_hybr', or 'galerkin'.")
        except Exception as e:
            print(f"Solver error: {e}")
            raise

    def state_space_matrix(self, coeffs):
        n_k = self.basis.node_number[0]
        n_z = self.basis.node_number[1] if self.model.use_continuous else len(self.model.z_values)
        c_matrix = np.zeros((n_k, n_z))
        y_matrix = np.zeros((n_k, n_z))
        k_next_matrix = np.zeros((n_k, n_z))
        k_nodes = self.basis.nodes[0]
        z_nodes = self.basis.nodes[1] if self.model.use_continuous else np.arange(len(self.model.z_values))
        for i, k in enumerate(k_nodes):
            for j, z_idx in enumerate(z_nodes):
                z = z_idx if self.model.use_continuous else self.model.z_values[int(z_idx)]
                state = np.array([k, z])
                c = self.policy_func(state, coeffs)
                y = self.model.production(k, z)
                k_next = self.model.capital_prime(k, z, c)
                c_matrix[i, j] = c
                y_matrix[i, j] = y
                k_next_matrix[i, j] = k_next
        return {'consumption': c_matrix, 'output': y_matrix, 'next_capital': k_next_matrix}

