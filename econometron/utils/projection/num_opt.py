import numpy as np

class NumericalOptimizer:
    def compute_jacobian(self, func, x, n_outputs, eps=None):
        """Compute Jacobian matrix using central difference method.
        
        Args:
            func: Function to differentiate
            x: Input point (numpy array)
            n_outputs: Number of function outputs
            eps: Step size (defaults to machine epsilon^(1/3))
            
        Returns:
            Jacobian matrix of shape (n_outputs, len(x))
        """
        if eps is None:
            eps = np.finfo(float).eps ** (1/3)
        
        n_inputs = len(x)
        jacobian = np.zeros((n_outputs, n_inputs))
        h0 = eps ** (1/3)
        x_plus = x.copy()
        x_minus = x.copy()
        
        for i in range(n_inputs):
            h = h0 * max(abs(x[i]), 1.0) * (-1 if x[i] < 0 else 1)
            x_plus[i] = x[i] + h
            x_minus[i] = x[i] - h
            h = x_plus[i] - x[i]  # Actual step size
            f_plus = func(x_plus)
            f_minus = func(x_minus)
            for j in range(n_outputs):
                jacobian[j, i] = (f_plus[j] - f_minus[j]) / (2 * h)
            x_plus[i] = x[i]
            x_minus[i] = x[i]
        
        return jacobian

    def newton_line_search(self, x0, dx0, grad, func, smult=1e-4, smin=0.1, smax=0.5, stol=1e-11):
        """Perform line search for Newton-Raphson method.
        
        Args:
            x0: Current point
            dx0: Search direction
            grad: Gradient at x0
            func: Objective function
            smult: Sufficient decrease parameter
            smin: Minimum step size
            smax: Maximum step size
            stol: Step tolerance
            
        Returns:
            Step size (or -1.0 if tolerance reached)
        """
        f0 = func(x0)
        g0 = 0.5 * np.sum(f0 ** 2)
        dgdx = grad @ dx0
        s1 = 1.0
        g1 = 0.5 * np.sum(func(x0 + dx0) ** 2)
        
        if g1 <= g0 + smult * dgdx:
            return s1
        
        s = -dgdx / (2 * (g1 - g0 - dgdx))
        s = min(max(s, smin), smax)
        x1 = x0 + s * dx0
        g2 = 0.5 * np.sum(func(x1) ** 2)
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
            g2 = 0.5 * np.sum(func(x1) ** 2)
        
        return s2

    def newton_raphson(self, x0, func, maxit=5000, stopc=1e-8, use_global=True, use_qr=False, verbose=False):
        """Solve nonlinear system using modified Newton-Raphson method."""
        x = x0.copy()
        crit = np.ones(5)
        crit[0] = 0
        itn = 0
        lam = 1e-6
        lam_max = 1e6
        lam_mult = 10.0
        min_step = 1e-6  # Increased from 1e-8
        
        while itn < maxit and crit[1] >= stopc:
            if verbose:
                print(f"[Newton] Iteration {itn}, coeffs[:5]: {x[:5]}")
            
            fx = func(x)
            df = self.compute_jacobian(func, x, len(fx))
            
            if np.any(np.isnan(df)):
                crit[0] = 1
                print("Jacobian contains NaN. Aborting.")
                return x, crit
            
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
            x_trial = x + step * dx
            f_trial = func(x_trial)
            obj_trial = 0.5 * np.sum(f_trial ** 2)
            
            while (not np.all(np.isfinite(f_trial)) or obj_trial > obj_val) and step > min_step:
                step *= 0.5
                x_trial = x + step * dx
                f_trial = func(x_trial)
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
            
            x = x + step * dx
            crit[1] = np.max(np.abs(func(x)))
            crit[2] = np.max(np.abs(step * dx) / np.maximum(np.abs(x), 1.0))
            crit[3] = 0.5 * np.sum(func(x) ** 2)
            crit[4] = itn
            itn += 1
        
        if itn >= maxit:
            crit[0] = 3
        
        return x, crit
    def compute_gradient_norm(self, grad, x, fx):
        """Compute relative gradient norm.
        
        Args:
            grad: Gradient vector
            x: Current point
            fx: Function value
            
        Returns:
            Maximum relative gradient norm
        """
        crit = np.abs(grad) * np.maximum(np.abs(x), 1.0) / np.maximum(np.abs(fx), 1.0)
        return np.max(crit)

    def compute_param_change(self, x, dx):
        """Compute relative parameter change.
        
        Args:
            x: Current point
            dx: Parameter change
            
        Returns:
            Maximum relative parameter change
        """
        return np.max(np.abs(dx) / np.maximum(np.abs(x), 1.0))

    def quasi_newton_line_search(self, x0, dx0, f0, grad, func, smult=1e-4, smin=0.1, smax=0.5, ptol=1e-12):
        """Perform line search for Quasi-Newton method.
        
        Args:
            x0: Current point
            dx0: Search direction
            f0: Function value at x0
            grad: Gradient at x0
            func: Objective function
            smult: Sufficient decrease parameter
            smin: Minimum step size
            smax: Maximum step size
            ptol: Parameter tolerance
            
        Returns:
            Tuple of (step size, return code)
        """
        dfdx = grad @ dx0
        s1 = 1.0
        f1 = func(x0 + dx0)
        
        if f1 <= f0 + smult * dfdx:
            return s1, 0
        
        s = -dfdx / (2 * (f1 - f0 - dfdx))
        s = min(max(s, smin), smax)
        x1 = x0 + s * dx0
        f2 = func(x1)
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
            f2 = func(x1)
        
        return s2, 0

    def quasi_newton(self, x0, func, maxit=500, gtol=None, ptol=1e-7, verbose=False):
        """Minimize scalar function using BFGS Quasi-Newton method.
        
        Args:
            x0: Initial guess
            func: Scalar objective function
            maxit: Maximum iterations
            gtol: Gradient tolerance
            ptol: Parameter tolerance
            verbose: Print iteration details
            
        Returns:
            x: Solution
            crit: Convergence criteria array
        """
        if gtol is None:
            gtol = np.finfo(float).eps ** (1/3)
        
        crit = np.zeros(5)
        hessian = np.eye(len(x0))
        x = x0.copy()
        f_val = func(x)
        grad = self.compute_jacobian(lambda x: np.array([func(x)]), x, 1).flatten()
        crit[1] = self.compute_gradient_norm(grad, x, f_val)
        
        if crit[1] < 1e-3 * gtol:
            crit[0], crit[3], crit[4] = 0, f_val, 0
            return x, crit
        
        itn = 1
        crit[2] = 1
        
        while itn < maxit:
            if verbose:
                print(f"Iteration {itn}: gTol = {crit[1]:.2e}, pTol = {crit[2]:.2e}, f(x) = {crit[3]:.2e}")
            
            dx = np.linalg.solve(hessian, -grad)
            step1 = 1.0
            
            while np.isnan(func(x + step1 * dx)):
                step1 /= 2
                if step1 < 1e-16:
                    crit[0] = 1
                    return x, crit
            
            dx = step1 * dx
            step2, rc = self.quasi_newton_line_search(x, dx, f_val, grad, func)
            dx = step2 * dx
            x_new = x + dx
            f_new = func(x_new)
            crit[3] = f_new
            grad_new = self.compute_jacobian(lambda x: np.array([func(x)]), x_new, 1).flatten()
            crit[1] = self.compute_gradient_norm(grad_new, x_new, f_new)
            crit[2] = self.compute_param_change(x_new, dx)
            
            if crit[1] > gtol and rc:
                crit[0] = 2
                return x_new, crit
            
            if crit[1] < gtol or crit[2] < ptol:
                crit[0] = 0
                return x_new, crit
            
            dgrad = grad_new - grad
            hessian -= np.outer(hessian @ dx, dx @ hessian) / (dx @ hessian @ dx) + np.outer(dgrad, dgrad) / (dgrad @ dx)
            
            grad, x, f_val = grad_new, x_new, f_new
            itn += 1
            crit[4] = itn
        
        crit[0] = 2
        return x, crit