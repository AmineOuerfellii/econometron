import numpy as np

class Root:
    
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
    def compute_gradient_norm(self, grad, x, fx):
        """Compute relative gradient norm.
        
        Args:
            grad: Gradient vector
            x: Current point
            fx: Function value
            
        Returns:
            Maximum relative gradient norm
        """
        crit=np.abs(grad)*np.maximum(np.abs(x),1.0)/np.maximum(np.abs(fx),1.0)
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
    
    
    
    
    