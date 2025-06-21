import numpy as np
from itertools import product

class ChebyshevBasis:
    def __init__(self, order_vector, lower_bounds, upper_bounds):
        """Initialize Chebyshev functional space.
        
        Parameters:
        -----------
        order_vector : array-like
            Order of approximation for each dimension
        lower_bounds : array-like
            Lower bounds of the domain
        upper_bounds : array-like
            Upper bounds of the domain
        """
        self.order_vector = np.asarray(order_vector, dtype=int)
        self.lower_bounds = np.asarray(lower_bounds, dtype=float)
        self.upper_bounds = np.asarray(upper_bounds, dtype=float)
        
        if len(self.order_vector) != len(self.lower_bounds) or len(self.lower_bounds) != len(self.upper_bounds):
            raise ValueError("Dimensions of order_vector, lower_bounds, and upper_bounds must match")
        
        self.n_dims = len(self.order_vector)
        
    def funnode(self):
        """Generate Chebyshev nodes for collocation.
        
        Returns:
        --------
        nodes : list of arrays
            List of Chebyshev nodes for each dimension
        """
        nodes = []
        for i, n in enumerate(self.order_vector):
            # Compute Chebyshev nodes in [-1, 1]
            k = np.arange(n + 1)
            cheb_nodes = np.cos(np.pi * (2 * k + 1) / (2 * (n + 1)))
            # Transform to [a, b]
            a, b = self.lower_bounds[i], self.upper_bounds[i]
            scaled_nodes = (b - a) / 2 * cheb_nodes + (a + b) / 2
            nodes.append(scaled_nodes[::-1])  # Reverse to match standard ordering
        return nodes
    
    def gridmake(self, nodes):
        """Create complete grid from nodes.
        
        Parameters:
        -----------
        nodes : list of arrays
            List of nodes for each dimension
            
        Returns:
        --------
        grid : ndarray
            Complete grid of shape (prod(n_i + 1), n_dims)
        """
        grid = np.array(list(product(*nodes)))
        return grid
    
    def funbas(self, grid):
        """Compute basis functions evaluated at grid points.
        
        Parameters:
        -----------
        grid : ndarray
            Grid points of shape (n_points, n_dims)
            
        Returns:
        --------
        basis : ndarray
            Basis matrix of shape (n_points, prod(n_i + 1))
        """
        if grid.ndim == 1:
            grid = grid.reshape(-1, 1)
            
        n_points, n_dims = grid.shape
        if n_dims != self.n_dims:
            raise ValueError(f"Grid must have {self.n_dims} dimensions")
            
        # Transform grid points to [-1, 1]
        x = np.zeros_like(grid)
        for d in range(n_dims):
            a, b = self.lower_bounds[d], self.upper_bounds[d]
            x[:, d] = 2 * (grid[:, d] - a) / (b - a) - 1
            x[:, d] = np.clip(x[:, d], -1, 1)
            
        # Initialize basis matrix
        total_basis_size = np.prod(self.order_vector + 1)
        basis = np.ones((n_points, total_basis_size))
        
        # Compute multi-dimensional basis
        idx = 0
        for multi_index in product(*[range(n + 1) for n in self.order_vector]):
            for d in range(n_dims):
                basis[:, idx] *= self._chebyshev_poly(x[:, d], multi_index[d])
            idx += 1
            
        return basis
    
    def funeval(self, coefficients, evaluation_points):
        """Evaluate approximated function at given points.
        
        Parameters:
        -----------
        coefficients : array-like
            Coefficients of the approximation
        evaluation_points : ndarray
            Points where function is evaluated
            
        Returns:
        --------
        values : ndarray
            Function values at evaluation points
        """
        if evaluation_points.ndim == 1:
            evaluation_points = evaluation_points.reshape(-1, 1)
            
        basis = self.funbas(evaluation_points)
        coefficients = np.asarray(coefficients).flatten()
        if basis.shape[1] != len(coefficients):
            raise ValueError("Number of coefficients must match basis size")
            
        values = basis @ coefficients
        return values
    
    def gauss_chebyshev_quadrature(self, n_nodes):
        """Compute Gauss-Chebyshev quadrature nodes and weights for each dimension.
        
        Parameters:
        -----------
        n_nodes : int or array-like
            Number of quadrature nodes per dimension (single int or list for each dimension)
            
        Returns:
        --------
        nodes : list of arrays
            Quadrature nodes for each dimension
        weights : list of arrays
            Quadrature weights for each dimension
        """
        if isinstance(n_nodes, int):
            n_nodes = [n_nodes] * self.n_dims
        else:
            n_nodes = np.asarray(n_nodes, dtype=int)
            if len(n_nodes) != self.n_dims:
                raise ValueError(f"Number of nodes must match number of dimensions ({self.n_dims})")
        
        nodes = []
        weights = []
        for i, n in enumerate(n_nodes):
            # Gauss-Chebyshev nodes (zeros of Tn(x)) in [-1, 1]
            k = np.arange(n)
            cheb_nodes = np.cos(np.pi * (2 * k + 1) / (2 * n))
            # Corresponding weights
            cheb_weights = np.full(n, np.pi / n)
            # Transform nodes to [a, b]
            a, b = self.lower_bounds[i], self.upper_bounds[i]
            scaled_nodes = (b - a) / 2 * cheb_nodes + (a + b) / 2
            # Adjust weights for interval [a, b]
            scaled_weights = cheb_weights * (b - a) / 2
            nodes.append(scaled_nodes[::-1])  # Reverse for consistency
            weights.append(scaled_weights)
        
        return nodes, weights
    
    def integrate(self, func, n_nodes):
        """Perform Gauss-Chebyshev quadrature for a given function.
        
        Parameters:
        -----------
        func : callable
            Function to integrate, takes an array of shape (n_points, n_dims)
        n_nodes : int or array-like
            Number of quadrature nodes per dimension
            
        Returns:
        --------
        integral : float
            Approximate integral value
        """
        nodes, weights = self.gauss_chebyshev_quadrature(n_nodes)
        grid = self.gridmake(nodes)
        f_values = func(grid)
        
        # Compute multi-dimensional quadrature
        total_weight = 1.0
        for w in weights:
            total_weight = np.kron(total_weight, w)
        
        integral = np.sum(total_weight * f_values)
        return integral

    def gauss_hermite_nodes(self, n ,sigma):
        """Compute Gauss-Hermite quadrature nodes and weights for N(0, sigma^2).

        Parameters:
        -----------
        n : int
            Number of nodes
        sigma : float
            Standard deviation

        Returns:
        --------
        nodes : ndarray
            Quadrature nodes (for z')
        weights : ndarray
            Quadrature weights
        """
        nodes, weights = np.polynomial.hermite.hermgauss(n)
        # Scale nodes and weights for N(0, sigma^2)
        nodes = np.sqrt(2) * sigma * nodes
        weights = weights / np.sqrt(np.pi)
        return np.exp(nodes), weights  # Transform to log-normal


    def _chebyshev_poly(self, x, n):
        """Compute nth Chebyshev polynomial at x.
        
        Parameters:
        -----------
        x : array-like
            Points where polynomial is evaluated
        n : int
            Polynomial degree
            
        Returns:
        --------
        poly : ndarray
            Polynomial values
        """
        if n == 0:
            return np.ones_like(x)
        if n == 1:
            return x
            
        T_nm2 = np.ones_like(x)
        T_nm1 = x
        for k in range(2, n + 1):
            T_n = 2 * x * T_nm1 - T_nm2
            T_nm2 = T_nm1
            T_nm1 = T_n
        return T_n