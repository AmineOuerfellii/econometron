import numpy as np
import matplotlib.pyplot as plt
from .basis import ChebyshevBasis
from econometron.utils.solver import nr_solve
from econometron.utils.optimizers import minimize_qn

class ProjectionSolver:
    def __init__(self, order_vector, lower_bounds, upper_bounds):
        """Initialize projection solver with Chebyshev basis.
        
        Parameters:
        -----------
        order_vector : array-like
            Order of Chebyshev polynomials for each dimension
        lower_bounds : array-like
            Lower bounds of the domain
        upper_bounds : array-like
            Upper bounds of the domain
        """
        self.cheb_basis = ChebyshevBasis(order_vector, lower_bounds, upper_bounds)
        self.optimizer = minimize_qn
        self.n_dims = len(order_vector)
        self.basis_size = np.prod(np.array(order_vector) + 1)
    
    def solve_collocation(self, residual_func, initial_guess=None, maxit=5000, stopc=1e-8, verbose=False):
        """Solve functional equation using collocation method.
        
        Parameters:
        -----------
        residual_func : callable
            Residual function R(x, f(x), c) returning residuals
        initial_guess : array-like, optional
            Initial guess for coefficients (default: zeros)
        maxit : int
            Maximum iterations for Newton-Raphson
        stopc : float
            Convergence criterion
        verbose : bool
            Print iteration details
            
        Returns:
        --------
        coeffs : ndarray
            Chebyshev coefficients
        crit : ndarray
            Convergence criteria
        """
        if initial_guess is None:
            initial_guess = np.zeros(self.basis_size)
        
        nodes = self.cheb_basis.funnode()
        grid = self.cheb_basis.gridmake(nodes)
        
        def residual(coeffs):
            f_values = self.cheb_basis.funeval(coeffs, grid)
            return residual_func(grid, f_values, coeffs)
        
        coeffs, crit = nr_solve(
            initial_guess, residual, maxit=maxit, stopc=stopc, verbose=verbose
        )
        return coeffs, crit
    
    def solve_galerkin(self, residual_func, initial_guess=None, maxit=5000, stopc=1e-8, verbose=False):
        """Solve functional equation using Galerkin method.
        
        Parameters:
        -----------
        residual_func : callable
            Residual function R(x, f(x), c) returning residuals
        initial_guess : array-like, optional
            Initial guess for coefficients (default: zeros)
        maxit : int
            Maximum iterations for Newton-Raphson
        stopc : float
            Convergence criterion
        verbose : bool
            Print iteration details
            
        Returns:
        --------
        coeffs : ndarray
            Chebyshev coefficients
        crit : ndarray
            Convergence criteria
        """
        if initial_guess is None:
            initial_guess = np.zeros(self.basis_size)
        
        # Use more nodes for quadrature than basis order for accuracy
        quad_nodes = [n + 5 for n in self.cheb_basis.order_vector]
        nodes, weights = self.cheb_basis.gauss_chebyshev_quadrature(quad_nodes)
        grid = self.cheb_basis.gridmake(nodes)
        basis = self.cheb_basis.funbas(grid)
        
        def residual(coeffs):
            f_values = self.cheb_basis.funeval(coeffs, grid)
            residuals = residual_func(grid, f_values, coeffs)
            # Compute weighted residuals: ∫ R(x, f(x), c) * φ_i(x) dx
            weighted_residuals = np.zeros(self.basis_size)
            for i in range(self.basis_size):
                total_weight = 1.0
                for w in weights:
                    total_weight = np.kron(total_weight, w)
                weighted_residuals[i] = np.sum(total_weight * residuals * basis[:, i])
            return weighted_residuals
        
        coeffs, crit =nr_solve(
            initial_guess, residual, maxit=maxit, stopc=stopc, verbose=verbose
        )
        return coeffs, crit
    
    def solve_least_squares(self, residual_func, initial_guess=None, maxit=500, gtol=None, ptol=1e-7, verbose=False):
        """Solve functional equation using least squares method.
        
        Parameters:
        -----------
        residual_func : callable
            Residual function R(x, f(x), c) returning residuals
        initial_guess : array-like, optional
            Initial guess for coefficients (default: zeros)
        maxit : int
            Maximum iterations for Quasi-Newton
        gtol : float, optional
            Gradient tolerance
        ptol : float
            Parameter tolerance
        verbose : bool
            Print iteration details
            
        Returns:
        --------
        coeffs : ndarray
            Chebyshev coefficients
        crit : ndarray
            Convergence criteria
        """
        if initial_guess is None:
            initial_guess = np.zeros(self.basis_size)
        
        # Use more nodes than basis order for overdetermined system
        grid_nodes = [n + 5 for n in self.cheb_basis.order_vector]
        nodes = self.cheb_basis.funnode()
        grid = self.cheb_basis.gridmake(nodes)
        
        def objective(coeffs):
            f_values = self.cheb_basis.funeval(coeffs, grid)
            residuals = residual_func(grid, f_values, coeffs)
            return 0.5 * np.sum(residuals ** 2)
        
        coeffs, crit = self.optimizer(
            initial_guess, objective, maxit=maxit, gtol=gtol, ptol=ptol, verbose=verbose
        )
        return coeffs, crit
    
    def evaluate_solution(self, coeffs, points):
        """Evaluate the solution at given points.
        
        Parameters:
        -----------
        coeffs : array-like
            Chebyshev coefficients
        points : ndarray
            Points where to evaluate the solution
            
        Returns:
        --------
        values : ndarray
            Solution values at the points
        """
        return self.cheb_basis.funeval(coeffs, points)