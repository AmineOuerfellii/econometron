from numpy.random import uniform, rand, seed
from colorama import Fore, Style  # Assuming these are imported for colored output
import numpy as np
from scipy.stats import norm
from .eval import evaluate_func

################################################## Simulated Annealing #####################################################
def simulated_annealing(function, x, lower_bounds, upper_bounds,T, cooling_rate, num_temperatures, num_steps, seed_value, max_evals, eps=1e-2):
    """
    Simulated annealing optimization algorithm.

    Parameters:
    -----------
    function : callable
        Objective function to minimize.
    x : list or ndarray
        Initial parameter vector.
    lower_bounds : list or ndarray
        Lower bounds for parameters.
    upper_bounds : list or ndarray
        Upper bounds for parameters.
    initial_temp : float
        Initial temperature.
    cooling_rate : float
        Temperature reduction factor (0 < cooling_rate < 1).
    num_temperatures : int
        Number of temperature iterations.
    num_steps : int
        Number of steps per temperature.
    seed_value : int
        Random seed for reproducibility.
    max_evals : int
        Maximum number of function evaluations.
    eps : float, optional
        Convergence threshold (default: 1e-2).

    Returns:
    --------
    dict
        - x: Optimal parameters.
        - fun: Objective function value at optimum.
        - N_FUNC_EVALS: Number of function evaluations.
        - message: Termination message.
    """
    np.random.seed(seed_value)
    lower_bounds = lower_bounds
    upper_bounds = upper_bounds
    N = len(x)
    N_EPS = 4
    epsilon = eps
    N_ACCEPTED = 0
    N_OUT_OF_BOUNDS = 0
    N_FUNC_EVALS = 0
    F_STAR = [np.inf] * N_EPS
    N_opt_value = 0
    # Input validation
    if T <= 0:
        return {'message': 'Temperature must be positive.'}
    if len(upper_bounds) != N or len(lower_bounds) != N:
        return {'message': 'Bounds length must match parameter vector length.'}
    if any(x[i] < lower_bounds[i] or x[i] > upper_bounds[i] for i in range(N)):
        return {'message': 'Initial parameters must be within bounds.'}

    # Initial function evaluation
    F = evaluate_func(function, x)
    print('Initial loss function value:', F)
    N_FUNC_EVALS += 1
    X_opt = x
    F_opt = F
    F_STAR[0] = F
    print(F_STAR)
    VM = [a_i - b_i for a_i, b_i in zip(upper_bounds, lower_bounds)]    
    continue_flag = True

    while continue_flag:
        N_UP = 0
        N_REJECTED = 0
        N_DOWN = 0
        N_OUT_OF_BOUNDS_LOCAL = 0
        N_NEW = 0
        N_ACCEPTED_PER_PARAM = [0] * N
        for m in range(num_temperatures):
            for j in range(num_steps):
                for h in range(N):
                    if N_FUNC_EVALS >= max_evals:
                        continue_flag = False
                        return {'x': X_opt, 'fun': F_opt, 'N_FUNC_EVALS': N_FUNC_EVALS,
                                'message': 'Maximum function evaluations reached.'}


                    x_proposal = x.copy()
                    x_proposal[h] = x[h] + VM[h] * (2 * np.random.rand(1, 1) - 1.0).item(0)
                    if (x_proposal[h] < lower_bounds[h]) or (x_proposal[h] > upper_bounds[h]):
                        x_proposal[h] = lower_bounds[h] + (upper_bounds[h] - lower_bounds[h]) * np.random.rand(1, 1).item(0)
                        N_OUT_OF_BOUNDS_LOCAL += 1
                        N_OUT_OF_BOUNDS += 1

                    F_proposal = evaluate_func(function, x_proposal)
                    N_FUNC_EVALS += 1

                    if F_proposal <= F:
                        x = x_proposal
                        F = F_proposal
                        N_ACCEPTED += 1
                        N_ACCEPTED_PER_PARAM[h] += 1
                        N_UP += 1
                        if F_proposal < F_opt:
                            X_opt = x_proposal
                            F_opt = F_proposal
                            N_opt_value=N_FUNC_EVALS
                            N_NEW += 1
                    else:
                        p = np.exp((F-F_proposal)/T)
                        pp = np.random.rand(1, 1).item(0)
                        if pp < p:
                            x = x_proposal
                            F = F_proposal
                            N_ACCEPTED += 1
                            N_ACCEPTED_PER_PARAM[h] += 1
                            N_DOWN += 1
                        else:
                            N_REJECTED += 1

            # Adjust step sizes
            c = [2] * N
            for i in range(N):
                ratio = N_ACCEPTED_PER_PARAM[i] / num_steps
                if ratio > 0.6:
                    VM[i] *= (1 + c[i]*(ratio-0.6)/0.4)
                elif ratio < 0.4:
                    VM[i] /= (1 + c[i] * (0.4 - ratio) / 0.4)
                if VM[i] > (upper_bounds[i] - lower_bounds[i]):
                    VM[i] = upper_bounds[i] - lower_bounds[i]

            # Print iteration results
            print('*****************************Results***********************************', end='\n * ')
            print(f"Total number of steps is:{Fore.GREEN}{N_UP + N_DOWN + N_REJECTED}{Style.RESET_ALL}", end='\n * ')
            print(f"Current number of evaluations is:{Fore.GREEN}{N_FUNC_EVALS}{Style.RESET_ALL}", end='\n *')
            print(f"Current temperature is: {Fore.GREEN}{T}{Style.RESET_ALL}", end='\n *')
            print(f"Number of downward steps is :{Fore.GREEN}{N_DOWN}{Style.RESET_ALL}", end='\n *')
            print(f"Number of upward accepted steps is :{Fore.GREEN}{N_UP}{Style.RESET_ALL}", end='\n *')
            print(f"Number of downward rejected steps is:{Fore.GREEN}{N_REJECTED}{Style.RESET_ALL}", end='\n *')
            print(f"Number of times X exceeds the interval is :{Fore.GREEN}{N_OUT_OF_BOUNDS_LOCAL}{Style.RESET_ALL}", end='\n*')
            print(f"New maximum temperature is :{Fore.GREEN}{N_NEW}{Style.RESET_ALL} ", end='\n*')
            print(f"The current optimal parameter vector is :{Fore.RED}{X_opt}{Style.RESET_ALL}", end='\n*')
            print(f"The value of the current optimal point is:{Fore.RED}{F_opt}{Style.RESET_ALL}")
            print('**************************************************************************')
            for i in range(N):
              N_ACCEPTED_PER_PARAM[i]=0
        F_STAR[0] = F
        stop = (F_STAR[0] - F_opt) <= epsilon 
        if np.any([np.abs(el-F)>eps for el in F_STAR]):
          stop=False

        if stop:
            continue_flag = False
            print('*****************************Final Results***********************************', end='\n * ')
            print(f"Total number of steps is:{Fore.BLUE}{N_UP + N_DOWN + N_REJECTED}{Style.RESET_ALL} ", end='\n * ')
            print(f"Number of evaluations is:{Fore.BLUE}{N_FUNC_EVALS}{Style.RESET_ALL} ", end='\n * ')
            print(f"The optimal parameter vector is :{Fore.RED}{X_opt}{Style.RESET_ALL}", end='\n*')
            print(f"The value of the optimal point is:{Fore.RED}{F_opt}{Style.RESET_ALL}", end='\n*')
            print(f"Temperature is: {Fore.BLUE}{T}{Style.RESET_ALL}")
            print('**********************************************************************************')

        T =T*cooling_rate
        F_STAR[1:N_EPS] = F_STAR[0:N_EPS-1]
        x = X_opt
        F = F_opt

    return {'x': x , 'fun': F, 'N_FUNC_EVALS': N_FUNC_EVALS, 'message': 'Simulated Annealing terminated successfully.'}

################################################## Genetic Algorithm #####################################################

def genetic_algorithm(
    func,
    x0,
    lb,
    ub,
    pop_size=50,
    n_gen=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elite_frac=0.1,
    seed=1,
    verbose=True
):
    """
    Genetic Algorithm for parameter optimization.

    Returns:
    --------
    dict
        Dictionary with optimized parameters, objective value, nfev, and message
    """
    try:
        np.random.seed(seed)
        x0 = np.array(x0, dtype=float)
        lb = np.maximum(np.array(lb, dtype=float), 1e-6)
        ub = np.array(ub, dtype=float)
        N = len(x0)

        # Validate inputs
        if len(lb) != N or len(ub) != N:
            result = {'x': None, 'fun': None, 'nfev': 0, 'message': "Bounds length does not match parameter vector length."}
            raise ValueError(f"Returning early due to invalid bounds: {result}")
            return result
        if np.any(x0 < lb) or np.any(x0 > ub):
            result = {'x': None, 'fun': None, 'nfev': 0, 'message': "Initial guess outside bounds."}
            raise ValueError(f"Returning early due to invalid x0: {result}")
            return result

        # Initialize population
        population = np.random.uniform(lb, ub, (pop_size, N))
        population[0] = x0
        fitness = np.array([evaluate_func(func, ind) for ind in population])
        nfev = pop_size
        n_elite = max(1, int(elite_frac * pop_size))
        x_opt = population[np.argmin(fitness)]
        f_opt = np.min(fitness)

        # GA loop
        for gen in range(n_gen):
            parents = np.zeros((pop_size - n_elite, N))
            for i in range(pop_size - n_elite):
                tournament = np.random.choice(pop_size, 3)
                parents[i] = population[tournament[np.argmin(fitness[tournament])]]
            offspring = parents.copy()
            for i in range(0, pop_size - n_elite, 2):
                if i + 1 < pop_size - n_elite and np.random.rand() < crossover_rate:
                    alpha = np.random.uniform(-0.5, 1.5, N)
                    offspring[i] = parents[i] + alpha * (parents[i + 1] - parents[i])
                    offspring[i + 1] = parents[i + 1] + alpha * (parents[i] - parents[i + 1])
                    offspring[i] = np.clip(offspring[i], lb, ub)
                    offspring[i + 1] = np.clip(offspring[i + 1], lb, ub)
            for i in range(pop_size - n_elite):
                if np.random.rand() < mutation_rate:
                    mutation_scale = np.where(offspring[i] < 1e-6, 0.2 * (ub - lb), 0.1 * (ub - lb))
                    offspring[i] += np.random.normal(0, mutation_scale, N)
                    offspring[i] = np.clip(offspring[i], lb, ub)
            offspring_fitness = np.array([evaluate_func(func, ind) for ind in offspring])
            nfev += pop_size - n_elite
            population = np.vstack([population[np.argsort(fitness)[:n_elite]], offspring])
            fitness = np.concatenate([fitness[np.argsort(fitness)[:n_elite]], offspring_fitness])
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < f_opt:
                x_opt = population[best_idx]
                f_opt = fitness[best_idx]
            if verbose and gen % 10 == 0:
                near_zero = np.sum(np.any(population < 1e-6, axis=1))
                print(f"Generation {gen}: Best fitness = {f_opt}, Mean fitness = {np.mean(fitness)}, Near-zero params = {near_zero}")
                print({
                    'x': x_opt,
                    'fun': f_opt,
                    'nfev': nfev,
                    'message': 'Genetic Algorithm iteration.'
                })

        # Prepare result
        result = {
            'x': x_opt,
            'fun': float(f_opt),  # Convert np.float64 to float
            'nfev': nfev,
            'message': 'Genetic Algorithm completed successfully.'
        }

        if verbose:
            print(f"Final GA result: {result}")

        return result

    except Exception as e:
        error_result = {
            'x': None,
            'fun': None,
            'nfev': None,
            'message': f'GA failed: {str(e)}'
        }
        print(f"Error in genetic_algorithm: {e}, returning: {error_result}")
        return error_result
    
####################################################Quasi_newton############################################################################
import econometron.utils.solver.root as root

r=root()

def minimize_qn(self, x0, func, maxit=500, gtol=None, ptol=1e-7, verbose=False):
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
        grad = r.compute_jacobian(lambda x: np.array([func(x)]), x, 1).flatten()
        crit[1] = r.compute_gradient_norm(grad, x, f_val)
        
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
            step2, rc = r.quasi_newton_line_search(x, dx, f_val, grad, func)
            dx = step2 * dx
            x_new = x + dx
            f_new = func(x_new)
            crit[3] = f_new
            grad_new = r.compute_jacobian(lambda x: np.array([func(x)]), x_new, 1).flatten()
            crit[1] = r.compute_gradient_norm(grad_new, x_new, f_new)
            crit[2] = r.compute_param_change(x_new, dx)
            
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
################################################### Bayesian Optimization #####################################################

#################################################### Random Walk Metropolis ###################################################


def compute_proposal_sigma(n_params, lb, ub, base_std=0.1):
    """
    Compute proposal standard deviations based on parameter bounds.

    Parameters:
    -----------
    n_params : int
        Number of parameters.
    lb : ndarray
        Lower bounds for parameters.
    ub : ndarray
        Upper bounds for parameters.
    base_std : float or ndarray, optional
        Base standard deviation (scalar or array of length n_params, default: 0.1).

    Returns:
    --------
    ndarray
        Proposal standard deviations (shape: n_params).
    """
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)
    if len(lb) != n_params or len(ub) != n_params:
        raise ValueError("lb and ub must have length n_params")
    ranges = ub - lb
    ranges = np.where(ranges > 0, ranges, 1.0)  # Avoid zero ranges
    if np.isscalar(base_std):
        sigma = base_std * ranges / 10  # Scale to 10% of range
    else:
        base_std = np.array(base_std, dtype=float)
        if len(base_std) != n_params:
            raise ValueError("base_std must have length n_params")
        sigma = base_std * ranges / 10
    return sigma

# def rwm(objecti_func, prior, x0, lb, ub, n_iter=10000, burn_in=1000, thin=1, sigma=0.1, seed=42, verbose=True):
#     """
#     Random Walk Metropolis algorithm for posterior sampling.

#     Parameters:
#     -----------
#     objecti_func : callable
#         Function to compute log-likelihood, signature: likelihood(params).
#     prior : callable
#         Function to compute log-prior, signature: prior(params).
#     x0 : ndarray
#         Initial parameter vector.
#     lb : ndarray
#         Lower bounds for parameters.
#     ub : ndarray
#         Upper bounds for parameters.
#     n_iter : int, optional
#         Number of MCMC iterations (default: 10000).
#     burn_in : int, optional
#         Number of burn-in iterations (default: 1000).
#     thin : int, optional
#         Thinning factor (default: 1).
#     sigma : float or ndarray, optional
#         Proposal standard deviation (scalar or per-parameter, default: 0.1).
#     seed : int, optional
#         Random seed (default: 42).
#     verbose : bool, optional
#         Print summary statistics if True (default: True).

#     Returns:
#     --------
#     dict
#         - samples: Array of posterior samples (n_iter // thin x N).
#         - log_posterior: Array of log-posterior values (n_iter // thin).
#         - acceptance_rate: Fraction of accepted proposals.
#         - message: Status message.
#     """
#     try:
#       np.random.seed(seed)
#       x = np.array(x0, dtype=float)
#       lb = np.array(lb, dtype=float)
#       ub = np.array(ub, dtype=float)
#       sigma = np.array([sigma] * len(x) if np.isscalar(sigma) else sigma, dtype=float)
#       N = len(x)

#       # Input validation
#       if len(lb) != N or len(ub) != N or len(sigma) != N:
#           return {'samples': None, 'log_posterior': None, 'acceptance_rate': 0, 'message': "Invalid input dimensions."}
#       if np.any(x < lb) or np.any(x > ub):
#           return {'samples': None, 'log_posterior': None, 'acceptance_rate': 0, 'message': "Initial guess outside bounds."}

#       # Initialize arrays
#       samples = np.zeros((n_iter // thin, N))
#       log_posterior = np.zeros(n_iter // thin)
#       n_accept = 0
#       x_current = x.copy()
#       log_post_current = evaluate_func(objecti_func,x_current) + prior(x_current)
#       print(evaluate_func(objecti_func,x_current), prior(x_current))
#       # Check initial log-posterior
#       if np.isinf(log_post_current) or np.isnan(log_post_current):
#           res= {'samples': None, 'log_posterior': None, 'acceptance_rate': 0, 'message': "Invalid initial log-posterior."}

#       # Main loop
#       for i in range(n_iter):
#           # Propose new parameters
#           x_proposed = x_current + np.random.normal(0, sigma, N)
#           if np.any(x_proposed < lb) or np.any(x_proposed > ub):
#               continue  # Reject if outside bounds
#           log_prior_proposed = prior(x_proposed)
#           if log_prior_proposed == -np.inf:
#               R = 0  # Reject if prior is zero
#           else:
#               log_like_proposed = evaluate_func(objecti_func,x_proposed)
#               if log_like_proposed == -np.inf:
#                   R = 0  # Reject if likelihood is zero
#               else:
#                   log_post_proposed = log_like_proposed + log_prior_proposed
#                   R = np.exp(min(0, log_post_proposed - log_post_current))  # Acceptance ratio
#           # Accept/reject step
#           if np.random.rand() <= R:
#               x_current = x_proposed
#               log_post_current = log_post_proposed
#               n_accept += 1
#           # Store samples after burn-in
#           if i >= burn_in and i % thin == 0:
#               samples[i // thin] = x_current
#               log_posterior[i // thin] = log_post_current

#       # Compute acceptance rate
#       acceptance_rate = n_accept / n_iter



#       res= {'samples': samples,'log_posterior': log_posterior,'acceptance_rate': acceptance_rate,'mean Posterior parameters':np.mean(samples, axis=0),'Std posterior parameters':np.std(samples, axis=0),
#           'message': 'RWM completed successfully.'}
#             # Print summary
#       if verbose:
#           print("RWM Summary:")
#           print(f"Acceptance rate: {acceptance_rate:.3f}")
#           print(f"Mean posterior parameters: {np.mean(samples, axis=0)}")
#           print(f"Std posterior parameters: {np.std(samples, axis=0)}")
#       return res

#     except Exception as e:
#         print(f"Error in rwm: {e}")
#         return None

def rwm(objecti_func, prior, x0, lb, ub, n_iter=10000, burn_in=1000, thin=1, sigma=0.1, seed=42, verbose=True):
    try:
        np.random.seed(seed)
        x = np.array(x0, dtype=float)
        lb = np.array(lb, dtype=float)
        ub = np.array(ub, dtype=float)
        sigma = np.array([sigma] * len(x) if np.isscalar(sigma) else sigma, dtype=float)
        N = len(x)

        # Input validation
        if len(lb) != N or len(ub) != N or len(sigma) != N:
            return {
                'samples': None, 'log_posterior': None, 'acceptance_rate': 0,
                'message': "Invalid input dimensions."
            }

        if np.any(x < lb) or np.any(x > ub):
            return {
                'samples': None, 'log_posterior': None, 'acceptance_rate': 0,
                'message': "Initial guess outside bounds."
            }

        # Initialize arrays
        samples = np.zeros((n_iter // thin, N))
        log_posterior = np.zeros(n_iter // thin)
        n_accept = 0
        x_current = x.copy()
        log_post_current = evaluate_func(objecti_func, x_current) + prior(x_current)

        if np.isinf(log_post_current) or np.isnan(log_post_current):
            return {
                'samples': None, 'log_posterior': None, 'acceptance_rate': 0,
                'message': "Invalid initial log-posterior."
            }

        # Main loop with explicit sample counter
        sample_idx = 0
        for i in range(n_iter):
            x_proposed = x_current + np.random.normal(0, sigma, N)

            if np.any(x_proposed < lb) or np.any(x_proposed > ub):
                continue  # Reject if outside bounds

            log_prior_proposed = prior(x_proposed)
            
            if log_prior_proposed > -np.inf:
                log_like_proposed = evaluate_func(objecti_func, x_proposed)

                if log_like_proposed == -np.inf:
                      R = 0
                else:
                  log_post_proposed = log_like_proposed + log_prior_proposed
              
                  R = np.exp(min(0, log_post_proposed - log_post_current))
            else:
              R=0
            # Accept/reject step
            if np.random.rand(1, 1).item(0) <= R:
                x_current = x_proposed
                log_post_current = log_post_proposed
                n_accept += 1

            # Store samples after burn-in
            if i >= burn_in and i % thin == 0:
                if sample_idx < len(samples):
                    samples[sample_idx] = x_current
                    log_posterior[sample_idx] = log_post_current
                    if verbose:
                        print(f"Stored sample {sample_idx}: {x_current}")
                    sample_idx += 1

        acceptance_rate = n_accept / n_iter

        res = {
            'samples': samples,
            'log_posterior': log_posterior,
            'acceptance_rate': acceptance_rate,
            'mean_posterior_parameters': np.mean(samples, axis=0),
            'std_posterior_parameters': np.std(samples, axis=0),
            'message': 'RWM completed successfully.'
        }

        if verbose:
            print("RWM Summary:")
            print(f"Acceptance rate: {acceptance_rate:.3f}")
            print(f"Mean posterior parameters: {res['mean_posterior_parameters']}")
            print(f"Std posterior parameters: {res['std_posterior_parameters']}")
            print(f"First 5 samples:\n{samples[:5]}")

        return res

    except Exception as e:
        print(f"Error in rwm: {e}")
        return None