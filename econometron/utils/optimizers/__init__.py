from numpy.random import uniform, rand, seed
from colorama import Fore, Style  # Assuming these are imported for colored output
import numpy as np
from scipy.stats import norm
import numpy.random as npr
from econometron.utils.solver import Root
####################################################### Evaluation Function ##################################################
__all__ = ['evaluate_func','genetic_algorithm','simulated_annealing','rwm','minimize_qn']


def evaluate_func(function, params):
    print('current evaluated vector of parameters: \n', params)
    return function(params) if callable(function) else float('inf')

################################################## Simulated Annealing #####################################################
def simulated_annealing(function,x, lower_bounds, upper_bounds,T, cooling_rate, num_temperatures, num_steps, seed_value, max_evals, eps=1e-2):
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
    LB=lower_bounds
    UB=upper_bounds
    npr.seed(seed_value)
    sa_neps=4                                
    sa_eps=1e-2                     
    sa_maxeval=max_evals if max_evals is not None else 10**4                  
    sa_nargs=len(LB)
    sa_nobds=0
    sa_nacc=0                               
    sa_nevals=0                              
    fstar=[np.inf]*sa_neps
    f=evaluate_func(function,x)
    print('initial loss function value:',f)
    sa_nevals=sa_nevals+1
    xopt=x
    fopt=f
    xtot=x
    fstar[0]=f
    sa_opteval=0
    VM=[a_i - b_i for a_i, b_i in zip(UB, LB)]
    cont=True
    while cont:
        nup=0
        nrej=0
        nnew=0
        ndown=0
        lnobds=0
        nacp=[0]*sa_nargs
        for m in range(num_temperatures):
            for j in range(num_steps):
                for h in range(sa_nargs):
                    if sa_nevals>=sa_maxeval:
                        print('too many function evaluations')
                        cont=False
                    xp=x.copy()
                    
                    xp[h]=x[h]+VM[h]*(npr.uniform(-1,1))
                    if (xp[h]<LB[h]) or (xp[h]>UB[h]):
                        xp[h]=LB[h]+(UB[h]-LB[h])*npr.rand()
                        lnobds=lnobds+1
                        sa_nobds=sa_nobds+1
                    fp=evaluate_func(function,xp)
                    sa_nevals=sa_nevals+1
                    if fp<=f:
                        x=xp
                        f=fp
                        sa_nacc=sa_nacc+1
                        nacp[h]=nacp[h]+1
                        nup=nup+1
                        if fp<fopt:
                            xopt=xp
                            fopt=fp
                            sa_opteval=sa_nevals
                            nnew=nnew+1
                    else:
                        p=np.exp((f-fp)/T)
                        pp=npr.rand()
                        if pp<p:
                            x=xp
                            f=fp
                            sa_nacc=sa_nacc+1
                            nacp[h]=nacp[h]+1
                            ndown=ndown+1
                        else:
                            nrej=nrej+1
            c=[2]*sa_nargs
            for i in range(sa_nargs):
                ratio=nacp[i]/num_steps
                if ratio>0.6:
                    VM[i]=VM[i] * (1+c[i]*(ratio-0.6)/0.4)
                elif ratio <0.4:
                    VM[i]=VM[i]/(1+c[i]*((0.4-ratio)/0.4))
                if VM[i]>(UB[i]-LB[i]):
                    VM[i]=UB[i]-LB[i]
            print('xop',xopt)
            print('fop',fopt)
            for i in range(sa_nargs):
                nacp[i] = 0
        fstar[0]=f
        stop= ((fstar[0]-fopt) <= sa_eps)
        if np.any([np.abs(el-f)>sa_eps for el in fstar]):         
            stop=False
        if stop:
            cont=False

            print('optimum function value',fopt)
        T=T*cooling_rate
        fstar[1:sa_neps]=fstar[0:sa_neps-1]
        x=xopt
        f=fopt
    return {'x': xopt , 'fun': -fopt, 'N_FUNC_EVALS': sa_nevals, 'message': 'Simulated Annealing terminated successfully.'}
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

def minimize_qn(x0, func, maxit=500, gtol=None, ptol=1e-7, verbose=False):
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
        r=Root()
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
