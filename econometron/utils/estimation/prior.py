import numpy as np
from typing import List, Tuple, Dict, Callable

from scipy.stats import gamma, beta as beta_dist
import numpy as np
from typing import List, Tuple, Dict, Callable

def make_prior_function(
    param_names: List[str],
    priors: Dict[str, Tuple[Callable, Dict]],
    bounds: Dict[str, Tuple[float, float]],
    verbose: bool = False
):
    """
    Create a generalized log-prior function for a model.

    Parameters:
    -----------
    param_names : list of str
        Names of the parameters in the order they appear in the vector.
    priors : dict
        Mapping from parameter name to a tuple (distribution, parameters),
        e.g., 'beta': (beta_dist, {'a': 99, 'b': 1})
    bounds : dict
        Mapping from parameter name to (lower_bound, upper_bound)
    verbose : bool
        Whether to print debug output.

    Returns:
    --------
    Function that takes a parameter vector and returns the log-prior.
    """
    
    def prior(params: List[float]) -> float:
      if len(params) != len(param_names):
          if verbose:
              print("Error: Parameter vector length mismatch.")
          return -np.inf

      # First pass: check bounds
      for name, value in zip(param_names, params):
          lb, ub = bounds[name]
          if not (lb < value < ub):
              if verbose:
                  print(f"[Bound Error] {name} = {value:.4f} not in ({lb}, {ub})")
              return -8e+30

      # Second pass: compute log-prior
      log_priors = []
      for name, value in zip(param_names, params):
          dist, kwargs = priors[name]
          try:
              logp = dist.logpdf(value, **kwargs)
              if not np.isfinite(logp):
                  raise ValueError("Non-finite logpdf value")
              log_priors.append(logp)
              if verbose:
                  print(f"[Log Prior] {name}: logpdf({value:.4f}) = {logp:.3f}")
          except Exception as e:
              if verbose:
                  print(f"[PDF Error] {name}: {e}")
              return -1e4

      total_log_prior = sum(log_priors)
      if verbose:
          print(f"[Total Log Prior] = {total_log_prior:.3f} | Params = {params}")

      return total_log_prior if np.isfinite(total_log_prior) else -np.inf

    return prior
