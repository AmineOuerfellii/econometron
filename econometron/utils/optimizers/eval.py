def evaluate_func(function, params):
    return function(params) if callable(function) else float('inf')