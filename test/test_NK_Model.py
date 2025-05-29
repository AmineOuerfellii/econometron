import numpy as np
import scipy as sp
import matplotlib as plt
import econometron

from econometron.Models import Model



def test_NK_Model():
        # Model setup
        equations = [
            "x_t - x_tp1 + (1/g) * (r_t - p_tp1) = 0",
            "p_t - beta * p_tp1 - kappa * (x_t - xbar_t) = 0",
            "- r_t + phi*p_t=0",
            "- xbar_tp1 + rho * xbar_t + sigmax = 0"
        ]

        variables = ['x', 'p', 'r','xbar']
        states = ['xbar']
        exo_states=['xbar']
        shock_names = ['sigmax']
        # Parameters dictionary
        parameters = {
            'g': 5,      # Inverse of relative risk aversion (1/g)
            'beta': 0.99,       # Discount factor
            'kappa': 0,
            'rho': 0.95,        # Persistence of output gap target
            'phi': 1.5,         # Taylor rule inflation coefficient
            'd': 0.5,          # Calvo parameter
            'sigmax':0.01
        }

        # Analytical steady state for initial guess
        sigma_X, beta,g, rho, phi, d = parameters['sigmax'],parameters['beta'],parameters['g'],parameters['rho'],parameters['phi'],parameters['d']
        parameters['kappa'] = ((1 - d) * (1 - d * beta)) / d
        initial_guess = [0, 0, 0 ,0]

        # Initialize and run model
        NK = Model(
            equations=equations,
            variables=variables,
            states=states,
            shock_names=shock_names,
            parameters=parameters,
            n_states=1,
            exo_states=exo_states,

        )

        NK.set_initial_guess(initial_guess)
        NK.compute_ss(guess=initial_guess, method='fsolve', options={'xtol': 1e-10})
        print(NK.ss)
        NK.approximate()
        A,B=NK.analytical_jacobians()
        NK.solve_model(A,B,NK.n_states)
        print("Policy Function (f):\n", NK.f)
        print("State Transition (p):\n", NK.p)
        NK.compute_irfs(T=41, t0=1, shocks={'sigmax': 0.01})
        print(NK.irfs)
        NK.simulate(T=51, drop_first=10, covariance_matrix=np.array([[0.01**2]]))
        NK.simulated.plot()
        NK.plot_irfs()

        # Print results
        print("Steady State:", NK.ss.to_dict())
        print("Policy Function (f):\n", NK.f)
        print("State Transition (p):\n", NK.p)
        print("Simulated Data (first 10 periods):\n", NK.simulated.head(10))


if __name__ == "__main__":

    test_NK_Model()
    print("Model test passed")