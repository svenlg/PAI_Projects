import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process import GaussianProcessRegressor

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
CONST_OFFSET = 1.5
SEED = 0


""" Solution """
class BO_algo():
    def __init__(self):
        kernel = Matern(length_scale=0.5, nu=2.5,length_scale_bounds="fixed")
        f_kernel = 0.5*kernel
        v_kernel = np.sqrt(2)*kernel
        """Initializes the algorithm with a parameter configuration. """
        self.x_old = []
        self.f_old = []
        self.v_old = []
        self.f_sol = []
        self.x_sol = []

        self.step = 0
        self.beta = 0.5

        self.gp_f = GaussianProcessRegressor(kernel=f_kernel,alpha=0.15**2)
        self.gp_v = GaussianProcessRegressor(kernel=v_kernel,alpha=0.0001**2)


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """
        #self.step += 2
        recommendation = self.optimize_acquisition_function()
        return recommendation


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain, approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])


    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """
        x_data = x.reshape(-1,1)
        
        mean_v, std_v = self.gp_v.predict(x_data, return_std=True)
        mean_f, std_f = self.gp_f.predict(x_data, return_std=True)

        mean_v += CONST_OFFSET

        # Fall ++
        if mean_v > 1.2 and mean_v - std_v > 1.2:
            #return mean_f[0] + self.beta/self.step * std_f[0]
            return mean_f[0] + self.beta * std_f[0]

        # Fall 0
        if (mean_v > 1.2 and mean_v - std_v <= 1.2) or (mean_v < 1.2 and mean_v + std_v >=1.2):
            # return mean_f[0] + self.beta/self.step * std_f[0] - 100
            return mean_f[0] + self.beta * std_f[0] - 100

        # Fall --
        if mean_v < 1.2 and mean_v + std_v < 1.2:
            # return mean_f[0] + self.beta/self.step * std_f[0] - 1000
            return mean_f[0] + self.beta * std_f[0] - 1000


        return #mean_f[0]


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        x -= CONST_OFFSET

        self.x_old.append(x)
        self.f_old.append(f)
        self.v_old.append(v)

        x_data = np.array(self.x_old, dtype=object).reshape(-1,1)
        f_data = np.array(self.f_old, dtype=object).reshape(-1,1)
        v_data = np.array(self.v_old, dtype=object).reshape(-1,1) 
        self.gp_f.fit(x_data,f_data)
        self.gp_v.fit(x_data,v_data)

        if v > SAFETY_THRESHOLD:
            self.x_sol.append(x)
            self.f_sol.append(f)


    def get_solution(self):
        f_x_max = np.argmax(np.array(self.f_sol))
        x_star = self.x_sol[f_x_max]
        return x_star


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init


def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}')


if __name__ == "__main__":
    main()