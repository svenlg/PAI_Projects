import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from scipy.stats import norm

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        maternf = 0.5*Matern(length_scale=0.5,nu=2.5,length_scale_bounds="fixed")
        # constantv = ConstantKernel(constant_value=1.5, constant_value_bounds="fixed")
        maternv = np.sqrt(2)*Matern(length_scale=0.5,nu=2.5,length_scale_bounds="fixed")
        self.f = GaussianProcessRegressor(kernel=maternf,alpha=0.15)
        self.v = GaussianProcessRegressor(kernel=maternv,alpha=0.0001)
        self.x_t = []
        self.flist = []
        self.vlist = []


    def next_recommendation(self):
        """
        Recommend the next input to sample.
        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.

        return self.optimize_acquisition_function()


    '''def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.
        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """
        ind = []
        collection_max_f = []
        for i in range(3):
            def objective(x):
                return -self.acquisition_function(x)[i]
            f_values = []
            x_values = []
            # Restarts the optimization 20 times and pick best solution
            for _ in range(20):
                x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                    np.random.rand(domain.shape[0])
                result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                    approx_grad=True)
                x_values.append(np.clip(result[0], *domain[0]))
                f_values.append(-result[1])
            collection_max_f.append(np.amax(f_values))
            ind.append(np.argmax(f_values))
        final_ind = ind[np.argmax(collection_max_f)]
        return np.atleast_2d(x_values[final_ind])'''
    
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

        # Restarts the optimization 20 times and pick best solution; COULD INCREASE THIS FOR SWEATY BOI GAINS 
        for _ in range(30):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                approx_grad=True)
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
        # TODO: enter your code here
        mean_v, std_v = self.v.predict(x.reshape(-1,1), return_std=True)

        # if mean_v[0] + std_v[0] < SAFETY_THRESHOLD - 1.5:
        if mean_v[0] < SAFETY_THRESHOLD:
            return 0 #, 0, 0

        mean, std = self.f.predict(x.reshape(-1,1), return_std=True)

        x_best = self.x_t[np.argmax(self.flist)]
        f_best = self.f.predict(x_best.reshape(-1,1))
        gamma = (f_best[0]-mean[0])/std[0]
        # Expected Improvement Acquisition Function
        a_EI = std[0]*(gamma*norm.cdf(gamma) + norm.pdf(gamma))

        # Probability of Improvement Acquisition Function
        a_PI = norm.cdf(gamma)

        # Constraint Weighted Expected Improvement 
        random_var = mean_v[0] - SAFETY_THRESHOLD 
        a_CWEI = a_EI * (1 - norm.cdf(random_var,loc=mean_v[0],scale=std_v[0]))
        
        # Thompson Sampling (also an Acquisition Function Variant)
        thompson = self.f.predict(x.reshape(-1,1))

        # Upper Confidence Bound Acquisition Function
        beta = 10000
        ucb = mean[0] + beta*std[0]

        return ucb #ucb, a_EI, a_PI, thompson


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

        # TODO: enter your code here

        # if v >= SAFETY_THRESHOLD:
        self.x_t.append(x)
        self.flist.append(f)
        self.vlist.append(v)
        x_data = np.array(self.x_t, dtype="object").reshape(-1,1)
        f_data = np.array(self.flist, dtype="object").reshape(-1,1)
        v_data = np.array(self.vlist, dtype="object").reshape(-1,1)

        self.f.fit(x_data, f_data)
        self.v.fit(x_data, v_data)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.
        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        accept = (np.asarray(self.vlist) >= SAFETY_THRESHOLD)
        x_accept = np.asarray(self.x_t)[accept]
        f_accept = np.asarray(self.flist)[accept]

        # If no optimum is found which matchens the constraint try to use the next-best point!
        if len(f_accept) == 0:
            return self.optimize_acquisition_function()
        
        x_star = x_accept[np.argmax(f_accept)]

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
    # Testing
    if x > 2 and x < 3:
        return 1
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
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()

