import numpy as np

class AitchisonAitkenKernel:
    def __init__(self, choice, n_choices, top=0.9):
        self.n_choices = n_choices
        self.choice = choice
        self.top = top

    def cdf(self, x):
        if x == self.choice:
            return self.top
        elif 0 <= x <= self.n_choices - 1:
            return (1. - self.top) / (self.n_choices - 1)
        else:
            raise ValueError('The choice must be between {} and {}, but {} was given.'.format(
                0, self.n_choices - 1, x))

    def log_cdf(self, x):
        return np.log(self.cdf(x))

    def cdf_for_numpy(self, xs):
        return_val = np.array([])
        for x in xs:
            return_val = np.append(return_val, self.cdf(x))
        return return_val

    def log_cdf_for_numpy(self, xs):
        return_val = np.array([])
        for x in xs:
            return_val = np.append(return_val, self.log_cdf(x))
        return return_val

    def probabilities(self):
        return np.array([self.cdf(n) for n in range(self.n_choices)])

    def sample_from_kernel(self, rng):
        choice_one_hot = rng.multinomial(n=1, pvals=self.probabilities(), size=1)
        return np.dot(choice_one_hot, np.arange(self.n_choices))[0]
