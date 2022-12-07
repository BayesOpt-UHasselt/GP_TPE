import numpy as np
from Functions.AitchsonAitkenKernel import AitchisonAitkenKernel
from Functions.UniformKernel import UniformKernel
import sys
eps = sys.float_info.epsilon

class CategoricalParzenEstimator:
    # note: this implementation has not been verified yet
    def __init__(self, samples, n_choices, weights_func, top=0.9):
        self.n_choices = n_choices
        self.mus = samples
        self.basis = [AitchisonAitkenKernel(c, n_choices, top=top) for c in samples]
        self.basis.append(UniformKernel(n_choices))
        self.weights = weights_func(len(samples) + 1)
        self.weights /= self.weights.sum()

    def sample_from_density_estimator(self, rng, n_ei_candidates):
        basis_samples = rng.multinomial(n=1, pvals=self.weights, size=n_ei_candidates)
        basis_idxs = np.dot(basis_samples, np.arange(self.weights.size))
        return np.array([self.basis[idx].sample_from_kernel(rng) for idx in basis_idxs])

    def log_likelihood(self, values):
        ps = np.zeros(values.shape, dtype=float)
        for w, b in zip(self.weights, self.basis):
            ps += w * b.cdf_for_numpy(values)
        return np.log(ps + eps)

    def basis_loglikelihood(self, xs):
        return_vals = np.zeros((len(self.basis), xs.size), dtype=float)
        for basis_idx, b in enumerate(self.basis):
            return_vals[basis_idx] += b.log_cdf_for_numpy(xs)
        return return_vals

