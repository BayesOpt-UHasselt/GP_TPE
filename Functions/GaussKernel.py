import numpy as np
import sys
import scipy.special
eps = sys.float_info.epsilon

class GaussKernel:
    def __init__(self, mu, sigma, lb, ub, q):
        self.mu = mu
        self.sigma = max(sigma, eps)
        self.lb, self.ub, self.q = lb, ub, q
        self.norm_const = 1.  # do not delete! this line is needed
        self.norm_const = 1. / (self.cdf(ub) - self.cdf(lb))

    def pdf(self, x):
        if self.q is None:
            z = 2.50662827 * self.sigma  # np.sqrt(2 * np.pi) * self.sigma
            mahalanobis = ((x - self.mu) / self.sigma) ** 2
            return self.norm_const / z * np.exp(-0.5 * mahalanobis)
        else:
            integral_u = self.cdf(np.minimum(x + 0.5 * self.q, self.ub))
            integral_l = self.cdf(np.maximum(x + 0.5 * self.q, self.lb))
            return np.maximum(integral_u - integral_l, eps)

    def log_pdf(self, x):
        if self.q is None:
            z = 2.50662827 * self.sigma  # np.sqrt(2 * np.pi) * self.sigma
            mahalanobis = ((x - self.mu) / self.sigma) ** 2
            return np.log(self.norm_const / z) - 0.5 * mahalanobis
        else:
            integral_u = self.cdf(np.minimum(x + 0.5 * self.q, self.ub))
            integral_l = self.cdf(np.maximum(x + 0.5 * self.q, self.lb))
            return np.log(np.maximum(integral_u - integral_l, eps))

    def cdf(self, x):
        z = (x - self.mu) / (1.41421356 * self.sigma)  # (x - self.mu) / (np.sqrt(2) * self.sigma)
        return np.maximum(self.norm_const * 0.5 * (1. + scipy.special.erf(z)), eps)

    def sample_from_kernel(self, rng):
        while True:
            sample = rng.normal(loc=self.mu, scale=self.sigma)
            if self.lb <= sample <= self.ub:
                return sample
