import numpy as np
from Functions.GaussKernel import GaussKernel
import sys
eps = sys.float_info.epsilon

class NumericalParzenEstimator:
    def __init__(self, samples, lb, ub, weights_func, q=None, rule='james'):
        self.lb, self.ub, self.q, self.rule = lb, ub, q, rule
        self.weights, self.mus, self.sigmas = self._calculate(samples, weights_func)
        self.basis = [GaussKernel(m, s, lb, ub, q) for m, s in zip(self.mus, self.sigmas)]

    def sample_from_density_estimator(self, rng, n_ei_candidates):
        samples = np.asarray([], dtype=float)
        while samples.size < n_ei_candidates:
            active = np.argmax(rng.multinomial(1, self.weights))
            drawn_hp = self.basis[active].sample_from_kernel(rng)
            samples = np.append(samples, drawn_hp)

        return samples if self.q is None else np.round(samples / self.q) * self.q

    def log_likelihood(self, xs):
        ps = np.zeros(xs.shape, dtype=float)
        for w, b in zip(self.weights, self.basis):
            ps += w * b.pdf(xs)

        return np.log(ps + eps)

    def basis_loglikelihood(self, xs):
        return_vals = np.zeros((len(self.basis), xs.size), dtype=float)
        for basis_idx, b in enumerate(self.basis):
            return_vals[basis_idx] += b.log_pdf(xs)

        return return_vals

    def _calculate(self, samples, weights_func):
        if self.rule == 'james':
            return self._calculate_by_james_rule(samples, weights_func)
        else:
            raise ValueError('unknown rule')

    def _calculate_by_james_rule(self, samples, weights_func):
        mus = np.append(samples, 0.5 * (self.lb + self.ub))
        sigma_bounds = [(self.ub - self.lb) / min(100.0, mus.size), self.ub - self.lb]

        order = np.argsort(mus)
        sorted_mus = mus[order]
        original_order = np.arange(mus.size)[order]
        prior_pos = np.where(original_order == mus.size - 1)[0][0]

        sorted_mus_with_bounds = np.insert([sorted_mus[0], sorted_mus[-1]], 1, sorted_mus)
        sigmas = np.maximum(sorted_mus_with_bounds[1:-1] - sorted_mus_with_bounds[0:-2],
                            sorted_mus_with_bounds[2:] - sorted_mus_with_bounds[1:-1])
        sigmas = np.clip(sigmas, sigma_bounds[0], sigma_bounds[1])
        sigmas[prior_pos] = sigma_bounds[1]

        weights = weights_func(mus.size)
        weights /= weights.sum()

        return weights, mus, sigmas[original_order]