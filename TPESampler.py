import numpy as np
from Functions.GammaFunction import GammaFunction
from Functions.GaussKernel import GaussKernel
from Estimators.NumericalParzenEstimator import NumericalParzenEstimator
from Estimators.CategoricalParzenEstimator import CategoricalParzenEstimator
from sklearn.preprocessing import MinMaxScaler
import sys

eps = sys.float_info.epsilon
from pymoo.factory import get_performance_indicator


def default_weights(x):
    # default is uniform weights
    # we empirically confirmed that the recency weighting heuristic adopted in
    # Bergstra et al. (2013) seriously degrades performance in multiobjective optimization
    if x == 0:
        return np.asarray([])
    else:
        return np.ones(x)


class TPESampler:
    def __init__(self,
                 hp,
                 observations,
                 random_state,
                 n_ei_candidates=24,
                 rule='james',
                 gamma_func=GammaFunction(),
                 weights_func=default_weights,
                 split_cache=None,
                 scalarize=False,
                 W=None):
        self.hp = hp
        self._observations = observations
        self._random_state = random_state
        self.n_ei_candidates = n_ei_candidates
        self.gamma_func = gamma_func
        self.weights_func = weights_func
        self.opt = self.sample
        self.rule = rule
        self.scalarize = scalarize
        self.W = W
        if split_cache:
            self.split_cache = split_cache
        else:
            self.split_cache = dict()

    def sample_from_lower(self, useNewSampling=1):
        hp_values, ys = self._load_hp_values()
        n_lower = self.gamma_func(len(hp_values))
        if ys.shape[1] == 1:
            lower_vals, upper_vals = self._split_observationsSO(hp_values, ys, self.gamma_func.gamma)
        else:
            if self.scalarize:
                lower_vals, upper_vals = self._split_observationsScalarize(hp_values, ys, self.gamma_func.gamma)
            else:
                lower_vals, upper_vals = self._split_observations(hp_values, ys, n_lower)

        var_type = self._distribution_type()

        sorted_by_score =[]
        sorted_scores = 0
        if var_type in [float, int]:
            hp_value, sorted_by_score, sorted_scores = self._sample_numerical_from_lower(var_type, lower_vals, upper_vals)
        else:
            #TODO this method was not update with the new sampling
            hp_value = self._sample_categorical(lower_vals, upper_vals)

        for i in range(len(hp_value)):
            if useNewSampling!=1:
                hp_value[i] = self._revert_hp(sorted_by_score[i])
            else:
                hp_value[i] = self._revert_hp(hp_value[i])
        return hp_value, sorted_scores

    def sample(self):
        hp_values, ys = self._load_hp_values()
        n_lower = self.gamma_func(len(hp_values))
        if ys.shape[1] == 1:
            lower_vals, upper_vals = self._split_observationsSO(hp_values, ys, self.gamma_func.gamma)
        else:
            if self.scalarize:
                lower_vals, upper_vals = self._split_observationsScalarize(hp_values, ys, self.gamma_func.gamma)
            else:
                lower_vals, upper_vals = self._split_observations(hp_values, ys, n_lower)
        var_type = self._distribution_type()

        if var_type in [float, int]:
            hp_value = self._sample_numerical(var_type, lower_vals, upper_vals)
        else:
            hp_value = self._sample_categorical(lower_vals, upper_vals)
        return self._revert_hp(hp_value)

    def _split_observationsSO(self, hp_values, ys, gamma):
        SPLITCACHE_KEY = str(ys)
        if SPLITCACHE_KEY in self.split_cache:
            lower_indices = self.split_cache[SPLITCACHE_KEY]['lower_indices']
            upper_indices = self.split_cache[SPLITCACHE_KEY]['upper_indices']
        else:
            indices = np.array(range(len(ys)))
            lower_indices = np.array([], dtype=int)

            percentile = np.percentile(ys, gamma * 100)
            lower_indices = np.argwhere(ys[:, 0] < percentile)[:, 0]

            upper_indices = np.setdiff1d(indices, lower_indices)

            self.split_cache[SPLITCACHE_KEY] = {
                'lower_indices': lower_indices, 'upper_indices': upper_indices}

        return hp_values[lower_indices], hp_values[upper_indices]

    def _split_observations(self, hp_values, ys, n_lower):
        SPLITCACHE_KEY = str(ys)
        if SPLITCACHE_KEY in self.split_cache:
            lower_indices = self.split_cache[SPLITCACHE_KEY]['lower_indices']
            upper_indices = self.split_cache[SPLITCACHE_KEY]['upper_indices']
        else:
            rank = self.nondominated_sort(ys)
            indices = np.array(range(len(ys)))
            lower_indices = np.array([], dtype=int)

            # nondominance rank-based selection
            i = 0
            while len(lower_indices) + sum(rank == i) <= n_lower:
                lower_indices = np.append(lower_indices, indices[rank == i])
                i += 1

            # hypervolume contribution-based selection
            ys_r = ys[rank == i]
            indices_r = indices[rank == i]
            worst_point = np.max(ys, axis=0)
            reference_point = np.maximum(
                np.maximum(
                    1.1 * worst_point,  # case: value > 0
                    0.9 * worst_point  # case: value < 0
                ),
                np.full(len(worst_point), eps)  # case: value = 0
            )

            S = []
            contributions = []
            for j in range(len(ys_r)):
                hv = get_performance_indicator("hv", ref_point=reference_point)
                contributions.append(hv.calc(np.asarray([ys_r[j]])))
            while len(lower_indices) + 1 <= n_lower:
                hv_S = 0
                if len(S) > 0:
                    hv = get_performance_indicator("hv", ref_point=reference_point)
                    hv_S = hv.calc(np.asarray(S))
                index = np.argmax(contributions)
                contributions[index] = -1e9  # mark as already selected
                for j in range(len(contributions)):
                    if j == index:
                        continue
                    p_q = np.max([ys_r[index], ys_r[j]], axis=0)

                    hv = get_performance_indicator("hv", ref_point=reference_point)

                    contributions[j] = contributions[j] \
                                       - (hv.calc(np.asarray(S + [p_q])) - hv_S)
                S = S + [ys_r[index]]
                lower_indices = np.append(lower_indices, indices_r[index])
            upper_indices = np.setdiff1d(indices, lower_indices)

            self.split_cache[SPLITCACHE_KEY] = {
                'lower_indices': lower_indices, 'upper_indices': upper_indices}

        return hp_values[lower_indices], hp_values[upper_indices]

    def _split_observationsScalarize(self, hp_values, ys, gamma):
        SPLITCACHE_KEY = str(ys)
        if SPLITCACHE_KEY in self.split_cache:
            lower_indices = self.split_cache[SPLITCACHE_KEY]['lower_indices']
            upper_indices = self.split_cache[SPLITCACHE_KEY]['upper_indices']
        else:
            # Proceed the scalarization
            # Randomly select a point
            weight = self.W[np.random.randint(0, self.W.shape[0]), :]
            replications = 1  # Keep it generic for Stochastic optimization

            Y_mean = np.zeros(ys.shape[0])
            for i in range(ys.shape[0]):
                point = np.asarray(ys[i, :])
                pcheby_term1 = np.max(point * np.tile(weight, (replications, 1)), axis=1)
                pcheby_term2 = np.sum(point * np.tile(weight, (replications, 1)), axis=1)
                pcheby = pcheby_term1 + 0.05 * pcheby_term2
                Y_mean[i] = pcheby

            # # Normalize objectives
            scaler = MinMaxScaler()
            Y = scaler.fit_transform(np.asarray([Y_mean]).T)

            indices = np.array(range(len(ys)))

            percentile = np.percentile(Y[:, 0], gamma * 100)
            lower_indices = np.argwhere(Y < percentile)[:, 0]

            upper_indices = np.setdiff1d(indices, lower_indices)

            self.split_cache[SPLITCACHE_KEY] = {
                'lower_indices': lower_indices, 'upper_indices': upper_indices}

        return hp_values[lower_indices], hp_values[upper_indices]

    def _distribution_type(self):
        cs_dist = str(type(self.hp))

        if 'Integer' in cs_dist:
            return int
        elif 'Float' in cs_dist:
            return float
        elif 'Categorical' in cs_dist:
            var_type = type(self.hp.choices[0])
            if var_type == str or var_type == bool:
                return var_type
            else:
                raise ValueError('The type of categorical parameters must be "bool" or "str".')
        else:
            raise NotImplementedError('The distribution is not implemented.')

    def _get_hp_info(self):
        try:
            if not self.hp.log:
                return self.hp.lower, self.hp.upper, self.hp.q, self.hp.log
            else:
                return np.log(self.hp.lower), np.log(self.hp.upper), self.hp.q, self.hp.log
        except NotImplementedError:
            raise NotImplementedError('Categorical parameters do not have the log scale option.')

    def _convert_hp(self, hp_value):
        try:
            lb, ub, _, log = self._get_hp_info()
            hp_value = np.log(hp_value) if log else hp_value
            return (hp_value - lb) / (ub - lb)
        except NotImplementedError:
            raise NotImplementedError('Categorical parameters do not have lower and upper options.')

    def _revert_hp(self, hp_converted_value):
        try:
            cs_dist = str(type(self.hp))
            if 'Categorical' in cs_dist:
                return hp_converted_value
            else:
                lb, ub, q, log = self._get_hp_info()
                var_type = self._distribution_type()
                hp_value = (ub - lb) * hp_converted_value + lb
                hp_value = np.exp(hp_value) if log else hp_value
                hp_value = np.round(hp_value / q) * q if q is not None else hp_value
                return float(hp_value) if var_type is float else int(np.round(hp_value))
        except NotImplementedError:
            raise NotImplementedError('Categorical parameters do not have lower and upper options.')

    def _load_hp_values(self):
        hp_values = np.array([h['x'][self.hp.name]
                              for h in self._observations if self.hp.name in h['x']])
        cs_dist = str(type(self.hp))
        if 'Categorical' not in cs_dist:
            hp_values = np.array([self._convert_hp(hp_value) for hp_value in hp_values])
        ys = np.array([np.array(list(h['f'].values())) \
                       for h in self._observations if self.hp.name in h['x']])
        # order the newest sample first
        hp_values = np.flip(hp_values)
        ys = np.flip(ys, axis=0)
        return hp_values, ys

    def _sample_numerical(self, var_type, lower_vals, upper_vals):
        q, log, lb, ub, converted_q = self.hp.q, self.hp.log, 0., 1., None

        if var_type is int or q is not None:
            if not log:
                converted_q = 1. / (self.hp.upper - self.hp.lower) \
                    if q is None else q / (self.hp.upper - self.hp.lower)
                lb -= 0.5 * converted_q
                ub += 0.5 * converted_q

        pe_lower = NumericalParzenEstimator(
            lower_vals, lb, ub, self.weights_func, q=converted_q, rule=self.rule)
        pe_upper = NumericalParzenEstimator(
            upper_vals, lb, ub, self.weights_func, q=converted_q, rule=self.rule)
        return self._compare_candidates(pe_lower, pe_upper)

    def _sample_numerical_from_lower(self, var_type, lower_vals, upper_vals):
        q, log, lb, ub, converted_q = self.hp.q, self.hp.log, 0., 1., None

        if var_type is int or q is not None:
            if not log:
                converted_q = 1. / (self.hp.upper - self.hp.lower) \
                    if q is None else q / (self.hp.upper - self.hp.lower)
                lb -= 0.5 * converted_q
                ub += 0.5 * converted_q

        pe_lower = NumericalParzenEstimator(
            lower_vals, lb, ub, self.weights_func, q=converted_q, rule=self.rule)
        pe_upper = NumericalParzenEstimator(
            upper_vals, lb, ub, self.weights_func, q=converted_q, rule=self.rule)

        samples_lower = pe_lower.sample_from_density_estimator(
            self._random_state, self.n_ei_candidates)
        scores = pe_lower.log_likelihood(samples_lower) - pe_upper.log_likelihood(samples_lower)
        index_sort = np.argsort(scores)

        return samples_lower, samples_lower[index_sort], scores[index_sort]

    def _sample_categorical(self, lower_vals, upper_vals):
        choices = self.hp.choices
        n_choices = len(choices)
        lower_vals = [choices.index(val) for val in lower_vals]
        upper_vals = [choices.index(val) for val in upper_vals]

        pe_lower = CategoricalParzenEstimator(
            lower_vals, n_choices, self.weights_func)
        pe_upper = CategoricalParzenEstimator(
            upper_vals, n_choices, self.weights_func)

        best_choice_idx = int(self._compare_candidates(pe_lower, pe_upper))
        return choices[best_choice_idx]

    def _compare_candidates(self, pe_lower, pe_upper):
        samples_lower = pe_lower.sample_from_density_estimator(
            self._random_state, self.n_ei_candidates)
        best_idx = np.argmax(
            pe_lower.log_likelihood(samples_lower) - pe_upper.log_likelihood(samples_lower))
        return samples_lower[best_idx]

    def nondominated_sort(self, points):
        points = points.copy()
        ranks = np.zeros(len(points))
        r = 0
        c = len(points)
        while c > 0:
            extended = np.tile(points, (points.shape[0], 1, 1))
            dominance = np.sum(np.logical_and(
                np.all(extended <= np.swapaxes(extended, 0, 1), axis=2),
                np.any(extended < np.swapaxes(extended, 0, 1), axis=2)), axis=1)
            points[dominance == 0] = 1e9  # mark as used
            ranks[dominance == 0] = r
            r += 1
            c -= np.sum(dominance == 0)
        return ranks

    def setRandomState(self, rs):
        self._random_state = rs
