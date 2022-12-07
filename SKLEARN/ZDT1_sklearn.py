from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt.sampler import Lhs
from skopt.space import Space
import math
from pyswarms.single.global_best import GlobalBestPSO
import pyswarms as ps
from scipy.stats import norm
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from sklearn.neighbors import KernelDensity
from numpy.random import uniform
from sklearn.preprocessing import MinMaxScaler
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from Hypervolume import HyperVolume
import optproblems.zdt

random_seed = 1990


def function_noiseless(x):
    D = x.shape[0]
    #     print(x.shape)
    f = optproblems.zdt.ZDT1(D)
    fitness = np.asarray(f.objective_function(x))

    return fitness

def optimal_set():
    true_front = np.empty((0, 2))

    for f1 in np.linspace(0, 1, num=1000):
        f2 = 1 - np.sqrt(f1)
        true_front = np.vstack([true_front, [f1, f2]])

    true_front = pd.DataFrame(true_front, columns=['f1', 'f2'])  # convert to DataFrame
    return np.asarray(true_front)


def function_noisy(x, replications, rng, constants, level=1, case=1):
    #     return a matrix of r replications x M objectives
    #     x is a point
    f = np.zeros((replications, constants.shape[0]))
    #     print(f.shape)

    fr = function_noiseless(x)
    for m in range(f.shape[1]):  # for each objective
        a = constants[m, 0];
        t = constants[m, 1];

        temp = []
        for r in range(replications):
            noise = rng.normal(loc=0, scale=(a * fr[m] + a * t))
            f[r, m] = fr[m] + noise

    return f


def heterogeneus_noise(objectives, level=1, case=1, rep_budget=5):
    M = objectives.shape[1]
    linear_const = np.zeros((M, 2))
    noise_sd_range = np.zeros((M, 2))

    for m in range(M):
        obj_i = objectives[:, m]
        rank_obj_i = np.sort(obj_i)[::-1]
        rank_obj_i[0] = 1 if m == 1 else 10
        rank_obj_i[-1] = 0
        range_obj_i = np.abs(rank_obj_i[0] - rank_obj_i[-1]);

        # Bounds for noise s.d.
        if level == 1:
            lower_obj_i = 0.01 / np.sqrt(rep_budget) * range_obj_i;
            upper_obj_i = 0.5 / np.sqrt(rep_budget) * range_obj_i;
        elif level == 2:
            lower_obj_i = 0.5 / np.sqrt(rep_budget) * range_obj_i;
            upper_obj_i = 1.5 / np.sqrt(rep_budget) * range_obj_i;
        else:
            lower_obj_i = 1 / np.sqrt(rep_budget) * range_obj_i;
            upper_obj_i = 2 / np.sqrt(rep_budget) * range_obj_i;

        if case == 1:  # Best case
            b_obj_i = (rank_obj_i[0] * lower_obj_i - rank_obj_i[-1] * upper_obj_i) / (upper_obj_i - lower_obj_i);
            a_obj_i = lower_obj_i / (rank_obj_i[-1] + b_obj_i);
        else:  # Worst case
            b_obj_i = (rank_obj_i[-1] * lower_obj_i - rank_obj_i[0] * upper_obj_i) / (upper_obj_i - lower_obj_i);
            a_obj_i = lower_obj_i / (rank_obj_i[0] + b_obj_i);

        min_noise = a_obj_i * rank_obj_i[-1] + a_obj_i * b_obj_i;
        max_noise = a_obj_i * rank_obj_i[0] + a_obj_i * b_obj_i;

        linear_const[m, 0] = a_obj_i;
        linear_const[m, 1] = b_obj_i;
        noise_sd_range[m, 0] = min_noise;
        noise_sd_range[m, 1] = max_noise;

    return linear_const

tf = optimal_set()
def plot_pf(non_dominated_points, true_points, solutions):
    m = np.mean(non_dominated_points, axis=1)
    std = np.sqrt(np.var(non_dominated_points, axis=1))
    max_data = m + std  # np.max(non_dominated_points, axis=1)
    min_data = m - std  # np.min(non_dominated_points, axis=1)

    ells = [Ellipse(xy=[m[i, 0], m[i, 1]],
                    width=np.abs(max_data[i, 0] - min_data[i, 0]),
                    height=np.abs(max_data[i, 1] - min_data[i, 1]))
            for i in range(m.shape[0])]

    fig, ax = plt.subplots(figsize=(10, 10))

    index = 0
    for e in ells:
        ax.add_patch(e)

        if index == 0:
            e.set(clip_box=ax.bbox, alpha=0.2, facecolor="peachpuff", edgecolor='silver', label='Uncertainty')
        else:
            e.set(clip_box=ax.bbox, alpha=0.2, facecolor="peachpuff", edgecolor='silver')
        index += 1

    #         e.set_edgecolor('silver')
    max_axes = np.max([max_data[:, 0], max_data[:, 1]])
    ax.set_xlim(0, max_axes + .5)
    ax.set_ylim(0, max_axes + .5)

    ax.scatter(m[:, 0], m[:, 1], color='k', s=50,
               zorder=len(ells) + 1, label="Observed PF")
    ax.scatter(tf[:, 0], tf[:, 1], label="Optimal PF")
    ax.scatter(solutions[:, 0], solutions[:, 1], color='orange', marker='+', zorder=len(ells) + 2, label="Solutions")
    ax.legend()
    return ax

def compute_hypervolume(m, referencePoint):
    hv = HyperVolume(referencePoint)
    volume = hv.compute(m)
    return volume

def pareto_front(points, level=0, index=False):

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
    if index:
#         return ranks
        return [i for i in range(len(ranks)) if ranks[i] == level]
    else:
        ind = [i for i in range(len(ranks)) if ranks[i] == level]
        return points[ind]

def pso_function(candidates, model, model_det, Z_min):
    #     print(candidates.shape)

    Z_min = Z_min[0, 0]
    SK_gau = model.predict(candidates, return_std=False)
    [SK_gau_det, dk_mse] = model_det.predict(candidates, return_std=True)
    SK_gau = SK_gau[:, 0]
    mse = dk_mse #np.sqrt(np.abs(dk_mse))

    mei = (Z_min - SK_gau) * norm.cdf((Z_min - SK_gau) / mse, 0, 1) + mse * norm.pdf((Z_min - SK_gau) / mse, 0, 1)

    return -mei


minx = 0.
maxx = 1.
D = 5
n = 11 * D - 1
M = 2
predictions = 1000
iterations = 130
replications = 50

rng = np.random.RandomState(random_seed)

# Prepare heterogeneus noise
range_objectives = np.asarray([[1, 10],
                               [0, 0]])
constants = heterogeneus_noise(range_objectives, rep_budget=replications)

# Options_PSO
bounds = ([minx, minx, minx, minx, minx], [maxx, maxx, maxx, maxx, maxx])
options_pso = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}

# space = Space([(minx, maxx), (minx, maxx), (minx, maxx), (minx, maxx), (minx, maxx)])
# lhs = Lhs(criterion="maximin", iterations=10000)
# X = np.asarray(lhs.generate(space.dimensions, n, rng))

# Read the initial set from csv
X = pd.read_csv("../data/lhs_ZDT1.csv", sep=',', header=0)
X = X.values[:, 1:]

k = X.shape[1]

truth = np.empty((X.shape[0], M))  # It's a biobjective problem
for index in range(truth.shape[0]):
    row = np.transpose(X[index, :])
    [truth[index, 0], truth[index, 1]] = function_noiseless(row)

# Replications
data = []
for p in X:
    point_replications = function_noisy(np.transpose([p]), replications, rng, constants)
    data.append(point_replications)
data = np.asarray(data)  # format >> [points, replications, objective]

# Weights for the scalarization
n_weights = iterations;
W = np.empty([n_weights, 2])
W[:, 0] = np.linspace(0, 1, n_weights)
W[:, 1] = np.abs(W[:, 0] - np.transpose(np.ones((n_weights, 1))))

hv_history = []
kernel = RBF(1, (1e-2, 1e2))

# Optimization procedure
for it in range(iterations):
    print("Iteration: ", it, "/", iterations)

    # Prepare de optimization
    Y_mean = np.zeros(X.shape[0])
    Vhat = np.zeros(X.shape[0])
    B = np.asarray([np.ones(X.shape[0])])

    #### Scalarization
    # Randomly select a point
    weight = W[np.random.randint(0, W.shape[0]), :]

    for i in range(X.shape[0]):
        point = np.asarray(data[i])
        pcheby_term1 = np.max(point * np.tile(weight, (replications, 1)), axis=1)
        pcheby_term2 = np.sum(point * np.tile(weight, (replications, 1)), axis=1)
        pcheby = pcheby_term1 + 0.05 * pcheby_term2
        Y_mean[i] = np.mean(pcheby)
        Vhat[i] = np.var(pcheby)
    Vhat = np.asarray([Vhat]).T
    # Normalize objectives
    scaler = MinMaxScaler()
    Y_mean = scaler.fit_transform(np.asarray([Y_mean]).T)

    #     print(X.shape, Y_mean.shape, B.T.shape, Vhat.shape)


    #     skriging_model_2 = eng.SKfit_new(X_matlab,Y_matlab, B, Vhat, 2, 3,
    #                                      matlab.double(np.asarray([random_seed]).tolist()), nargout=1);
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=Vhat[:, 0],
                                   n_restarts_optimizer=10).fit(X, Y_mean)
    gpr_det = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                       n_restarts_optimizer=10).fit(X, Y_mean)

    #     Obtain the best point and evaluate it with SK
    index_best = np.argmin(Y_mean)

    SK_gau  = gpr.predict(np.asarray([X[index_best, :]]), return_std=False)

    print("Searching infill point...")
    optimizer = GlobalBestPSO(n_particles=50, dimensions=D, options=options_pso, bounds=bounds)
    [mei, pos] = optimizer.optimize(pso_function, 100, verbose=True, model=gpr, model_det=gpr_det, Z_min=SK_gau)
    print("    Done")

    # Save the new point
    X = np.append(X, np.asarray([pos]), axis=0)

    # Simulate the new point
    rep = function_noisy(np.transpose([pos]), replications, rng, constants)
    data = np.append(data, [rep], axis=0)

    # Compute the Hypervolume
    m = np.mean(data, axis=1)
    pf = pareto_front(m, index=True)
    pf = data[pf]
    volume = compute_hypervolume(np.mean(pf, axis=1), [1, 10])
    hv_history.append(volume)

#     plot_evolution(X, XK, skriging_model_2, truthk, it)
m = np.mean(data, axis=1)
pf = pareto_front(m, index=True)
pf = data[pf]
volume=compute_hypervolume(np.mean(pf, axis=1), [1, 10])


ax = plot_pf(pf, tf, m)
ax.scatter(m[:, 0], m[:, 1], color='orange', marker='+', zorder=m.shape[0], label="Solutions")
plt.title("HV "+str(volume))
plt.ylim([0,1])
plt.xlim([0,1])
plt.legend()
plt.show()


fig, ax = plt.subplots(figsize=(10,10))
plt.plot(hv_history)
plt.xlabel("Iterations (Infill points)")
plt.ylabel("Hypervolume")
plt.show()