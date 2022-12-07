import ConfigSpace as CS
import sys
import matplotlib.pyplot as plt
import pandas as pd
from Hypervolume import HyperVolume
from Problems.ZDT import ZDT1
from Problems.Problem import Problem
from MOTPE import MOTPE
import numpy as np

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

def optimal_set():
    true_front = np.empty((0, 2))

    for f1 in np.linspace(0, 1, num=1000):
        f2 = 1 - np.sqrt(f1)
        true_front = np.vstack([true_front, [f1, f2]])

    true_front = pd.DataFrame(true_front, columns=['f1', 'f2'])  # convert to DataFrame
    return np.asarray(true_front)

def compute_hypervolume(m, referencePoint):
    hv = HyperVolume(referencePoint)
    volume = hv.compute(m)
    return volume

eps = sys.float_info.epsilon

seed                = 1990   #args.seed
num_variables       = 5     #args.num_variables
num_initial_samples = 11*num_variables-1    #args.num_initial_samples
num_max_evals       = 130   #args.num_max_evals
num_objectives      = 2     #args.num_objectives
replications        = 50
k                   = 1     #args.k
num_candidates      = 1000    #args.num_candidates
init_method         = 'lhs' #args.init_method
gamma               = 0.1   #args.gamma

base_configuration  = {
    'num_objectives': num_objectives,
    'num_variables': num_variables,
    'replications': replications,
    'k': k,
    'seed': seed}
benchmark_name      = "ZDT1" #args.benchmark_name

f       = ZDT1(benchmark_name, base_configuration)
cs      = f.make_cs(CS.ConfigurationSpace(seed=seed))
problem = Problem(f, cs)
solver  = MOTPE(seed=seed)

range_objectives = np.asarray([[1, 10],
                              [0, 0]])
constants = solver.prepare_heterogeneus_noise(range_objectives, rep_budget=replications)
problem.set_noise(constants)
history = solver.solve(
    problem,
    {'num_initial_samples': num_initial_samples,
     'num_max_evals': num_max_evals,
     'init_method': init_method,
     'num_candidates': num_candidates,
     'scalarize': False,
     'gamma': gamma})


# Save the replications
rep = solver.history_replications()
mmm = pd.DataFrame(rep.reshape(rep.shape[0]*rep.shape[1],2))
mmm.to_csv("output/replications_noisy_MOTPE_ZDT1.csv")

# Save the exploration
m = solver.get_F()
df = pd.DataFrame(m)
df.to_csv("output/exploration_noisy_MOTPE_ZDT1.csv")

m = solver.get_F()
pf = pareto_front(m, index=True)
pf = m[pf]
volume=compute_hypervolume(pf, [1, 10])

fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(m[:, 0], m[:, 1], color='orange', marker='+', zorder=m.shape[0], label="Feasible solutions")
ax.scatter(pf[:, 0], pf[:, 1], color='k', s=50, label="Pareto front")
plt.title("HV "+str(volume))
tf = optimal_set()
ax.scatter(tf[:,0], tf[:,1], label="Optimal PF")
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.legend(loc='upper right')
plt.savefig('output/PF_noisy_MOTPE.png', dpi=1200, bbox_inches='tight')

m = solver.get_F()
hv_temp = []

for index in range(num_initial_samples, len(m)):
    pf = pareto_front(m[:index], index=True)
    pf = m[pf]
    volume=compute_hypervolume(pf, [1, 10])
    hv_temp.append(volume)

fig, ax = plt.subplots(figsize=(10,10))
plt.plot(hv_temp)
plt.xlabel("Iterations (Number of infill points)")
plt.ylabel("Hypervolume")
plt.savefig('output/HV_noisy_MOTPE.png', dpi=1200, bbox_inches='tight')

# Save the hyervolume history
df = pd.DataFrame(hv_temp)
df.to_csv("output/hypervolume_noisy_MOTPE_ZDT1.csv")

if num_objectives == 2:
    fig = plt.figure(figsize=(8, 6))
    m = solver.get_F()
    f1ss = m[:, 0]
    f2ss = m[:, 1]
    f1s = [fs['f1'] for fs in history['f']]
    f2s = [fs['f2'] for fs in history['f']]
    plt.scatter(f1s, f2s)
    plt.scatter(f1ss, f2ss, marker='+')
    plt.title(benchmark_name)
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.grid()
    plt.show()
else:
#     print(history)
    print("Done!")