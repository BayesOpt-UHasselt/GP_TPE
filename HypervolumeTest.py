
import numpy as np
import pandas as pd
from Hypervolume import HyperVolume
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

font = {'size': 16}
plt.rc('font', **font)


def plot_hypervolumen(file_name, initial_point, gp, motpe, gp_motpe, ref_point):
    hv_gp = []
    hv_motpe = []
    hv_gp_motpe = []
    for index in range(initial_point, gp.shape[0]):
        pf = pareto_front(gp[:index], index=True)
        pf = gp[pf]
        volumen = compute_hypervolume(pf, ref_point)
        hv_gp.append(volumen)

        pf = pareto_front(motpe[:index], index=True)
        pf = motpe[pf]
        volumen = compute_hypervolume(pf, ref_point)
        hv_motpe.append(volumen)

        pf = pareto_front(gp_motpe[:index], index=True)
        pf = gp_motpe[pf]
        volumen = compute_hypervolume(pf, ref_point)
        hv_gp_motpe.append(volumen)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(hv_gp, label="GP")
    ax.plot(hv_motpe, label="MOTPE")
    ax.plot(hv_gp_motpe, label="GP_MOTPE")
    ax.set_xlabel("Iterations (Number of infill points)")
    ax.set_ylabel("Hypervolumen")
    plt.legend(loc="lower right")
    plt.savefig(file_name, dpi=1200, bbox_inches='tight')
    plt.show()


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

# load data exploration
ML_val_motpe = pd.read_csv("notebook\PF_data\ML\exploration_MOTPE_val_ML.csv", sep=',', header=0).values[:,1:]
ML_test_motpe = pd.read_csv("notebook\PF_data\ML\exploration_MOTPE_test_ML.csv", sep=',', header=0).values[:,1:]

ML_val_gp = pd.read_csv("notebook\PF_data\ML\exploration_GP_val_ML.csv", sep=',', header=0).values[:,1:]
ML_test_gp = pd.read_csv("notebook\PF_data\ML\exploration_GP_test_ML.csv", sep=',', header=0).values[:,1:]

ML_val_gp_motpe = pd.read_csv("notebook\PF_data\ML\exploration_GP_MOTPE_val_ML.csv", sep=',', header=0).values[:,1:]
ML_test_gp_motpe = pd.read_csv("notebook\PF_data\ML\exploration_GP_MOTPE_test_ML.csv", sep=',', header=0).values[:,1:]

# load data replications
rML_motpe = pd.read_csv("notebook/PF_data/ML/replications_MOTPE_ML.csv", sep=',', header=0).values[:,1:]
rML_motpe = rML_motpe.reshape((ML_val_motpe.shape[0],
                        int(rML_motpe.shape[0]/ML_val_motpe.shape[0]),
                         ML_val_motpe.shape[1]))

rML_gp = pd.read_csv("notebook/PF_data/ML/replications_GP_ML.csv", sep=',', header=0).values[:,1:]
rML_gp = rML_gp.reshape((ML_val_gp.shape[0],
                        int(rML_gp.shape[0]/ML_val_gp.shape[0]),
                         ML_val_gp.shape[1]))

rML_gp_motpe = pd.read_csv("notebook/PF_data/ML/replications_GP_MOTPE_ML.csv", sep=',', header=0).values[:,1:]
rML_gp_motpe = rML_gp_motpe.reshape((ML_val_gp_motpe.shape[0],
                        int(rML_gp_motpe.shape[0]/ML_val_gp_motpe.shape[0]),
                         ML_val_gp_motpe.shape[1]))

plot_hypervolumen("output\hv_ML_val.pdf", 54, ML_val_gp, ML_val_motpe, ML_val_gp_motpe, [1, 1])
# plot_hypervolumen(54, ML_val_gp, ML_val_motpe, ML_val_gp_motpe, [1, 1])

