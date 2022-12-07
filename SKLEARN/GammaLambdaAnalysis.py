import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Utilities import pareto_front, compute_hypervolume

problem_names = ["ZDT1", "WFG4", "DTLZ7"]
lambdas = np.linspace(0.01, 0.1, 10)
lambdas = np.append(lambdas, 1)
gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# Load gammas and l results
dict_gl = {}
dict_HV_gl = {}

for id_problem in range(len(problem_names)):
    i = 0
    hv_matrix = np.zeros((len(lambdas), len(gammas)))

    if id_problem == 0:
        ref_point = [1, 10]
    elif id_problem == 1:
        ref_point = [3, 5]
    else:  # 2
        ref_point = [1, 23]

    for l in lambdas:
        j = 0
        for g in gammas:

            file_name = "../output/parameter_exploration_GP_MOTPE/meanObjectives_GP_" + problem_names[id_problem] + "_gamma_" + str(g) + "_l_" + str(l) + ".csv"
            #         print(file_name)
            print(str(i) + str(j), " ", end='')
            data = pd.read_csv(file_name, sep=',', header=0).values[:, 1:]
            dict_gl[str(id_problem)+"_"+str(g) + "_" + str(np.round(l, 2))] = data

            pf = pareto_front(data, index=True)
            pf = data[pf]
            volumen = compute_hypervolume(pf, ref_point)
            hv_matrix[i, j] = volumen

            j += 1
        print()
        i += 1
    dict_HV_gl[str(id_problem)] = hv_matrix

for id_problem in range(len(problem_names)):
    fig, axn = plt.subplots(figsize=(10,10))
    g4 = sns.heatmap(dict_HV_gl[str(id_problem)], linewidth = 0.5 , cmap = 'vlag', annot=True, fmt=".4",
                     cbar_kws={'label': 'Hypervolume'}, ax=axn)
    g4.set_xticklabels(list(map(str,gammas)))
    g4.set_yticklabels([str(np.round(l,2)) for l in lambdas])
    g4.set_xlabel("gamma")
    g4.set_ylabel("lambda")
    plt.title(problem_names[id_problem])
    plt.show()
print("Done")