from optproblems.dtlz import DTLZ1
import numpy as np

f = DTLZ1(2, 3)

n= np.random.randn(3,1000)

j = f.objective_function(np.asarray(n))
print(j)