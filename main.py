import numpy as np
from rms.rms import RmsAlg

# rthres, influence, risk_default, sim_func_name='manhattan', risk_func_name='inverse'
alg = RmsAlg(-1, 2, 0)
# print(alg.calc_similarity(np.array([1, 2]), np.array([2, 3])))
# print(alg.calc_risk(np.array([1, 2]), np.array([2, 3])))

# alg.add_to_v(0, [0, 0])
# alg.add_to_v(1, [0, 1])
# alg.add_to_k(2, [0, 2])

alg.add_to_v(0, [0, 0])
# s, r, sprime, sprime_features=None
alg.update(0, +0, 1, [1, 0])
alg.update(1, +0, 2, [2, 0])
alg.update(2, +0, 3, [3, 0])
alg.update(3, -2, 4, [4, 0])
alg.update(4, -2, 5, [5, 0])

print(alg.v)
print(alg.k)

print(alg.get_risk(3))
