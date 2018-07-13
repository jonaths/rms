import numpy as np
import matplotlib.pyplot as plt

policy = np.load('statistics/policy.npy')
print(policy.shape)
exp_number = -3
for i in range(policy.shape[1]):
    print(i, policy[exp_number][i])

# cargar tabla q
q_table = np.load('statistics/q_table.npy')
print(q_table.shape)

# generar imagen a partir de datos
img_data = np.transpose(np.max(q_table[exp_number].reshape(8, 3, 2), axis=2))

# # para que se vea como el escenario original
# img_data = np.transpose(
#     # el maximo de las acciones
#     np.max(
#         # el promedio de todos los experimentos
#         # (states, actions)
#         np.average(
#             # poner el eje que tiene los experimentos al final
#             # (experiments, states, actions ) -> (states, actions, experiments)
#             np.transpose(q_table, (1, 2, 0)), axis=2).reshape(8, 8, 4), axis=2))
print(img_data)


plt.imshow(img_data, cmap='Blues')
plt.imsave('figures/qvalues.png', img_data, cmap='Blues')
plt.show()


end_states = np.load('statistics/end_state.npy')
unique, counts = np.unique(end_states, return_counts=True)
print(dict(zip(unique, counts)))