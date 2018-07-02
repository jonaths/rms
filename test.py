import numpy as np
# import gym
# import gym_windy
#
# # import C51
# from rms.rms import RmsAlg
#
# # from keras.utils import to_categorical
#
# # from plotter import Plotter
#
# done = False
# env = gym.make("beach-v0")
# env.reset()
# # rthres, influence, risk_default, sim_func_name='manhattan', risk_func_name='inverse'
# alg = RmsAlg(-1, 2, 0)
# current_state = env.start_state
# alg.add_to_v(current_state, env.ind2coord(current_state))
# print(current_state)
# episode = 0
# while not done:
#     print("Start ===========")
#     action = input("Action: ")
#     state, reward, done, _ = env.step(action)
#     # s, r, sprime, sprime_features=None
#     alg.update(current_state, reward, state, env.ind2coord(state))
#     print("=", current_state, action, state, reward)
#     env.render()
#     current_state = state
#     print("risk_dict", alg.get_risk_dict())
#     if done:
#         print("Done -----------------")
#         print("v", alg.v)
#         print("k", alg.k)
#         episode += 1
#         env.reset()
#         done = False

end_state = np.load('statistics/end_state.npy')
print end_state