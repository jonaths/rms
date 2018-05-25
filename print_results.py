import matplotlib.pyplot as plt
import numpy as np
import sys

reward = np.load('statistics/reward.npy')
steps = np.load('statistics/step.npy')
end_state = np.load('statistics/end_state.npy')


# evenly sampled time at 200ms intervals
t = np.arange(0., reward.shape[1], 1)

plt.subplot(3, 1, 1)
plt.plot(t, np.average(reward, axis=0))
# plt.plot(t, middle_mavg_reward)
# plt.plot(t, bottom_mavg_reward)
plt.title('Windyworld Environment behavior per 1000 episodes')
plt.ylabel('Average reward')

plt.subplot(3, 1, 2)
plt.plot(t, np.average(steps, axis=0))
# plt.plot(t, middle_mavg_steps)
# plt.plot(t, bottom_mavg_steps)
plt.ylabel('Average steps')

plt.subplot(3, 1, 3)
search_states = [15, 19, 23, 27]
count = [1 if s in search_states else 0 for s in end_state.flatten()]
count = np.array(count).reshape(reward.shape)
# print(end_state)
# print(count)
#
# print(np.cumsum(count, axis=1))
# print(np.average(np.cumsum(count, axis=1), axis=0))

plt.plot(t, np.average(np.cumsum(count, axis=1), axis=0))
# plt.plot(t, middle_end_count, label='middle')
# plt.plot(t, bottom_end_count, label='bottom')
plt.xlabel('Episodes x 100')
plt.ylabel('Average failure')

plt.legend()
plt.show()



