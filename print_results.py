import matplotlib.pyplot as plt
import numpy as np

mavg_reward = np.load('statistics/mavg_reward.npy')
mavg_steps =  np.load('statistics/mavg_steps.npy')
# mavg_end_count =


# # evenly sampled time at 200ms intervals
t = np.arange(0., len(mavg_reward), 1)

plt.subplot(3, 1, 1)
plt.plot(t, mavg_reward)
# plt.plot(t, middle_mavg_reward)
# plt.plot(t, bottom_mavg_reward)
plt.title('Windyworld Environment behavior per 1000 episodes')
plt.ylabel('Average reward')

plt.subplot(3, 1, 2)
plt.plot(t, mavg_steps)
# plt.plot(t, middle_mavg_steps)
# plt.plot(t, bottom_mavg_steps)
plt.ylabel('Average steps')

plt.subplot(3, 1, 3)
# plt.plot(t, end_count, label='learned')
# plt.plot(t, middle_end_count, label='middle')
# plt.plot(t, bottom_end_count, label='bottom')
plt.xlabel('Episodes x 100')
plt.ylabel('Average failure')

plt.legend()
plt.show()

