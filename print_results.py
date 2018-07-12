import matplotlib.pyplot as plt
import numpy as np
import sys
from conf_cartpole_exp import params

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Running mean/Moving average
def running_mean(array, period):
    return np.convolve(array, np.ones((period,))/period, mode='valid')


reward = np.load('statistics/reward.npy')
steps = np.load('statistics/step.npy')
end_state = np.load('statistics/end_state.npy')
rolling_window_size = 40

# evenly sampled time at 200ms intervals
t = np.arange(0., reward.shape[1], 1)
t_rolling_window = t[(rolling_window_size - 1):]

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)


axs[0].plot(t_rolling_window, running_mean(np.average(reward, axis=0), rolling_window_size))
# plt.plot(t, middle_mavg_reward)
# plt.plot(t, bottom_mavg_reward)
# plt.title('Windyworld Environment behavior per 1000 episodes')
axs[0].set_ylabel('Avg. reward')



axs[1].plot(t_rolling_window, running_mean(np.average(steps, axis=0), rolling_window_size))
# plt.plot(t, middle_mavg_steps)
# plt.plot(t, bottom_mavg_steps)
axs[1].set_ylabel('Avg. steps')


# search_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 54, 57, 58, 59, 60, 61, 62]
# search_states = [15, 19, 23, 27]
search_states = params['terminal_states']
count = [1 if s in search_states else 0 for s in end_state.flatten()]
count = np.array(count).reshape(reward.shape)

axs[2].plot(t, np.average(np.cumsum(count, axis=1), axis=0))
axs[2].set_ylabel('Avg. failure')
axs[2].set_xlabel('Episodes')

plt.tight_layout()
plt.savefig('figures/results.pdf')
plt.show()
