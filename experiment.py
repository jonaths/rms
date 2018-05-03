import gym
import gym_windy
# import C51
from utils.qlearning import QLearningAgent
# from keras.utils import to_categorical
import numpy as np
import sys

# from plotter import Plotter

env = gym.make("windy-v0")
# env.setWindyProb(0.1)

# agent= C51Agent(5,2,51)

actionFn = lambda state: env.get_possible_actions(state)
qLearnOpts = {'gamma': 0.9,
              'alpha': 0.1,
              'epsilon': 0.1,
              'numTraining': 5000,
              'actionFn': actionFn
              }
agent = QLearningAgent(**qLearnOpts)
s_t = env.reset()

misc = {'sum_reward': 0, 'step_seq': 0}
prev_misc = misc

final_state_buffer, reward_buffer, steps_buffer = [], [], []
GAME = 0
t = 0
r_t = 0
agent.startEpisode()
# real_time_plotter = Plotter()
for i_game in range(10000):
    #  env.render()
    # print(s_t)
    action_idx = agent.getAction(s_t)
    obs, r, done, misc = env.step(action_idx)
    agent.observeTransition(s_t, action_idx, obs, r)
    #   print (s_t,action_idx, obs, r)
    # print (obs,r,done,misc)
    prev_misc = misc
    s_t = obs
    t += 1
    r_t += r

    if env.done:
        #   last=misc['step_seq']
        #   print ("last",last)
        final_state_buffer.append(str(misc['step_seq'][-1]))
        s_t = env.reset()
        agent.stopEpisode()
        agent.startEpisode()
        GAME += 1
        steps_buffer.append(t)
        reward_buffer.append(r_t)
        t = 0
        r_t = 0

        if GAME % agent.stats_window_size == 0:
            # Reset rolling stats buffer
            agent.mavg_reward.append(np.mean(np.array(reward_buffer)))
            agent.var_reward.append(np.std(np.array(reward_buffer)))
            agent.mavg_steps.append(np.mean(np.array(steps_buffer)))
            agent.var_steps.append(np.std(np.array(steps_buffer)))
            agent.end_count.append(final_state_buffer)
            final_state_buffer, reward_buffer, steps_buffer = [], [], []

        if GAME % 100 == 0:
            # real_time_plotter.plot_learning_curve("Learning",agent.mavg_reward, agent.var_reward)
            print(GAME, agent.qvals)

        with open("statistics/ql_stats.txt", "w") as stats_file:
            stats_file.write('Games: ' + str(GAME) + '\n')

            stats_file.write('mavg_reward: ' + str(agent.mavg_reward) + '\n')
            np.save('statistics/mavg_reward.npy', np.array(agent.mavg_reward))

            stats_file.write('var_reward: ' + str(agent.var_reward) + '\n')
            np.save('statistics/var_reward.npy', np.array(agent.var_reward))

            stats_file.write('mavg_steps: ' + str(agent.var_steps) + '\n')
            np.save('statistics/mavg_steps.npy', np.array(agent.mavg_steps))

            stats_file.write('mavg_end_count: ' + str(agent.end_count) + '\n')
