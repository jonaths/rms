import gym
import gym_windy
# import C51
from utils.qlearning import QLearningAgent
# from keras.utils import to_categorical
import numpy as np
from rms.rms import RmsAlg
import sys

# from plotter import Plotter

env = gym.make("border-v0")

actionFn = lambda state: env.get_possible_actions(state)
qLearnOpts = {'gamma': 0.9,
              'alpha': 0.1,
              'epsilon': 0.1,
              'numTraining': 5000,
              'actionFn': actionFn
              }
agent = QLearningAgent(**qLearnOpts)

max_games = 1000
reps = 10

reward_results = np.zeros((reps, max_games))
steps_results = np.zeros((reps, max_games))
end_state_results = np.zeros((reps, max_games))

for rep in range(reps):

    print("Rep: {} =========================================================").format(rep)

    s_t = env.reset()

    alg = RmsAlg(-1, 2, 0)
    alg.add_to_v(s_t, env.ind2coord(s_t))

    misc = {'sum_reward': 0, 'step_seq': 0}
    prev_misc = misc

    final_state_buffer, reward_buffer, steps_buffer = [], [], []
    GAME = 0
    done = False
    t = 0
    r_t = 0
    agent.startEpisode()

    while GAME < max_games:

        print("Game: {} ---------------------------------------------------").format(GAME)

        while not done:

            action_idx = agent.getAction(s_t)
            obs, r, done, misc = env.step(action_idx)
            alg.update(s_t, r, obs, env.ind2coord(obs))
            risk_penalty = alg.get_risk(obs)
            # agent.observeTransition(s_t, action_idx, obs, r + risk_penalty)
            agent.observeTransition(s_t, action_idx, obs, r)

            print("=", s_t, action_idx, obs, r)
            env.render()
            # print("risk_dict", alg.get_risk_dict())

            prev_misc = misc
            s_t = obs
            t += 1
            r_t += r

            input("XXX")

        if env.done:
            done = False
            s_t = env.reset()
            agent.stopEpisode()
            agent.startEpisode()


            reward_results[rep][GAME] = r_t
            steps_results[rep][GAME] = t
            end_state_results[rep][GAME] = misc['step_seq'][-1]

            t = 0
            r_t = 0
            GAME += 1

    np.save('statistics/reward.npy', np.array(reward_results))
    np.save('statistics/step.npy', np.array(steps_results))
    np.save('statistics/end_state.npy', np.array(end_state_results))

    print("END")

    print(np.load('statistics/reward.npy'))
