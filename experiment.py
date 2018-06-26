import gym
import gym_windy
# import C51
from utils.qlearning import QLearningAgent
# from keras.utils import to_categorical
import numpy as np
from rms.rms import RmsAlg
import sys

# from plotter import Plotter

env = gym.make("beach-v0")

actionFn = lambda state: env.get_possible_actions(state)
qLearnOpts = {'gamma': 0.9,
              'alpha': 0.1,
              'epsilon': 0.1,
              'numTraining': 5000,
              'actionFn': actionFn
              }

num_actions = 4
num_states = 64
max_games = 1000
init_epsilon = 0.1
end_epsilon = 0.1
test_period = 0.9
epsilon_step = (init_epsilon - end_epsilon) / (max_games * test_period)
reps = 5

reward_results = np.zeros((reps, max_games))
steps_results = np.zeros((reps, max_games))
end_state_results = np.zeros((reps, max_games))
q_table = np.zeros((reps, num_states, num_actions))
policy = np.zeros((reps, num_states))

for rep in range(reps):

    agent = QLearningAgent(**qLearnOpts)
    agent.setEpsilon(init_epsilon)

    print("Rep: {} =========================================================").format(rep)

    s_t = env.reset()

    # RmsAlg(rthres, influence, risk_default)
    alg = RmsAlg(rthres=-1, influence=1, risk_default=0)
    alg.add_to_v(s_t, env.ind2coord(s_t))

    misc = {'sum_reward': 0, 'step_seq': 0, 'elevation': env.default_elevation}
    prev_misc = misc

    final_state_buffer, reward_buffer, steps_buffer = [], [], []
    GAME = 0
    done = False
    t = 0
    r_t = 0
    agent.startEpisode()

    while GAME < max_games:

        print("Game: {} ---------------------------------------------------").format(GAME)
        print(agent.epsilon)

        while not done:

            action_idx = agent.getAction(s_t)
            # action_idx = input("Action: ")
            obs, r, done, misc = env.step(action_idx)
            # env.render()

            # probando aqui... ver como se asigna el riesgo.

            # la playa esta hacia abajo
            # si prev - now < 0 entonces se movio hacia abajo -> penalizar
            # si prev - now >= 0 entonces no se movio o se movio hacia arriba -> no penalizar
            slope = (prev_misc['elevation'] - misc['elevation']) / 1.
            if slope > 0:
                slope_r = -5
            else:
                slope_r = r
            alg.update(s=s_t, r=slope_r, sprime=obs, sprime_features=env.ind2coord(obs))

            # alg.update(s=s_t, r=r, sprime=obs, sprime_features=env.ind2coord(obs))

            risk_penalty = alg.get_risk(obs)
            # print(r, risk_penalty)

            reward_signal = r + risk_penalty
            # reward_signal = r
            agent.observeTransition(s_t, action_idx, obs, reward_signal)

            print("=", s_t, action_idx, obs, reward_signal, misc['elevation'])

            # env.render()
            # print("risk_    dict", alg.get_risk_dict_no_zeros())

            prev_misc = misc
            s_t = obs
            t += 1
            r_t += r

        if GAME < max_games * test_period:
            agent.setEpsilon(agent.epsilon - epsilon_step)

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

    states = range(0, num_states)

    current_q_table = []
    current_policy = []
    for s in states:
        action_values = []
        for a in range(num_actions):
            action_values.append(agent.getQValue(s, a))
            policy_state = agent.getPolicy(s)
        print(s, policy_state, action_values)
        current_q_table.append(action_values)
        current_policy.append(policy_state)
    q_table[rep] = np.array(current_q_table)
    policy[rep] = np.array(current_policy)

np.save('statistics/q_table.npy', q_table)
np.save('statistics/policy.npy', policy)

test1 = np.load('statistics/q_table.npy')
test2 = np.load('statistics/policy.npy')
print(test1.shape)
print(test1[-1])
print(test2[-1])
print("END")
