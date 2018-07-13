import gym
import gym_windy
# import C51
from utils.qlearning import QLearningAgent
# from keras.utils import to_categorical
import numpy as np
from rms.rms import RmsAlg
import sys
import math
from conf_cartpole_exp import params
import time

# from plotter import Plotter


def tuple2int(buckets, tpl):
    """
    Recibe dos tuplas y regresa un int
    :param buckets: (1, 1, 6, 3)
    :param tpl: (0, 0, 5, 1)
    :return: 16
    """

    if len(tpl) != len(buckets):
        raise Exception('Invalid tuple len. ')

    len_t = len(buckets)
    for i in range(len_t):
        if tpl[i] >= buckets[i]:
            raise Exception('Invalid tuple value. ')

    # calcula multiplicadores
    mult = []
    for i in range(len_t):
        m = 1
        pointer = i + 1
        while pointer != len_t:
            m = m * buckets[pointer]
            pointer += 1
        mult.append(m)
    # suma y multiplica
    index = sum([tpl[i] * mult[i] for i in range(len_t)])
    return index


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple2int(NUM_BUCKETS, tuple(bucket_indice))


env = gym.make('CartPole-v0')

actionFn = lambda state: range(env.action_space.n)
qLearnOpts = {'gamma': 0.9,
              'alpha': 0.1,
              'epsilon': 0.1,
              'numTraining': 5000,
              'actionFn': actionFn
              }

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = params['num_buckets']  # (x, x', theta, theta')
max_games = params['max_games']
init_epsilon = params['init_epsilon']
end_epsilon = params['end_epsilon']
test_period = params['test_period']
reps = params['reps']

# Number of discrete actions
num_actions = env.action_space.n  # (left, right)
num_states = np.prod(np.array([i for i in NUM_BUCKETS]))
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[2] = params['state_bounds_2']
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
epsilon_step = (init_epsilon - end_epsilon) / (max_games * test_period)

print(STATE_BOUNDS)
# states = [
#     [0., 0., 0., 0.],
#     [0., 0., 0., -0.8726646259971648],
#     [0., 0., 0., -0.9],
# ]
#
# # states = [[0., 0., 0., step] for step in np.linspace(-0.8726646259971648, +0.8726646259971648, 200)]
# states = [[0., 0., step, 0.] for step in np.linspace(-0.23, +0.23, 20)]
#
# for state in states:
#     print(state_to_bucket(state), state)
# sys.exit(0)

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

    # aqui voy... comenzar a probar con render y ver que rms propageue el riesgo en egunda iteraciosn

    # RmsAlg(rthres, influence, risk_default)
    alg = RmsAlg(rthres=params['rms']['rthres'], influence=params['rms']['influence'], risk_default=0)
    alg.add_to_v(state_to_bucket(s_t), s_t)

    misc = {'step_seq': []}
    prev_misc = misc

    # final_state_buffer, reward_buffer, steps_buffer = [], [], []
    GAME = 0
    done = False
    t = 0
    r_t = 0
    agent.startEpisode()

    while GAME < max_games:

        print("Game: {} ---------------------------------------------------").format(GAME)
        print("epsilon: " + str(agent.epsilon))

        while not done:

            bucket_state = state_to_bucket(s_t)

            action_idx = agent.getAction(bucket_state)
            # action_idx = input("Action: ")
            obs, r, done, _ = env.step(action_idx)
            bucket_obs = state_to_bucket(obs)

            if GAME > max_games - 2:
                env.render()
                time.sleep(0.1)

            if bucket_obs in params['terminal_states']:
                slope_r = r
            else:
                slope_r = r
            alg.update(s=bucket_state, r=slope_r, sprime=bucket_obs, sprime_features=obs)

            risk_penalty = alg.get_risk(bucket_obs)

            reward_signal = r + risk_penalty
            # reward_signal = r

            agent.observeTransition(bucket_state, action_idx, bucket_obs, reward_signal)
            misc['step_seq'].append(bucket_state)

            # print("o", np.around(obs, 4))
            # print("=", state_to_bucket(s_t), action_idx, state_to_bucket(obs), reward_signal)
            # print('r', r, risk_penalty)
            # print("d", alg.get_risk_dict_no_zeros())

            prev_misc = misc
            s_t = obs
            t += 1
            r_t += r

        if GAME < max_games * test_period:
            agent.setEpsilon(agent.epsilon - epsilon_step)

        if done:
            done = False
            reward_results[rep][GAME] = r_t
            steps_results[rep][GAME] = t
            end_state_results[rep][GAME] = bucket_obs
            print("total reward: " + str(reward_results[rep][GAME]))
            print("final_state: " + str(bucket_obs))

            s_t = env.reset()
            bucket_state = state_to_bucket(s_t)
            alg.add_to_v(bucket_state, s_t)

            agent.stopEpisode()
            agent.startEpisode()

            t = 0
            r_t = 0
            GAME += 1
            misc['step_seq'] = []

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
