import gym
import gym_windy
# import C51
from utils.qlearning import QLearningAgent
# from keras.utils import to_categorical
import numpy as np
from rms.rms import RmsAlg
import sys
import math

# from plotter import Plotter


def tuple2int(tpl):
    int_str = ''
    for i in range(len(tpl)):
        int_str += str(tpl[i])
    return int(int_str)


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
    return tuple2int(tuple(bucket_indice))


env = gym.make('CartPole-v0')

actionFn = lambda state: range(env.action_space.n)
qLearnOpts = {'gamma': 0.9,
              'alpha': 0.1,
              'epsilon': 0.1,
              'numTraining': 5000,
              'actionFn': actionFn
              }

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = env.action_space.n  # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]

num_actions = env.action_space.n  # (left, right)
num_states = tuple2int(NUM_BUCKETS)
max_games = 500
init_epsilon = 1.
end_epsilon = 0.1
test_period = 0.4
epsilon_step = (init_epsilon - end_epsilon) / (max_games * test_period)
reps = 10

reward_results = np.zeros((reps, max_games))
steps_results = np.zeros((reps, max_games))
end_state_results = np.zeros((reps, max_games))
q_table = np.zeros((reps, num_states, num_actions))
policy = np.zeros((reps, num_states))

for rep in range(reps):

    agent = QLearningAgent(**qLearnOpts)
    agent.setEpsilon(init_epsilon)

    print("Rep: {} =========================================================").format(rep)

    for i in range(10):
        s_t = env.reset()

        # RmsAlg(rthres, influence, risk_default)
        alg = RmsAlg(rthres=-1, influence=1, risk_default=0)
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

            action_idx = agent.getAction(state_to_bucket(s_t))
            # action_idx = input("Action: ")
            obs, r, done, _ = env.step(action_idx)
            # env.render()

            if state_to_bucket(s_t) in [10, 42]:
                slope_r = r
            else:
                slope_r = r
            alg.update(s=state_to_bucket(s_t), r=slope_r, sprime=state_to_bucket(obs), sprime_features=obs)

            # alg.update(s=state_to_bucket(s_t), r=r, sprime=state_to_bucket(obs), sprime_features=obs)

            risk_penalty = alg.get_risk(state_to_bucket(obs))
            # print(r, risk_penalty)

            # reward_signal = r + risk_penalty
            reward_signal = r
            agent.observeTransition(state_to_bucket(s_t), action_idx, state_to_bucket(obs), reward_signal)
            misc['step_seq'].append(state_to_bucket(s_t))

            # print("=", state_to_bucket(s_t), action_idx, state_to_bucket(obs), reward_signal)

            # env.render()
            # print("risk_    dict", alg.get_risk_dict_no_zeros())

            prev_misc = misc
            s_t = obs
            t += 1
            r_t += r

        if GAME < max_games * test_period:
            agent.setEpsilon(agent.epsilon - epsilon_step)

        if done:
            done = False
            s_t = env.reset()
            alg.add_to_v(state_to_bucket(s_t), s_t)
            agent.stopEpisode()
            agent.startEpisode()

            reward_results[rep][GAME] = r_t
            steps_results[rep][GAME] = t
            end_state_results[rep][GAME] = misc['step_seq'][-1]

            print("total reward: " + str(reward_results[rep][GAME]))

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
