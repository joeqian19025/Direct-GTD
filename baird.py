#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from tqdm import tqdm
import copy
import pandas as pd
import math

# all states: state 0-5 are upper states
STATES = np.arange(0, 7)
# state 6 is lower state
LOWER_STATE = 6
# discount factor
DISCOUNT = 0.99

# each state is represented by a vector of length 8
FEATURE_SIZE = 8
FEATURES = np.zeros((len(STATES), FEATURE_SIZE))
for i in range(LOWER_STATE):
    FEATURES[i, i] = 2
    FEATURES[i, 7] = 1
FEATURES[LOWER_STATE, 6] = 1
FEATURES[LOWER_STATE, 7] = 2

# all possible actions
DASHED = 0
SOLID = 1
ACTIONS = [DASHED, SOLID]

# reward is always zero
REWARD = 0

# take @action at @state, return the new state
def transition(state, action):
    if action == SOLID:
        return LOWER_STATE
    return np.random.choice(STATES[: LOWER_STATE])

# target policy
def target_policy(state):
    return SOLID

# state distribution for the behavior policy
STATE_DISTRIBUTION = np.ones(len(STATES)) / 7
STATE_DISTRIBUTION_MAT = np.matrix(np.diag(STATE_DISTRIBUTION))
# projection matrix for minimize MSVE
PROJECTION_MAT = np.matrix(FEATURES) * \
                 np.linalg.pinv(np.matrix(FEATURES.T) * STATE_DISTRIBUTION_MAT * np.matrix(FEATURES)) * \
                 np.matrix(FEATURES.T) * \
                 STATE_DISTRIBUTION_MAT

# behavior policy
BEHAVIOR_SOLID_PROBABILITY = 1.0 / 7
def behavior_policy(state):
    if np.random.binomial(1, BEHAVIOR_SOLID_PROBABILITY) == 1:
        return SOLID
    return DASHED

# Semi-gradient off-policy temporal difference
# @state: state S_t
# @action: action A_t
# @next_state: state S_{t+1}
# @theta: weight for each component of the feature vector
# @alpha: step size
# @return: next state
def semi_gradient_off_policy_TD(state, action, next_state, theta, alpha):
    # get the importance ratio
    if action == DASHED:
        rho = 0.0
    else:
        rho = 1.0 / BEHAVIOR_SOLID_PROBABILITY
    delta = REWARD + DISCOUNT * np.dot(FEATURES[next_state, :], theta) - \
            np.dot(FEATURES[state, :], theta)
    delta *= rho * alpha
    # derivatives happen to be the same matrix due to the linearity
    theta += FEATURES[state, :] * delta

# temporal difference with gradient correction
# @state: state S_t
# @action: action A_t
# @next_state: state S_{t+1}
# @theta: weight of each component of the feature vector
# @weight: auxiliary trace for gradient correction
# @alpha: step size of @theta
# @beta: step size of @weight
def TDC(state, action, next_state, theta, weight, alpha, beta):
    # get the importance ratio
    if action == DASHED:
        rho = 0.0
    else:
        rho = 1.0 / BEHAVIOR_SOLID_PROBABILITY
    delta = REWARD + DISCOUNT * np.dot(FEATURES[next_state, :], theta) - \
            np.dot(FEATURES[state, :], theta)
    theta += alpha * rho * (delta * FEATURES[state, :] - DISCOUNT * FEATURES[next_state, :] * np.dot(FEATURES[state, :], weight))
    weight += beta * rho * (delta - np.dot(FEATURES[state, :], weight)) * FEATURES[state, :]

# direct gradient temporal difference
# @state1: state S_t
# @action1: action A_t 
# @next_state1: state S_{t+1}
# @state2: state S_{t+f(t)}
# @action2: action A_{t+f(t)}
# @next_state2: state 
# @theta: weight of each component of the feature vector
# @alpha: step size of @theta
def direct_GTD(state1, action1, next_state1, state2, action2, next_state2, theta, alpha):
    # get the importance ratios
    if action1 == DASHED:
        rho1 = 0.0
    else:
        rho1 = 1.0 / BEHAVIOR_SOLID_PROBABILITY
    if action2 == DASHED:
        rho2 = 0.0
    else:
        rho2 = 1.0 / BEHAVIOR_SOLID_PROBABILITY
    theta += alpha * rho1 * rho2 * np.dot(FEATURES[state2, :], FEATURES[state1, :]) \
        * (REWARD + DISCOUNT * np.dot(FEATURES[next_state1, :], theta) - np.dot(FEATURES[state1, :], theta)) \
        * (FEATURES[state2, :] - DISCOUNT * FEATURES[next_state2, :])


# compute RMSVE for a value function parameterized by @theta
# true value function is always 0 in this example
def compute_RMSVE(theta):
    return np.sqrt(np.dot(np.power(np.dot(FEATURES, theta), 2), STATE_DISTRIBUTION))

# compute RMSPBE for a value function parameterized by @theta
# true value function is always 0 in this example
def compute_RMSPBE(theta):
    bellman_error = np.zeros(len(STATES))
    for state in STATES:
        for next_state in STATES:
            if next_state == LOWER_STATE:
                bellman_error[state] += REWARD + DISCOUNT * np.dot(theta, FEATURES[next_state, :]) - np.dot(theta, FEATURES[state, :])
    bellman_error = np.dot(np.asarray(PROJECTION_MAT), bellman_error)
    return np.sqrt(np.dot(np.power(bellman_error, 2), STATE_DISTRIBUTION))

f_gap = lambda x: math.ceil(math.log(x+1)**2)

# Experiment of semi-grad TD, TDC, Direct GTD.
def experiments(seed):
    np.random.seed(seed)
    # Initialize the theta
    theta = np.ones(FEATURE_SIZE)
    theta[6] = 10
    theta_TD = copy.deepcopy(theta)
    theta_TDC = copy.deepcopy(theta)
    weight_TDC = np.zeros(FEATURE_SIZE)
    theta_DGTD = copy.deepcopy(theta)

    # Stepsize configurations for each algorithm
    alpha_TD = 0.01
    alpha_TDC = 0.005
    beta_TDC = 0.05
    alpha_DGTD = 0.001

    steps = 50000
    DGTD_step = 0
    logs = pd.DataFrame()
    state = np.random.choice(STATES)
    history = [state]

    for step in tqdm(range(steps)):
        # sample action and next state and update history
        action = behavior_policy(state)
        next_state = transition(state, action)
        history.append(action)
        history.append(next_state)
        # update the weights with each algorithms respectively
        semi_gradient_off_policy_TD(state, action, next_state, theta_TD, alpha_TD)
        TDC(state, action, next_state, theta_TDC, weight_TDC, alpha_TDC, beta_TDC)
        while step >= DGTD_step + f_gap(DGTD_step):
            direct_GTD(history[2*DGTD_step], history[2*DGTD_step+1], history[2*DGTD_step+2],
                       history[2*(DGTD_step+f_gap(DGTD_step))], history[2*(DGTD_step+f_gap(DGTD_step))+1],
                       history[2*(DGTD_step+f_gap(DGTD_step))+2], theta_DGTD, alpha_DGTD)
            DGTD_step+=1
        # update the current state
        state = next_state
        # log corresponding errors for each algorithm
        log = {
            'TD_RMSVE': compute_RMSVE(theta_TD), 
            'TD_RMSPBE': compute_RMSPBE(theta_TD), 
            'TDC_RMSVE': compute_RMSVE(theta_TDC), 
            'TDC_RMSPBE': compute_RMSPBE(theta_TDC), 
            'DGTD_RMSVE': compute_RMSVE(theta_DGTD), 
            'DGTD_RMSPBE': compute_RMSPBE(theta_DGTD),
        }
        for i in range(FEATURE_SIZE):
            log[f"TD_theta{i}"] = theta_TD[i]
            log[f"TDC_theta{i}"] = theta_TDC[i]
            log[f"DGTD_theta{i}"] = theta_DGTD[i]
        logs = logs.append(log, ignore_index=True)
    # save the logs
    logs.to_csv(f'./logs/logs_seed[{seed}].csv')

if __name__ == '__main__':
    # Perform the experiment on with 10 different random seeds
    for seed in range(10):
        experiments(seed)