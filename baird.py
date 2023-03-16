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

# set seed for reproducibility
SEED = 0
np.random.seed(SEED)

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
# @state: current state
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

# Semi-gradient DP
# @theta: weight for each component of the feature vector
# @alpha: step size
def semi_gradient_DP(theta, alpha):
    delta = 0.0
    # go through all the states
    for state in STATES:
        expected_return = 0.0
        # compute bellman error for each state
        for next_state in STATES:
            if next_state == LOWER_STATE:
                expected_return += REWARD + DISCOUNT * np.dot(theta, FEATURES[next_state, :])
        bellmanError = expected_return - np.dot(theta, FEATURES[state, :])
        # accumulate gradients
        delta += bellmanError * FEATURES[state, :]
    # derivatives happen to be the same matrix due to the linearity
    theta += alpha / len(STATES) * delta

# temporal difference with gradient correction
# @state: current state
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

# expected temporal difference with gradient correction
# @theta: weight of each component of the feature vector
# @weight: auxiliary trace for gradient correction
# @alpha: step size of @theta
# @beta: step size of @weight
def expected_TDC(theta, weight, alpha, beta):
    for state in STATES:
        # When computing expected update target, if next state is not lower state, importance ratio will be 0,
        # so we can safely ignore this case and assume next state is always lower state
        delta = REWARD + DISCOUNT * np.dot(FEATURES[LOWER_STATE, :], theta) - np.dot(FEATURES[state, :], theta)
        rho = 1 / BEHAVIOR_SOLID_PROBABILITY
        # Under behavior policy, state distribution is uniform, so the probability for each state is 1.0 / len(STATES)
        expected_update_theta = 1.0 / len(STATES) * BEHAVIOR_SOLID_PROBABILITY * rho * (
            delta * FEATURES[state, :] - DISCOUNT * FEATURES[LOWER_STATE, :] * np.dot(weight, FEATURES[state, :]))
        theta += alpha * expected_update_theta
        expected_update_weight = 1.0 / len(STATES) * BEHAVIOR_SOLID_PROBABILITY * rho * (
            delta - np.dot(weight, FEATURES[state, :])) * FEATURES[state, :]
        weight += beta * expected_update_weight

    # if *accumulate* expected update and actually apply update here, then it's synchronous
    # theta += alpha * expectedUpdateTheta
    # weight += beta * expectedUpdateWeight

# interest is 1 for every state
INTEREST = 1

# expected update of ETD
# @theta: weight of each component of the feature vector
# @emphasis: current emphasis
# @alpha: step size of @theta
# @return: expected next emphasis
def expected_emphatic_TD(theta, emphasis, alpha):
    # we perform synchronous update for both theta and emphasis
    expected_update = 0
    expected_next_emphasis = 0.0
    # go through all the states
    for state in STATES:
        # compute rho(t-1)
        if state == LOWER_STATE:
            rho = 1.0 / BEHAVIOR_SOLID_PROBABILITY
        else:
            rho = 0
        # update emphasis
        next_emphasis = DISCOUNT * rho * emphasis + INTEREST
        expected_next_emphasis += next_emphasis
        # When computing expected update target, if next state is not lower state, importance ratio will be 0,
        # so we can safely ignore this case and assume next state is always lower state
        next_state = LOWER_STATE
        delta = REWARD + DISCOUNT * np.dot(FEATURES[next_state, :], theta) - np.dot(FEATURES[state, :], theta)
        expected_update += 1.0 / len(STATES) * BEHAVIOR_SOLID_PROBABILITY * next_emphasis * 1 / BEHAVIOR_SOLID_PROBABILITY * delta * FEATURES[state, :]
    theta += alpha * expected_update
    return expected_next_emphasis / len(STATES)

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
def experiments():
    # Initialize the theta
    theta = np.ones(FEATURE_SIZE)
    theta[6] = 10

    theta_TD = copy.deepcopy(theta)
    theta_TDC = copy.deepcopy(theta)
    weight_TDC = np.zeros(FEATURE_SIZE)
    theta_DGTD = copy.deepcopy(theta)

    alpha_TD = 0.01
    alpha_grad = 0.005
    beta_grad = 0.05

    steps = 1000
    DGTD_step = 0
    logs = pd.DataFrame(columns=[
        'TD_RMSVE', 'TDC_RMSVE', 'DGTD_RMSVE', 
        'TD_RMSPBE', 'TDC_RMSPBE', 'DGTD_RMSPBE'
    ])
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
        TDC(state, action, next_state, theta_TDC, weight_TDC, alpha_grad, beta_grad)
        while step >= DGTD_step + f_gap(DGTD_step):
            direct_GTD(history[2*DGTD_step], history[2*DGTD_step+1], history[2*DGTD_step+2],
                       history[2*(DGTD_step+f_gap(DGTD_step))], history[2*(DGTD_step+f_gap(DGTD_step))+1],
                       history[2*(DGTD_step+f_gap(DGTD_step))+2], theta_DGTD, alpha_grad)
            DGTD_step+=1
        state = next_state
        # Log corresponding errors for each algorithm
        log = {
            'TD_RMSVE': compute_RMSVE(theta_TD), 
            'TD_RMSPBE': compute_RMSPBE(theta_TD), 
            'TDC_RMSVE': compute_RMSVE(theta_TDC), 
            'TDC_RMSPBE': compute_RMSPBE(theta_TDC), 
            'DGTD_RMSVE': compute_RMSVE(theta_DGTD), 
            'DGTD_RMSPBE': compute_RMSPBE(theta_DGTD),
        }
        logs = logs.append(log, ignore_index=True)
    logs.to_csv(f'./logs_approximated_seed[{SEED}].csv')

if __name__ == '__main__':
    experiments()