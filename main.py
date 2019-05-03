''' Assignment: Planning & Reinforcement Learning Assignment 1
    By: Justus Hübotter (2617135), Florence van der Voort (2652198), Stefan Wijtsma (2575874)
    Created @ April 2019'''

import numpy as np
import sys
import time

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
CRACK = [(1, 1), (1, 3), (2, 3), (3, 1), (3, 2), (3, 3)]
SHIP = (2, 2)
GOAL = (0, 3)
START = (3, 0)
SLIP = 0.05
DISCOUNT = 0.9
EPSILON = 1
TERMINAL = [x for x in CRACK]
TERMINAL.append(GOAL)


def main():

    while True:
        R, Q, V, states, actions = initialize(init=0)
        user_input = input("What method would you like to use? Choose from: <random policy>, <value iteration>, <policy iteration>, <simple policy iteration>, <manual> or <exit>.  \n"
                           "USER INPUT: ")

        if user_input.lower() == "random policy":
            random_policy(R, Q, V, states, actions)
        elif user_input.lower() == "value iteration":
            value_iteration(R, Q, V, states, actions)
        elif user_input.lower() == "policy iteration":
            policy_iteration(R, Q, V, states, actions)
        elif user_input.lower() == "backup policy iteration":
            backup_policy_iteration(R, Q, V, states, actions)
        elif user_input.lower() == "simple policy iteration":
            simple_policy_iteration(R, Q, V, states, actions)
        elif user_input.lower() == "manual":
            manual(R)
        elif user_input.lower() == "exit":
            sys.exit()
        else:
            print("Invalid input. \n "
                  "Please choose between: <random policy>, <value iteration>, <policy iteration>, <simple policy iteration>, <manual> or <exit>.")
            continue


def initialize(init=0):
    R = np.zeros((4, 4))

    states = []

    actions = [UP, RIGHT, DOWN, LEFT]

    for i in range(4):
        for j in range(4):
            states.append((i, j))
            if (i, j) in CRACK:
                R[i, j] = -10
            elif (i, j) == SHIP:
                R[i, j] = 20
            elif (i, j) == GOAL:
                R[i, j] = 100
            else:
                R[i, j] = 0

    V = np.ones((4, 4))
    V *= init
    Q = np.ones((len(states), len(actions)))
    Q *= init

    return R, Q, V, states, actions


def random_policy(R, Q, V, states, actions, gamma=DISCOUNT, theta=float(1e-3), debug=False):
    print('Starting random policy iteration')
    print('Discount factor:', gamma)
    print('State values after initialization:')
    print(V)

    deltas = []
    t0 = time.time_ns()
    pi = np.ones(Q.shape)
    pi *= 0.25
    i = 0
    while True:
        delta = 0
        newQ = np.zeros(Q.shape)
        for state in [s for s in states if s not in TERMINAL]:
            for a in actions:
                for (next_state, p) in getTransitionChances(state, a):
                    newQ[states.index(state)][a] += pi[states.index(state)][a] * p * (R[next_state] + (gamma * V[next_state]))

        for state in states:
            v = V[state].copy()
            V[state] = np.sum(newQ[states.index(state)])
            delta = max(delta, abs(v - V[state]))

        if debug: print('Delta:', delta)
        Q = newQ.copy()
        i += 1
        deltas.append(delta)
        if delta < theta:
            break

    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    print()
    print('Random policy iteration converged')
    print('Runtime in ms: ', runtime)
    print('State values after iteration %i:' % i)
    print(V)
    

    return deltas


def policy_iteration(R, Q, V, states, actions, gamma=DISCOUNT, theta=float(1e-3)):
    print('Starting policy iteration')
    print('Discount factor:', gamma)
    print('State values after initialization:')
    print(V)

    deltas = []
    t0 = time.time_ns()
    i = 0
    pi = np.argmax(Q, axis=1).copy()
    while True:
        while True:
            # here we evaluate the policy
            delta = 0.
            for state in [s for s in states if s not in TERMINAL]:
                v = V[state].copy()
                a = pi[states.index(state)]
                V[state] = sum([p * (R[next_state] + gamma * V[next_state]) for (next_state, p) in getTransitionChances(state, a)])
                delta = max(delta, abs(v - V[state]))
            deltas.append(delta)
            if delta < theta:
                break

        policy_stable = True
        # here we improve the policy by being greedy over all actions possible at each state
        for state in [s for s in states if s not in TERMINAL]:
            b = pi[states.index(state)].copy()
            pi[states.index(state)] = np.argmax([sum([p * (R[next_state] + gamma * V[next_state]) for (next_state, p) in getTransitionChances(state, a)]) for a in actions])
            if b != pi[states.index(state)]:
                policy_stable = False

        i += 1
        print()
        print('State values after iteration %i:' % i)
        print(V)
        if policy_stable:
            break
    
    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    print()
    print('Policy iteration converged')
    print('Runtime in ms: ', runtime)
    print('Final tate values after iteration %i:' % i)
    print(V)

    return deltas


def simple_policy_iteration(R, Q, V, states, actions, gamma=DISCOUNT, theta=float(1e-3)):
    print('Starting simple policy iteration')
    print('Discount factor:', gamma)
    print('State values after initialization:')
    print(V)

    deltas = []
    t0 = time.time_ns()
    i = 0
    pi = np.argmax(Q, axis=1).copy()
    while True:
        while True:
            # here we evaluate the policy
            delta = 0.
            for state in [s for s in states if s not in TERMINAL]:
                v = V[state].copy()
                a = pi[states.index(state)]
                V[state] = sum([p * (R[next_state] + gamma * V[next_state]) for (next_state, p) in getTransitionChances(state, a)])
                delta = max(delta, abs(v - V[state]))
            deltas.append(delta)
            if delta < theta:
                break

        policy_stable = True
        # here we improve the policy by being greedy over all actions possible at each state
        print('improve') 
        for state in [s for s in states if s not in TERMINAL]:
            b = pi[states.index(state)].copy()
            pi[states.index(state)] = np.argmax([sum([p * (R[next_state] + gamma * V[next_state]) for (next_state, p) in getTransitionChances(state, a)]) for a in actions])
            if b != pi[states.index(state)]:
                policy_stable = False
                break

        i += 1
        print()
        print('State values after iteration %i:' % i)
        print(V)
        if policy_stable:
            break
    
    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    print()
    print('Simple policy iteration converged')
    print('Runtime in ms: ', runtime)
    print('Final state values after iteration %i:' % i)
    print(V)


    return deltas


def value_iteration(R, Q, V, states, actions, gamma=DISCOUNT, theta=float(1e-3)):
    print('Starting value iteration')
    print('Discount factor:', gamma)
    print('State values after initialization:')
    print(V)

    deltas = []
    t0 = time.time_ns()
    i = 0
    while True:
        delta = 0
        newV = np.zeros(V.shape)
        for state in [s for s in states if s not in TERMINAL]:
            v = V[state]
            newV[state] = max([ sum([p * (R[next_state] + gamma * V[next_state]) for (next_state, p) in getTransitionChances(state, a)]) for a in actions])
            delta = max(delta, abs(v - newV[state]))

        V = newV.copy()
        i += 1
        deltas.append(delta)
        if delta < theta:
            break
    
    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    print()
    print('Value iteration converged')
    print('Runtime in ms: ', runtime)
    print('State values after iteration %i:' % i)
    print(V)

    return deltas
    

def manual(R):

    state = START
    G = 0
    print("You can now control the robot manually by typing <up>, <down>, <left> or <right>.")
    print("Starting position:", state)
    while (not isGameOver(state)):
        print()

        user_input = input("Where would you like to move?: ")
        if user_input.lower() == "up":
            action = UP
        elif user_input.lower() == "down":
            action = DOWN
        elif user_input.lower() == "left":
            action = LEFT
        elif user_input.lower() == "right":
            action = RIGHT
        else:
            print("Invalid input. Please choose between <up>, <down>, <left> or <right>.")
            continue
        state = getNextState(state, action, debug=True)
        G += R[state]
        print("Reward gained: ", R[state])
    print("The game has ended. You have collected a total of %i reward!" % G)


def getTransitionChances(pos, action, debug=False):
    if debug: print('Starting position:', pos)
    chances = []

    if action == UP:
        if debug: print('Going up.')
        if pos[0] > 0:
            new_pos = (pos[0] - 1, pos[1])
            chances.append((new_pos, 1 - SLIP))
            while (not isGameOver(new_pos)) and new_pos[0] != 0:
                new_pos = (new_pos[0] - 1, new_pos[1])
            chances.append((new_pos, SLIP))
        else:
            if debug: print('Can not move!')
            chances.append((pos, 1.))

    if action == DOWN:
        if debug: print('Going down.')
        if pos[0] < 3:
            new_pos = (pos[0] + 1, pos[1])
            chances.append((new_pos, 1 - SLIP))
            while (not isGameOver(new_pos)) and new_pos[0] != 3:
                new_pos = (new_pos[0] + 1, new_pos[1])
            chances.append((new_pos, SLIP))
        else:
            if debug: print('Can not move!')
            chances.append((pos, 1.))

    if action == LEFT:
        if debug: print('Going left.')
        if pos[1] > 0:
            new_pos = (pos[0], pos[1] - 1)
            chances.append((new_pos, 1 - SLIP))
            while (not isGameOver(new_pos)) and new_pos[1] != 0:
                new_pos = (new_pos[0], new_pos[1] - 1)
                chances.append((new_pos, SLIP))
        else:
            if debug: print('Can not move!')
            chances.append((pos, 1.))

    if action == RIGHT:
        if debug: print('Going right.')
        if pos[1] < 3:
            new_pos = (pos[0], pos[1] + 1)
            chances.append((new_pos, 1 - SLIP))
            while (not isGameOver(new_pos)) and new_pos[1] != 3:
                new_pos = (new_pos[0], new_pos[1] + 1)
            chances.append((new_pos, SLIP))
        else:
            if debug: print('Can not move!')
            chances.append((pos, 1.))

    if len(chances) > 1:
        if chances[0][0] == chances[1][0]:
            chances = [(new_pos, 1.)]

    return chances


def getNextState(pos, action, debug=True):
    if debug: print('Starting position:', pos)
    if action == UP:
        if debug: print('Going up.')
        if pos[0] > 0:
            new_pos = (pos[0] - 1, pos[1])
            if np.random.binomial(1, SLIP):
                while (not isGameOver(new_pos)) and new_pos[0] != 0:
                    new_pos = (new_pos[0] - 1, new_pos[1])
                if debug: print('Slipped!')
        else:
            new_pos = pos
            if debug: print('Can not move!')
    if action == DOWN:
        if debug: print('Going down')
        if pos[0] < 3:
            new_pos = (pos[0] + 1, pos[1])
            if np.random.binomial(1, SLIP):
                while (not isGameOver(new_pos)) and new_pos[0] != 3:
                    new_pos = (new_pos[0] + 1, new_pos[1])
                if debug: print('Slipped!')
        else:
            new_pos = pos
            if debug: print('Can not move!')
    if action == LEFT:
        if debug: print('Going left.')
        if pos[1] > 0:
            new_pos = (pos[0], pos[1] - 1)
            if np.random.binomial(1, SLIP):
                while (not isGameOver(new_pos)) and new_pos[1] != 0:
                    new_pos = (new_pos[0], new_pos[1] - 1)
                if debug: print('Slipped!')
        else:
            new_pos = pos
            if debug: print('Can not move!')
    if action == RIGHT:
        if debug: print('Going right.')
        if pos[1] < 3:
            new_pos = (pos[0], pos[1] + 1)
            if np.random.binomial(1, SLIP):
                while (not isGameOver(new_pos)) and new_pos[1] != 3:
                    new_pos = (new_pos[0], new_pos[1] + 1)
                if debug: print('Slipped!')
        else:
            new_pos = pos
            if debug: print('Can not move!')

    if debug: print('New position:', new_pos)

    return new_pos


'''
def backup(R, Q, V, states, actions, gamma, state):
    x = np.zeros(len(actions))
    for a in actions:
        for next_state, p in getTransitionChances(state, a):
            x[a] += p * (R[next_state] + gamma * V[next_state])
    return x
'''

def isGameOver(state):
    return state in TERMINAL


if __name__ == '__main__':
    main()