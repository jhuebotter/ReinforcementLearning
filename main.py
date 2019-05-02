''' Assignment: Planning & Reinforcement Learning Assignment 1
    By: Justus Hübotter, Florence van der Voort, Stefan Wijtsma
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
        user_input = input("What method would you like to use? Choose from: <random policy>, <value iteration>, <policy iteration>, <simple policy iteration>, <manual> or <exit>. USER INPUT: ")

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
            print("Invalid input. Please choose between: <random policy>, <value iteration>, <policy iteration>, <simple policy iteration>, <manual> or <exit>.")
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

        if debug: print('delta:', delta)
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
    while True:
        delta = 0
        newQ = np.zeros(Q.shape)
        for state in [s for s in states if s not in TERMINAL]:
            # here we evaluate the policy
            #backup(R, Q, V, states, actions, gamma, state)
            newQ[states.index(state)] = [sum([p * (R[next_state] + gamma * V[next_state]) for (next_state, p) in getTransitionChances(state, a)]) for a in actions]
            #newQ[states.index(state)] = backup(R, Q, V, states, actions, gamma, state)

        for state in states: 
            # here we improve the policy by being greedy over all actions possible at each state
            v = V[state].copy()
            V[state] = np.max(newQ[states.index(state)])
            delta = max(delta, abs(v - V[state]))

        Q = newQ.copy()
        i += 1
        deltas.append(delta)
        if delta < theta:
            break
    
    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    print()
    print('Policy iteration converged')
    print('Runtime in ms: ', runtime)
    print('State values after iteration %i:' % i)
    print(V)

    return deltas


def backup_policy_iteration(R, Q, V, states, actions, gamma=DISCOUNT, theta=float(1e-3)):
    print('Starting policy iteration')
    print('Discount factor:', gamma)
    print('State values after initialization:')
    print(V)

    t0 = time.time_ns()
    i = 0
    while True:
        newQ = np.zeros(Q.shape)
        for state in [s for s in states if s not in TERMINAL]:
            for a in actions:
                for (next_state, p) in getTransitionChances(state, a):
                    for a2 in actions:
                        for (next_state2, p2) in getTransitionChances(next_state, a2):
                            # here we evaluate the policy
                            newQ[states.index(state)][a] += p * (R[next_state] + gamma * max([sum([p2 * (R[next_state2] + gamma**2 * V[next_state2])])]))

        for state in states: 
            # here we improve the policy by being greedy over all actions possible at each state
            v = V[state].copy()
            V[state] = np.max(newQ[states.index(state)])
            delta = max(delta, abs(v - V[state]))

        Q = newQ.copy()
        i += 1
        if delta < theta:
            break
    
    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    print()
    print('Policy iteration converged')
    print('Runtime in ms: ', runtime)
    print('State values after iteration %i:' % i)
    print(V)


def backup(R, Q, V, states, actions, gamma, state):
    x = np.zeros(len(actions))
    for a in actions:
        for next_state, p in getTransitionChances(state, a):
            x[a] += p * (R[next_state] + gamma * V[next_state])
    return x


def simple_policy_iteration(R, Q, V, states, actions, gamma=DISCOUNT, theta=float(1e-3)):
    print('Starting simple policy iteration')
    print('Discount factor:', gamma)
    print('State values after initialization:')
    print(V)

    deltas = []
    t0 = time.time_ns()
    newQ = np.zeros(Q.shape)
    i = 0
    while True:
        delta = 0
        for state in [s for s in states if s not in TERMINAL]:
            for a in actions:
                #q = Q[states.index(state)][a].copy()
                # here we evaluate the policy
                newQ[states.index(state)][a] = sum([p * (R[next_state] + gamma * V[next_state]) for (next_state, p) in getTransitionChances(state, a)])
                #delta = max(delta, abs(q - newQ[states.index(state)][a]))
            delta = max(delta, abs(max(newQ[states.index(state)])))
            if max(newQ[states.index(state)]) == max(Q[states.index(state)]):
                pass
            else:
                deltas.append(delta)
                i += 1
                break

        for state in states: 
            # here we improve the policy by being greedy over all actions possible at each state
            V[state] = np.max(newQ[states.index(state)])

        Q = newQ.copy()

        if delta <= theta:
            break

    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    print()
    print('Simple policy iteration converged')
    print('Runtime in ms: ', runtime)
    print('State values after iteration %i:' % i)
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
        if debug: print('going up')
        if pos[0] > 0:
            new_pos = (pos[0] - 1, pos[1])
            chances.append((new_pos, 1 - SLIP))
            while (not isGameOver(new_pos)) and new_pos[0] != 0:
                new_pos = (new_pos[0] - 1, new_pos[1])
            chances.append((new_pos, SLIP))
        else:
            if debug: print('cant move!')
            chances.append((pos, 1.))

    if action == DOWN:
        if debug: print('going down')
        if pos[0] < 3:
            new_pos = (pos[0] + 1, pos[1])
            chances.append((new_pos, 1 - SLIP))
            while (not isGameOver(new_pos)) and new_pos[0] != 3:
                new_pos = (new_pos[0] + 1, new_pos[1])
            chances.append((new_pos, SLIP))
        else:
            if debug: print('cant move!')
            chances.append((pos, 1.))

    if action == LEFT:
        if debug: print('going left')
        if pos[1] > 0:
            new_pos = (pos[0], pos[1] - 1)
            chances.append((new_pos, 1 - SLIP))
            while (not isGameOver(new_pos)) and new_pos[1] != 0:
                new_pos = (new_pos[0], new_pos[1] - 1)
                chances.append((new_pos, SLIP))
        else:
            if debug: print('cant move!')
            chances.append((pos, 1.))

    if action == RIGHT:
        if debug: print('going right')
        if pos[1] < 3:
            new_pos = (pos[0], pos[1] + 1)
            chances.append((new_pos, 1 - SLIP))
            while (not isGameOver(new_pos)) and new_pos[1] != 3:
                new_pos = (new_pos[0], new_pos[1] + 1)
            chances.append((new_pos, SLIP))
        else:
            if debug: print('cant move!')
            chances.append((pos, 1.))

    if len(chances) > 1:
        if chances[0][0] == chances[1][0]:
            chances = [(new_pos, 1.)]

    return chances


def getNextState(pos, action, debug=True):
    if debug: print('Starting position:', pos)
    if action == UP:
        if debug: print('going up')
        if pos[0] > 0:
            new_pos = (pos[0] - 1, pos[1])
            if np.random.binomial(1, SLIP):
                while (not isGameOver(new_pos)) and new_pos[0] != 0:
                    new_pos = (new_pos[0] - 1, new_pos[1])
                if debug: print('Slipped!')
        else:
            new_pos = pos
            if debug: print('cant move!')
    if action == DOWN:
        if debug: print('going down')
        if pos[0] < 3:
            new_pos = (pos[0] + 1, pos[1])
            if np.random.binomial(1, SLIP):
                while (not isGameOver(new_pos)) and new_pos[0] != 3:
                    new_pos = (new_pos[0] + 1, new_pos[1])
                if debug: print('Slipped!')
        else:
            new_pos = pos
            if debug: print('cant move!')
    if action == LEFT:
        if debug: print('going left')
        if pos[1] > 0:
            new_pos = (pos[0], pos[1] - 1)
            if np.random.binomial(1, SLIP):
                while (not isGameOver(new_pos)) and new_pos[1] != 0:
                    new_pos = (new_pos[0], new_pos[1] - 1)
                if debug: print('Slipped!')
        else:
            new_pos = pos
            if debug: print('cant move!')
    if action == RIGHT:
        if debug: print('going right')
        if pos[1] < 3:
            new_pos = (pos[0], pos[1] + 1)
            if np.random.binomial(1, SLIP):
                while (not isGameOver(new_pos)) and new_pos[1] != 3:
                    new_pos = (new_pos[0], new_pos[1] + 1)
                if debug: print('Slipped!')
        else:
            new_pos = pos
            if debug: print('cant move!')

    if debug: print('new position:', new_pos)

    return new_pos


def isConverged(Q, newQ, theta):
    return np.abs(np.sum(Q - newQ)) < theta


def isGameOver(state):
    return state in TERMINAL


if __name__ == '__main__':
    main()