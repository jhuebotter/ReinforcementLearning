''' Assignment: Planning & Reinforcement Learning 1
    By: Justus Hübotter, Florence van der Voort, Stefan Wijtsma
    Created @ April 2019'''

import numpy as np
import sys

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
        user_input = input("What method would you like to use? Choose from: <random>, <value iteration>, <(simple) policy iteration>, <manual> or <exit>. USER INPUT: ")

        if user_input.lower() == "random":
            random_policy(R, Q, V, states, actions)
        elif user_input.lower() == "value iteration":
            value_iteration(R, Q, V, states, actions)
        elif user_input.lower() == "policy iteration":
            policy_iteration(R, Q, V, states, actions)
        elif user_input.lower() == "simple policy iteration":
            simple_policy_iteration(R, Q, V, states, actions)
        elif user_input.lower() == "manual":
            manual(R, Q, V, states, actions)
        elif user_input.lower() == "exit":
            sys.exit()
        else:
            print("Invalid input. Please choose between <random>, <value iteration> or <manual>.")
            continue


def initialize(init=0):
    R = np.zeros((4, 4))

    V = np.zeros((4, 4))

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

    Q = np.ones((len(states), len(actions)))
    Q *= init

    return R, Q, V, states, actions


def random_policy(R, Q, V, states, actions):
    print('State values after initialization:')
    print(V)

    done = False
    pi = np.ones(Q.shape)
    pi *= 0.25

    i = 0
    while (not done):
        newQ = np.zeros(Q.shape)
        for state in [s for s in states if s not in TERMINAL]:
            print()
            print('state:', state)
            for a in actions:
                print()
                print('action:', a)
                for (next_state, p) in getTransitionChances(state, a):
                    print('potential next state:', next_state)
                    print('probability of getting there:', p)
                    inter = pi[states.index(state)][a] * p * (R[next_state] + (DISCOUNT * V[next_state]))
                    print('potential value:', inter)
                    newQ[states.index(state)][a] += inter
                print('total estimated value for action in state:', newQ[states.index(state)][a])

        for state in states:
            V[state] = np.sum(newQ[states.index(state)])

        if (Q == newQ).all():
            done = True
        Q = newQ.copy()
        i += 1
        print()
        print('State values after iteration %i:' % i)
        print(Q)
        print(V)
    print('Random policy iteration converged')

    return V, Q, i


def policy_iteration(R, Q, V, states, actions):
    print('State values after initialization:')
    print(V)

    done = False
    i = 0
    while (not done):
        newQ = np.zeros(Q.shape)
        for state in [s for s in states if s not in TERMINAL]:
            # here we evaluate the policy
            newQ[states.index(state)] = [sum([p * (R[next_state] + DISCOUNT * V[next_state]) for (next_state, p) in getTransitionChances(state, a)]) for a in actions]

        for state in states: 
            # here we improve the policy by being greedy over all actions possible at each state
            V[state] = np.max(newQ[states.index(state)])

        if (Q == newQ).all():
            done = True
        Q = newQ.copy()
        i += 1
        print()
        print('State values after iteration %i:' % i)
        print(V)
    print('Policy iteration converged')


def policy_iteration(R, Q, V, states, actions):
    print('State values after initialization:')
    print(V)

    done = False
    i = 0
    while (not done):
        newQ = np.zeros(Q.shape)
        for state in [s for s in states if s not in TERMINAL]:
            # here we evaluate the policy
            newQ[states.index(state)] = [sum([p * (R[next_state] + DISCOUNT * V[next_state]) for (next_state, p) in getTransitionChances(state, a)]) for a in actions]

        for state in states: 
            # here we improve the policy by being greedy over all actions possible at each state
            V[state] = np.max(newQ[states.index(state)])

        if (Q == newQ).all():
            done = True
        Q = newQ.copy()
        i += 1
        print()
        print('State values after iteration %i:' % i)
        print(V)
    print('Policy iteration converged')


def simple_policy_iteration(R, Q, V, states, actions):
    print('State values after initialization:')
    print(V)

    done = False
    i = 0
    newQ = np.zeros(Q.shape)
    while (not done):
        for state in [s for s in states if s not in TERMINAL]:
            # here we evaluate the policy
            newQ[states.index(state)] = [sum([p * (R[next_state] + DISCOUNT * V[next_state]) for (next_state, p) in getTransitionChances(state, a)]) for a in actions]
            if (newQ[states.index(state)] != Q[states.index(state)]).any():
                i += 1
                break
        for state in states: 
            # here we improve the policy by being greedy over all actions possible at each state
            V[state] = np.max(newQ[states.index(state)])

        if (Q == newQ).all():
            done = True
        Q = newQ.copy()
        #i += 1
        print()
        print('State values after iteration %i:' % i)
        print(V)
    print('Policy iteration converged')


def value_iteration(R, Q, V, states, actions):
    print('State values after initialization:')
    print(V)

    done = False
    i = 0
    while (not done):
        newV = np.zeros(V.shape)
        for state in [s for s in states if s not in TERMINAL]:
            newV[state] = max([ sum([p * (R[next_state] + DISCOUNT * V[next_state]) for (next_state, p) in getTransitionChances(state, a)]) for a in actions])
        if (V == newV).all():
            done = True
        V = newV.copy()
        i += 1
        print()
        print('State values after iteration %i:' % i)
        print(V)
    print('Value iteration converged')


def manual(R, Q, V, states, actions):
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


def isConverged(Q, newQ):
    return (Q == newQ).all()


def isGameOver(state):
    return state in TERMINAL


if __name__ == '__main__':
    main()