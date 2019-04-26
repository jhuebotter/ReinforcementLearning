''' Assignment: Planning & Reinforcement Learning 1
    By: Justus Hubotter, Florence van der Voort, Stefan Wijtsma
    Created @ April 2019'''


import numpy as np

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
    R, Q, V, states, actions = initialize()
    user_input = input("What method would you like to use? Choose from: random policy iteration, value iteration, manual. USER INPUT: ")

    if user_input.lower() == "random":
        random_policy(R, Q, V, states, actions)
    elif user_input.lower() == "value iteration":
        value_iteration(R, Q, V, states, actions)
    elif user_input.lower() == "manual":
        manual(R, Q, V, states, actions)
    else:
        pass


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
    done = False
    while (not done):
        newQ = np.zeros(Q.shape)
        for state in [s for s in states if s not in TERMINAL]:
            for action in actions:
                for j in getTransitionChances(state, action):
                    next_state = j[0]
                    prob = j[1]
                    newQ[states.index(state)][action] += prob * (
                                R[next_state] + DISCOUNT * np.random.choice(Q[states.index(next_state)]))
            V[state] = np.max(newQ[states.index(state)])

        if (Q == newQ).all():
            done = True
        Q = newQ.copy()

        print(V)


def value_iteration(R, Q, V, states, actions):
    done = False
    i = 0
    while (not done):
        newQ = np.zeros(Q.shape)
        for state in [s for s in states if s not in TERMINAL]:
            for action in actions:
                for j in getTransitionChances(state, action):
                    next_state = j[0]
                    prob = j[1]
                    newQ[states.index(state)][action] += prob * (
                                R[next_state] + DISCOUNT * max(Q[states.index(next_state)]))
            V[state] = np.max(newQ[states.index(state)])

        if (Q == newQ).all():
            done = True
        Q = newQ.copy()
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