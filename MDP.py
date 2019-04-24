import numpy as np
import matplotlib.pyplot as plt
import world

np.random.seed(42)

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
actions = [UP, RIGHT, DOWN, LEFT]
states = []
crack = [(1,1),(1,3),(2,3),(3,1),(3,2),(3,3)]
ship = (2,2)
goal = (0,3)
start = (3,0)
slip = 0.05
discount = 0.9
epsilon = 0.1
decay = 0.0
init = -10
iterations = 1000
terminal = [x for x in crack]
terminal.append(goal)


def main():
    R, Q = initialize(init=init)

    rewards = []

    #env = world.World()

    e = epsilon

    for i in range(iterations):

        pos = start
        G = 0
        newQ = Q.copy()

        while (not isGameOver(pos)):
            
            if np.random.uniform() < e:
                action = np.random.choice(actions)
            else:
                possible = Q[states.index(pos)]
                best = np.argwhere(possible == np.max(possible)).flatten().tolist()
                action = np.random.choice(best)

            for j in getTransitionChances(pos, action):
                pot_pos = j[0]
                prob = j[1]
                newQ[states.index(pos)][action] += prob * (R[pot_pos] + discount * max(newQ[states.index(pot_pos)]) - newQ[states.index(pos)][action])

            pos = getNextState(pos, action)
            
            G += R[pos]

        if e-decay >= 0.:
            e -= decay

        print('finished iteration:', i)
        print('epsilon: %.2f' % e)
        print('cummulative reward:', G)
        print()
        rewards.append(G)

        if (abs(Q - newQ) == 0.).all():
            print('no update')
        else:
            print('update')
            print((Q - newQ))
        Q = newQ.copy()


    V = np.zeros((4,4))
    for state in states:
        V[state] = np.max(Q[states.index(state)])
    print()
    print('final Q table:\n', Q)
    print('final state values:\n', V)
    ma_rewards = np.convolve(rewards, np.ones((1,))/1,mode='valid')
    plt.plot(ma_rewards)
    plt.show()


def initialize(init=0):

    R = np.zeros((4,4))

    for i in range(4):
        for j in range(4):
            states.append((i,j))
            if (i,j) in crack:
                R[i,j] = -10
            elif (i,j) == ship:
                R[i,j] = 20
            elif (i,j) == goal:
                R[i,j] = 100
            else:
                R[i,j] = 0

    Q = np.ones((len(states),len(actions)))
    Q *= init

    return R, Q


def isGameOver(pos):
    return pos in terminal
        

def getNextState(pos, action, debug=True):
    if debug: print('starting position:', pos)
    if action == UP:
        if debug: print('going up')
        if pos[0] > 0:
            new_pos = (pos[0]-1, pos[1])
            if np.random.binomial(1, slip):
                while (not isGameOver(new_pos)) and new_pos[0] != 0:
                    new_pos = (new_pos[0]-1, new_pos[1])
                if debug: print('slipped!')
        else:
            new_pos = pos
            if debug: print('cant move!')
    if action == DOWN:
        if debug: print('going down')
        if pos[0] < 3:
            new_pos = (pos[0]+1, pos[1])
            if np.random.binomial(1, slip):
                while (not isGameOver(new_pos)) and new_pos[0] != 3:
                    new_pos = (new_pos[0]+1, new_pos[1])
                if debug: print('slipped!')
        else:
            new_pos = pos
            if debug: print('cant move!')
    if action == LEFT:
        if debug: print('going left')
        if pos[1] > 0:
            new_pos = (pos[0], pos[1]-1)
            if np.random.binomial(1, slip):
                while (not isGameOver(new_pos)) and new_pos[1] != 0:
                    new_pos = (new_pos[0], new_pos[1]-1)
                if debug: print('slipped!')
        else:
            new_pos = pos
            if debug: print('cant move!')
    if action == RIGHT:
        if debug: print('going right')
        if pos[1] < 3:
            new_pos = (pos[0], pos[1]+1)
            if np.random.binomial(1, slip):
                while (not isGameOver(new_pos)) and new_pos[1] != 3:
                    new_pos = (new_pos[0], new_pos[1]+1)
                if debug: print('slipped!')
        else:
            new_pos = pos
            if debug: print('cant move!')

    if debug: print('new position:', new_pos)

    return new_pos


def getTransitionChances(pos, action, debug=False):
    if debug: print('starting position:', pos)
    chances = []

    if action == UP:
        if debug: print('going up')
        if pos[0] > 0:
            new_pos = (pos[0]-1, pos[1])
            chances.append((new_pos, 1-slip))
            while (not isGameOver(new_pos)) and new_pos[0] != 0:
                new_pos = (new_pos[0]-1, new_pos[1])
            chances.append((new_pos, slip))
        else:
            if debug: print('cant move!')
            chances.append((pos, 1.))

    if action == DOWN:
        if debug: print('going down')
        if pos[0] < 3:
            new_pos = (pos[0]+1, pos[1])
            chances.append((new_pos, 1-slip))
            while (not isGameOver(new_pos)) and new_pos[0] != 3:
                new_pos = (new_pos[0]+1, new_pos[1])
            chances.append((new_pos, slip))
        else:
            if debug: print('cant move!')
            chances.append((pos, 1.))

    if action == LEFT:
        if debug: print('going left')
        if pos[1] > 0:
            new_pos = (pos[0], pos[1]-1)
            chances.append((new_pos, 1-slip))
            while (not isGameOver(new_pos)) and new_pos[1] != 0:
                new_pos = (new_pos[0], new_pos[1]-1)
                chances.append((new_pos, slip))
        else:
            if debug: print('cant move!')
            chances.append((pos, 1.))

    if action == RIGHT:
        if debug: print('going right')
        if pos[1] < 3:
            new_pos = (pos[0], pos[1]+1)
            chances.append((new_pos, 1-slip))
            while (not isGameOver(new_pos)) and new_pos[1] != 3:
                new_pos = (new_pos[0], new_pos[1]+1)
            chances.append((new_pos, slip))
        else:
            if debug: print('cant move!')
            chances.append((pos, 1.))

    if len(chances) > 1:
        if chances[0][0] == chances[1][0]:
            chances = [(new_pos, 1.)]

    return chances


if __name__ == '__main__':
    main()
