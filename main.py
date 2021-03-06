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
EPSILON = 0.05
LEARNING_RATE = 0.1
TEMPERATURE = 1.0
TERMINAL = [x for x in CRACK]
TERMINAL.append(GOAL)


def main():

    while True:
        R, Q, V, states, actions = initialize(init=0)
        #user_input = input("What method would you like to use? Choose from: <q learning>, <soft max>, <sarsa>, <random policy>, <value iteration>, <policy iteration>, <simple policy iteration>, <manual> or <exit>.  \n"
        user_input = input("What method would you like to use? Choose from: <Q Learning>, <Soft Max>, <SARSA>, <Double Q Learning>, <manual> or <exit>.  \n"
                           "USER INPUT: ").lower()

        if user_input == "random policy":
            random_policy(R, Q, V, states, actions)
        elif user_input == "value iteration":
            value_iteration(R, Q, V, states, actions)
        elif user_input == "policy iteration":
            policy_iteration(R, Q, V, states, actions)
        elif user_input == "simple policy iteration":
            simple_policy_iteration(R, Q, V, states, actions)
        elif user_input == "q learning":
            q_learning(R, Q, V, states, actions, verbose=True)
        elif user_input == "double q learning":
            double_q_learning(R, Q, V, states, actions, verbose=True)
        elif user_input == "eligibility traces":
            q_learning_et(R, Q, V, states, actions, verbose=True)
        elif user_input == "soft max":
            soft_max(R, Q, V, states, actions, verbose=True)
        elif user_input == "sarsa":
            sarsa(R, Q, V, states, actions, verbose=True)
        elif user_input == "manual":
            manual(R)
        elif user_input == "exit":
            sys.exit()
        else:
            print("Invalid input. \n")
            continue


def initialize(init=0):
    R = np.zeros((4, 4))

    states = []

    actions = [UP, RIGHT, DOWN, LEFT]

    V = np.ones((4, 4))
    V *= init

    for i in range(4):
        for j in range(4):
            states.append((i, j))
            if (i, j) in CRACK:
                R[i, j] = -10
                V[i, j] = 0
            elif (i, j) == SHIP:
                R[i, j] = 20
            elif (i, j) == GOAL:
                R[i, j] = 100
                V[i, j] = 0
            else:
                R[i, j] = 0

    Q = np.ones((len(states), len(actions)))
    Q *= init

    return R, Q, V, states, actions


def double_q_learning(R, Q, V, states, actions, epsilon=EPSILON,
    gamma=DISCOUNT, lr=LEARNING_RATE, n_episodes=1000, e_decay=1., verbose=False):
    if verbose:
        print('Starting double Q learning')
        print('Episodes: ', n_episodes)
        print('Discount factor:', gamma)
        print('Epsilon:', epsilon)
        print('Epsilon decay:', e_decay)
        print('Learning rate:', lr)

    t0 = time.time_ns()
    Gs = []
    Qa, Va = Q.copy(), V.copy()
    Qb, Vb = Q.copy(), V.copy()
    for e in np.arange(1, n_episodes+1):
        state = START
        G = 0
        while state not in TERMINAL:
            
            # This is epsilon-greedy
            if np.random.binomial(1, epsilon):
                # with chance of epsilon, pick a random action
                action = np.random.choice(actions)
            else:
                # otherwise pick a random action amongst the highest q value only
                Q = np.add(Qa, Qb)
                best = np.argwhere(Q[states.index(state)]==np.max(Q[states.index(state)])).flatten()
                action = np.random.choice(best)

            # get next state
            next_state = getNextState(state, action, debug=False)

            # get reward
            r = R[next_state]
            G = r + (G * gamma)

            # update q
            if np.random.binomial(1, 0.5):
                a_star = np.argmax(Qa[states.index(next_state)])
                Qa[states.index(state)][action] += lr * (r + gamma * Qb[states.index(next_state)][a_star] - Qa[states.index(state)][action])
            else:
                b_star = np.argmax(Qb[states.index(next_state)])
                Qb[states.index(state)][action] += lr * (r + gamma * Qa[states.index(next_state)][b_star] - Qb[states.index(state)][action])

            state = next_state

        # decay epsilon if needed
        epsilon *= e_decay

        Gs.append(G)

    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    if verbose:
        print()
        print('Double Q Learning completed')
        print('Mean Cummulative Reward:', np.mean(Gs))
        print('Runtime in ms: ', runtime)
        print('Qa:\n', Qa)
        print('Qb:\n', Qb)

        for state in states:
                Va[state] = np.max(Qa[states.index(state)])
                Vb[state] = np.max(Qb[states.index(state)])
        print('Va:\n', Va)
        print('Vb:\n', Vb)
        print()

    return Gs, runtime


def q_learning_er(R, Q, V, states, actions, epsilon=EPSILON,
    gamma=DISCOUNT, lr=LEARNING_RATE, n_episodes=1000, e_decay=1., verbose=False):
    
    # THIS FUNCTION IS NOT DONE AND WILL NOT YIELD THE EXPECTED OUTPUT!

    if verbose:
        print('Starting Q learning with experience replay')
        print('Episodes: ', n_episodes)
        print('Discount factor:', gamma)
        print('Epsilon:', epsilon)
        print('Epsilon decay:', e_decay)
        print('Learning rate:', lr)

    t0 = time.time_ns()
    Gs = []
    for e in np.arange(1, n_episodes+1):
        state = START
        G = 0
        while state not in TERMINAL:
            
            # This is epsilon-greedy
            if np.random.binomial(1, epsilon):
                # with chance of epsilon, pick a random action
                action = np.random.choice(actions)
            else:
                # otherwise pick a random action amongst the highest q value only
                best = np.argwhere(Q[states.index(state)]==np.max(Q[states.index(state)])).flatten()
                action = np.random.choice(best)

            # get next state
            next_state = getNextState(state, action, debug=False)

            # get reward
            r = R[next_state]
            G = r + (G * gamma)

            # update q
            Q[states.index(state)][action] += lr * (r + gamma * np.max(Q[states.index(next_state)]) - Q[states.index(state)][action])

            state = next_state

        # decay epsilon if needed
        epsilon *= e_decay

        Gs.append(G)

    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    if verbose:
        print()
        print('Q Learning with experience replay completed')
        print('Mean Cummulative Reward:', np.mean(Gs))
        print('Runtime in ms: ', runtime)
        print('Q:\n', Q)

        for state in states:
                V[state] = np.max(Q[states.index(state)])
        print('V:\n', V)
        print()

    return Gs, runtime


def q_learning(R, Q, V, states, actions, epsilon=EPSILON,
    gamma=DISCOUNT, lr=LEARNING_RATE, n_episodes=1000, e_decay=1., verbose=False):
    if verbose:
        print('Starting Q learning')
        print('Episodes: ', n_episodes)
        print('Discount factor:', gamma)
        print('Epsilon:', epsilon)
        print('Epsilon decay:', e_decay)
        print('Learning rate:', lr)

    t0 = time.time_ns()
    Gs = []
    for e in np.arange(1, n_episodes+1):
        state = START
        G = 0
        while state not in TERMINAL:
            
            # This is epsilon-greedy
            if np.random.binomial(1, epsilon):
                # with chance of epsilon, pick a random action
                action = np.random.choice(actions)
            else:
                # otherwise pick a random action amongst the highest q value only
                best = np.argwhere(Q[states.index(state)]==np.max(Q[states.index(state)])).flatten()
                action = np.random.choice(best)

            # get next state
            next_state = getNextState(state, action, debug=False)

            # get reward
            r = R[next_state]
            G = r + (G * gamma)

            # update q
            Q[states.index(state)][action] += lr * (r + gamma * np.max(Q[states.index(next_state)]) - Q[states.index(state)][action])

            state = next_state

        # decay epsilon if needed
        epsilon *= e_decay

        Gs.append(G)

    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    if verbose:
        print()
        print('Q Learning completed')
        print('Mean Cummulative Reward:', np.mean(Gs))
        print('Runtime in ms: ', runtime)
        print('Q:\n', Q)

        for state in states:
                V[state] = np.max(Q[states.index(state)])
        print('V:\n', V)
        print()

    return Gs, runtime


def softmax(x, t=1.):
    #Compute softmax values for each sets of scores in x.
    x = x.copy()
    x -= np.max(x)
    
    return np.exp(np.array(x)/(t + 1e-16)) / np.sum(np.exp(np.array(x)/(t + 1e-16)), axis=0)


def soft_max(R, Q, V, states, actions, temperature=TEMPERATURE,
    gamma=DISCOUNT, lr=LEARNING_RATE, n_episodes=1000, t_decay=1., verbose=False):
    if verbose:
        print('Starting Soft Max exploration')
        print('Episodes: ', n_episodes)
        print('Discount factor:', gamma)
        print('Temperature:', temperature)
        print('Temperature decay:', t_decay)
        print('Learning rate:', lr)

    t0 = time.time_ns()
    Gs = []
    for e in np.arange(1, n_episodes+1):
        state = START
        G = 0
        while state not in TERMINAL:
            
            # pick action
            p = softmax(Q[states.index(state)], t=temperature)
            action = np.random.choice(actions, p=p)

            # get next state
            next_state = getNextState(state, action, debug=False)

            # get reward
            r = R[next_state]
            G = r + (G * gamma)

            # update q
            Q[states.index(state)][action] += lr * (r + gamma * np.max(Q[states.index(next_state)]) - Q[states.index(state)][action])

            state = next_state

        # decay temperature if needed
        temperature *= t_decay

        Gs.append(G)

    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    if verbose:
        print()
        print('Soft Max exploration completed')
        print('Runtime in ms: ', runtime)
        print('Mean Cummulative Reward:', np.mean(Gs))
        print('Q:\n', Q)

        for state in states:
                V[state] = np.max(Q[states.index(state)]).copy()
        print('V:\n', V)

    return Gs, runtime


def q_learning_et(R, Q, V, states, actions, epsilon=EPSILON,
               gamma=DISCOUNT, lr=LEARNING_RATE, n_episodes=1000,
               lamb=0.5, e_decay=1., verbose=False):
    if verbose:
        print('Starting Q learning with eligibility traces')
        print('Episodes: ', n_episodes)
        print('Discount factor:', gamma)
        print('Epsilon:', epsilon)
        print('Epsilon decay:', e_decay)
        print('Learning rate:', lr)

    t0 = time.time_ns()
    Gs = []
    E = np.zeros(Q.shape)                            # 4 by 16 table
    for e in np.arange(1, n_episodes + 1):
        # Initialize (s,a)
        state = START                                   # Starting coordinates
        action = np.random.choice(actions)              # Random action (0 to 3)
        G = 0
        while state not in TERMINAL:

            # Observe next_state
            next_state = getNextState(state, action, debug=False)    # Coordinates next state

            # Observe reward
            r = R[next_state]
            G = r + (G * gamma)

            # Choosing action from policy
            if np.random.binomial(1, epsilon):
                # with chance of epsilon, pick a random action
                next_action = np.random.choice(actions)
            else:
                # otherwise pick a random action amongst the highest q value only
                best = np.argwhere(Q[states.index(next_state)] == np.max(Q[states.index(next_state)])).flatten()
                next_action = np.random.choice(best)

            # Determine a*
            best_next_action = np.argmax(Q[next_state])

            # Determine budew-looking thing (TD_error)
            budew = r + gamma * Q[states.index(next_state)][best_next_action] - Q[states.index(state)][action]

            # Update e(s,a)
            E[states.index(state)][action] = E[states.index(state)][action] + 1

            for s in states:                # s = coordinates
                for a in actions:           # a = value from 0 to 3
                    Q[states.index(s)][a] = Q[states.index(s)][a] + lr * budew * E[states.index(s)][a]
                    if next_action == best_next_action:
                        E[states.index(s)][a] = gamma * lamb * E[states.index(s)][a]
                    else:
                        E[states.index(s)][a] = 0

            state = next_state
            action = next_action

        # decay epsilon if needed
        epsilon *= e_decay

        Gs.append(G)

    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    if verbose:
        print()
        print('Q Eligibility Traces completed')
        print('Mean Cummulative Reward:', np.mean(Gs))
        print('Runtime in ms: ', runtime)
        print('Q:\n', Q)

        for state in states:
            V[state] = np.max(Q[states.index(state)])
        print('V:\n', V)

    return Gs, runtime


def sarsa(R, Q, V, states, actions, epsilon=EPSILON,
    gamma=DISCOUNT, lr=LEARNING_RATE, n_episodes=1000, e_decay=1., verbose=False):
    if verbose:
        print('Starting SARSA')
        print('Episodes: ', n_episodes)
        print('Discount factor:', gamma)
        print('Epsilon:', epsilon)
        print('Epsilon decay:', e_decay)
        print('Learning rate:', lr)

    t0 = time.time_ns()
    Gs = []
    for e in np.arange(1, n_episodes+1):
        state = START
        G = 0
        step = 1
        # Epsilon greedy
        if np.random.binomial(1, epsilon):
            # with chance of epsilon, pick a random action
            action = np.random.choice(actions)
        else:
            # otherwise pick a random action amongst the highest q value only
            best = np.argwhere(Q[states.index(state)] == np.max(Q[states.index(state)])).flatten()
            action = np.random.choice(best)

        while state not in TERMINAL:
            # get new state
            next_state = getNextState(state, action, debug=False)

            # get reward
            r = R[next_state]
            G = r + (G * gamma)
            #G += r * (gamma ** step)

            # Choose action
            if np.random.binomial(1, epsilon):
                # with chance of epsilon, pick a random action
                next_action = np.random.choice(actions)
            else:
                # otherwise pick a random action amongst the highest q value only
                best = np.argwhere(Q[states.index(next_state)]==np.max(Q[states.index(next_state)])).flatten()
                next_action = np.random.choice(best)

            # update q
            Q[states.index(state)][action] += lr * (r + gamma * Q[states.index(next_state)][next_action] - Q[states.index(state)][action])

            state = next_state
            action = next_action

        # decay epsilon if needed
        epsilon *= e_decay
        Gs.append(G)

    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    if verbose:
        print()
        print('SARSA completed')
        print('Runtime in ms: ', runtime)
        print('Mean Cummulative Reward:', np.mean(Gs))
        print('Q:\n', Q)

        for state in states:
                V[state] = np.max(Q[states.index(state)])
        print('V:\n', V)

    return Gs, runtime


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
        print()
        print('State values after iteration %i:' % i)
        print(V)
        if delta < theta:
            break

    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    print()
    print('Random policy iteration converged')
    print('Gamma:', gamma)
    print('Runtime in ms: ', runtime)
    print('Number of sweeps over state space:', len(deltas))
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
    print('Gamma:', gamma)
    print('Runtime in ms: ', runtime)
    print('Number of sweeps over state space:', len(deltas))
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
    print('Gamma:', gamma)
    print('Runtime in ms: ', runtime)
    print('Number of sweeps over state space:', len(deltas))
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
        print()
        print('State values after iteration %i:' % i)
        print(V)
        if delta < theta:
            break
    
    t1 = time.time_ns()
    runtime = (t1 - t0) / 1000000
    print()
    print('Value iteration converged')
    print('Gamma:', gamma)
    print('Runtime in ms: ', runtime)
    print('Number of sweeps over state space:', len(deltas))
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