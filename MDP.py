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
lr = 0.01
epsilon = 0.1 #0.10
decay = 0.0#01
init = 0
terminal = [x for x in crack]
terminal.append(goal)


R = np.zeros((4,4))
V = np.zeros((4,4))

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
			R[i,j] = 0 #5 

print(R)

Q = np.ones((len(states),len(actions)))
Q *= init

def main():
	rewards = []

	env = world.World()

	for i in range(1000):

		pos = start

		G = 0

		while (not isGameOver(pos)):
			e = epsilon
			#board = V.copy()
			#board[pos] = 1
			#print('current position:', pos)
			#print(board)
			if np.random.uniform() < e:
				action = np.random.choice(actions)
			else:
				possible = Q[states.index(pos)]
				best = np.argwhere(possible == np.max(possible)).flatten().tolist()
				action = np.random.choice(best)

			new_pos = getNextState(pos, action)
			
			Q[states.index(pos)][action] += lr * (R[new_pos] + discount * max(Q[states.index(new_pos)]) - Q[states.index(pos)][action])

			reward = R[new_pos]

			G += reward

			#print('reward gained:', reward)
			#print('new position:', new_pos)

			V[new_pos] = V[pos] + (discount * V[new_pos] - V[pos]) #reward
			Q[states.index(new_pos), action] = Q[states.index(pos), action] + lr * (reward - Q[states.index(pos), action])

			pos = new_pos
			
			if e > 0.:
				e -= decay

		print()
		print('iteration:', i)
		#print('the game has ended.')
		print('cummulative reward:', G)
		#print(Q)
		rewards.append(G)

	print(Q)
	ma_rewards = np.convolve(rewards, np.ones((50,))/50,mode='valid')
	plt.plot(ma_rewards)
	plt.show()


def isGameOver(pos):
	return pos in terminal
		

def getNextState(pos, action, debug=False):
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

if __name__ == '__main__':
	main()
