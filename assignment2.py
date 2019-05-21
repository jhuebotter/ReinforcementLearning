''' Assignment: Planning & Reinforcement Learning Assignment 2
    By: Justus Hübotter (2617135), Florence van der Voort (2652198), Stefan Wijtsma (2575874)
    Created @ April 2019'''

import main
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

RUNS = 5
EPISODES = 10
LEARNING_RATE = 0.1


def task8():
	print('TASK 8:')
	print(f'{RUNS} Runs of Q learning on {EPISODES} episodes and three different learning rates.')
	results = {}
	for i in [0.005, 0.05, 0.2]:
		run = []
		for k in range(RUNS):
			R, Q, V, states, actions = main.initialize(init=0)
			print('Run', k+1)
			Gs = main.q_learning(R, Q, V, states, actions, lr=LEARNING_RATE, epsilon=i, n_episodes=EPISODES)
			run.append(Gs)
			print('\n\n')
		results.update({'epsilon = %.3f' % i: run})
	print('Task 8 complete.')
	print()

	return results


def task9():
	print('TASK 9:')
	print(f'{RUNS} Runs of Soft Max exploration strategy on {EPISODES} episodes and three different learning rates.')
	results = {}
	for i in [1.0]:
		run = []
		for k in range(RUNS):
			R, Q, V, states, actions = main.initialize(init=0)
			print('Run', k+1)
			Gs = main.soft_max(R, Q, V, states, actions, lr=LEARNING_RATE, temperature=i, n_episodes=EPISODES)
			run.append(Gs)
			print('\n\n')
		results.update({'temp = %.3f' % i: run})
	print('Task 9 complete.')
	print()

	return results


def task10():
	print('TASK 10:')
	print(f'{RUNS} Runs of SARSA on {EPISODES} episodes and three different learning rates.')
	results = {}
	for i in [0.005, 0.05, 0.2]:
		run = []
		for k in range(RUNS):
			R, Q, V, states, actions = main.initialize(init=0)
			print('Run', k+1)
			Gs = main.sarsa(R, Q, V, states, actions, lr=LEARNING_RATE, epsilon=i, n_episodes=EPISODES)
			run.append(Gs)
			print('\n\n')
		results.update({'epsilon = %.3f' % i: run})
	print('Task 10 complete.')
	print()

	return results


def plot_results(results, title, savefig=True):

	fig, ax = plt.subplots(1)
	for k, v in results.items():
		Y = np.array(v)#.cumsum(axis=1)
		M = Y.mean(axis=0)
		S = Y.std(axis=0)
		X = np.arange(1,len(M)+1)
		ax.plot(X, M, label=k)
		ax.fill_between(X, M-S, M+S, alpha=0.5)
		#ax.fill_between(X, Y.min(axis=0), Y.max(axis=0), alpha=0.2)
	plt.title('Performance of ' + title)
	plt.xlabel('Episodes')
	plt.ylabel('Mean cumulative reward')
	plt.legend()
	plt.tight_layout()
	if savefig:
		plt.savefig(title)
	else:
		plt.show()


if __name__ == '__main__':
	results = task8()
	plot_results(results, 'Q Learning')
	#input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task9()
	plot_results(results, 'Soft Max')
	#input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task10()
	plot_results(results, 'SARSA')