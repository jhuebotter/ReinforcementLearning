''' Assignment: mainning & Reinforcement Learning Assignment 2
    By: Justus Hübotter (2617135), Florence van der Voort (2652198), Stefan Wijtsma (2575874)
    Created @ April 2019'''

import main
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def task8():
	print('TASK 1:')
	print('Q learning on 1000 episodes and three different learning rates.')
	results = {}
	R, Q, V, states, actions = main.initialize(init=0)
	for i in [0.2, 0.1, 0.05, 0.01, 0.001]:
		Gs = main.q_learning(R, Q, states, actions, lr=i, n_episodes=1000)
		results.update({'lr = %.3f' % i: Gs})
	print('Task 1 complete.')
	print()

	return results


def task9():
	print('TASK 2:')
	print('Soft Max exploration strategy on 1000 episodes and three different learning rates.')
	results = {}
	R, Q, V, states, actions = main.initialize(init=0)
	for g in np.arange(0.1, 1.1, 0.4):
		deltas = main.soft_max(R, Q, V, states, actions, gamma=g)
		results.update({'gamma = %.1f' % g : deltas})
		print('\n\n')
	print('Task 2 complete.')
	print()

	return results


def task10():
	print('TASK 3:')
	print('SARSA on 1000 episodes and three different learning rates.')
	results = {}
	for g in np.arange(0.1, 1.1, 0.4):
		R, Q, V, states, actions = main.initialize(init=0)
		deltas = main.sarsa(R, Q, V, states, actions, gamma=g)
		results.update({'gamma = %.1f' % g : deltas})
		print('\n\n')
	print('Task 3 complete.')
	print()

	return results


def plot_results(results, title):

	for k, v in results.items():
			plt.plot(np.arange(1,len(v)+1), v, label=k)
	plt.title('Convergance of ' + title)
	plt.xlabel('Episodes')
	plt.ylabel('Cummulative reward')
	plt.legend()
	plt.show()



if __name__ == '__main__':
	#all_deltas = {}
	results = task8()
	plot_results(results, 'Q Learning')
	input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task9()
	plot_results(results, 'Soft Max')
	input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task10()
	plot_results(results, 'SARSA')