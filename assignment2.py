''' Assignment: Planning & Reinforcement Learning Assignment 2
    By: Justus Hübotter (2617135), Florence van der Voort (2652198), Stefan Wijtsma (2575874)
    Created @ April 2019'''

import main
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def task8():
	print('TASK 8:')
	print('Q learning on 100 episodes and three different learning rates.')
	results = {}
	R, Q, V, states, actions = main.initialize(init=0)
	for i in [0.5, 0.1, 0.01, 0.001, 0.0001]:
		Gs = main.q_learning(R, Q, states, actions, lr=i, epsilon=0.05, n_episodes=100)
		results.update({'lr = %.3f' % i: Gs})
		print('\n\n')
	print('Task 8 complete.')
	print()

	return results


def task9():
	print('TASK 9:')
	print('Soft Max exploration strategy on 100 episodes and three different learning rates.')
	results = {}
	R, Q, V, states, actions = main.initialize(init=0)
	for i in [0.5, 0.1, 0.01, 0.001]:
		Gs = main.soft_max(R, Q, states, actions, lr=i, n_episodes=100)
		results.update({'lr = %.3f' % i: Gs})
		print('\n\n')
	print('Task 9 complete.')
	print()

	return results


def task10():
	print('TASK 10:')
	print('SARSA on 100 episodes and three different learning rates.')
	results = {}
	R, Q, V, states, actions = main.initialize(init=0)
	for i in [0.5, 0.1, 0.01, 0.001]:
		Gs = main.sarsa(R, Q, states, actions, lr=i, n_episodes=100)
		results.update({'lr = %.3f' % i: Gs})
		print('\n\n')
	print('Task 10 complete.')
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
	results = task8()
	plot_results(results, 'Q Learning')
	'''
	input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task9()
	plot_results(results, 'Soft Max')
	input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task10()
	plot_results(results, 'SARSA')
	'''