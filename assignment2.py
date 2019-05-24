''' Assignment: Planning & Reinforcement Learning Assignment 2
    By: Justus Hübotter (2617135), Florence van der Voort (2652198), Stefan Wijtsma (2575874)
    Created @ April 2019'''

import main
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

RUNS = 256
EPISODES = 1000
LEARNING_RATE = 0.1

def task8():
	print('TASK 8:')
	print(f'{RUNS} Runs of Q learning on {EPISODES} episodes.')
	results = {}
	time = {}
	for i in [0.01, 0.1]:
		run = []
		runtimes = []
		for k in range(RUNS):
			R, Q, V, states, actions = main.initialize(init=0)
			print('Run', k+1)
			Gs, runtime = main.q_learning(R, Q, V, states, actions, lr=LEARNING_RATE, epsilon=i, n_episodes=EPISODES)
			run.append(Gs)
			runtimes.append(runtime)
			print('\n\n')
		results.update({'epsilon = %.2f' % (i): run})
		time.update({'epsilon = %.2f' % (i): runtimes})
	for k, v in time.items():
		print(k)
		print('Mean runtime [ms] +- SE: %.2f (+- %.2f)' % (np.mean(v), np.std(v)/np.sqrt(RUNS)))
		print()
	print('Task 8 complete.')
	print()

	return results


def task9():
	print('TASK 9:')
	print(f'{RUNS} Runs of Soft Max exploration strategy on {EPISODES} episodes.')
	results = {}
	time = {}
	for i in [1.0, 20.0, 50.0]:
		run = []
		runtimes = []
		for k in range(RUNS):
			R, Q, V, states, actions = main.initialize(init=0)
			print('Run', k+1)
			Gs, runtime = main.soft_max(R, Q, V, states, actions, lr=LEARNING_RATE, temperature=i, n_episodes=EPISODES)
			run.append(Gs)
			runtimes.append(runtime)
			print('\n\n')
		results.update({'temp = %.1f' % i: run})
		time.update({'temp = %.1f' % i: runtimes})
	for k, v in time.items():
		print(k)
		print('Mean runtime [ms] +- SE: %.2f (+- %.2f)' % (np.mean(v), np.std(v)/np.sqrt(RUNS)))
		print()
	print('Task 9 complete.')
	print()

	return results


def task10():
	print('TASK 10:')
	print(f'{RUNS} Runs of SARSA on {EPISODES} episodes.')
	results = {}
	time = {}
	for i in [0.01, 0.1]:
		run = []
		runtimes = []
		for k in range(RUNS):
			R, Q, V, states, actions = main.initialize(init=0)
			print('Run', k+1)
			Gs, runtime = main.sarsa(R, Q, V, states, actions, lr=LEARNING_RATE, epsilon=i, n_episodes=EPISODES)
			run.append(Gs)
			runtimes.append(runtime)
			print('\n\n')
		results.update({'epsilon = %.3f' % i: run})
		time.update({'epsilon = %.2f' % i: runtimes})
	for k, v in time.items():
		print(k)
		print('Mean runtime [ms] +- SE: %.2f (+- %.2f)' % (np.mean(v), np.std(v)/np.sqrt(RUNS)))
		print()
	print('Task 10 complete.')
	print()

	return results


def task11():
	print('TASK 11:')
	print(f'{RUNS} Runs of Q learning with experience replay on {EPISODES} episodes.')
	results = {}
	time = {}
	for i in [0.01, 0.1]:
		run = []
		runtimes = []
		for k in range(RUNS):
			R, Q, V, states, actions = main.initialize(init=0)
			print('Run', k+1)
			Gs, runtime = main.q_learning_er(R, Q, V, states, actions, lr=LEARNING_RATE, epsilon=i, n_episodes=EPISODES)
			run.append(Gs)
			runtimes.append(runtime)
			print('\n\n')
		results.update({'epsilon = %.3f' % i: run})
		time.update({'epsilon = %.2f' % i: runtimes})
	print('Task 11 complete.')
	print()

	return results


def task12():
	print('TASK 12:')
	print(f'{RUNS} Runs of Q learning with eligibility traces on {EPISODES} episodes.')
	results = {}
	time = {}
	for i in [0.01, 0.1]:
		run = []
		runtimes = []
		for k in range(RUNS):
			R, Q, V, states, actions = main.initialize(init=0)
			print('Run', k+1)
			Gs, runtime = main.q_learning_et(R, Q, V, states, actions, lr=LEARNING_RATE, epsilon=i, n_episodes=EPISODES)
			run.append(Gs)
			runtimes.append(runtime)
			print('\n\n')
		results.update({'epsilon = %.3f' % i: run})
		time.update({'epsilon = %.2f' % i: runtimes})
	print('Task 12 complete.')
	print()

	return results


def task14c():
	print('TASK 14c:')
	print(f'{RUNS} Runs of double Q learning on {EPISODES} episodes.')
	results = {}
	time = {}
	for i in [0.01, 0.1]:
		run = []
		for k in range(RUNS):
			R, Q, V, states, actions = main.initialize(init=0)
			print('Run', k+1)
			Gs, runtime = main.double_q_learning(R, Q, V, states, actions, lr=LEARNING_RATE, epsilon=i, n_episodes=EPISODES)
			run.append(Gs)
			runtimes.append(runtime)
			print('\n\n')
		results.update({'epsilon = %.3f' % i: run})
		time.update({'epsilon = %.2f' % i: runtimes})
	for k, v in time.items():
		print(k)
		print('Mean runtime [ms] +- SE: %.2f (+- %.2f)' % (np.mean(v), np.std(v)/np.sqrt(RUNS)))
		print()
	print('Task 14c complete.')
	print()

	return results


def plot_results(results, title, savefig=True):

	fig, ax = plt.subplots(1)
	for k, v in results.items():
		Y = np.array(v)
		M = Y.mean(axis=0)
		S = Y.std(axis=0) / np.sqrt(RUNS)
		X = np.arange(1,len(M)+1)
		ax.plot(X, M, label=k)
		#ax.fill_between(X, M-S, M+S, alpha=0.5)
		#ax.fill_between(X, Y.min(axis=0), Y.max(axis=0), alpha=0.2)
	plt.title('Performance of ' + title)
	plt.xlabel('Episodes')
	plt.ylabel('Mean cumulative reward')
	plt.legend()
	plt.tight_layout()
	if savefig:
		plt.savefig('%s_r%i_e%i.csv' % (title, RUNS, EPISODES))
	else:
		plt.show()


def save_results(results, name, a=[9, 99, 999, 1999, EPISODES-1]):
	dict_of_df = {k: pd.DataFrame(np.array(v).T) for k,v in results.items()}
	df = pd.concat(dict_of_df, axis=1)
	df.loc[a].reindex().to_csv('%s_r%i_e%i.csv' % (name, RUNS, EPISODES), index=False, sep=';',mode='w')


if __name__ == '__main__':
	results = task8()
	plot_results(results, 'Q Learning')
	save_results(results, 'Q Learning')
	#input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task9()
	plot_results(results, 'Soft Max')
	save_results(results, 'Soft Max')
	#input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task10()
	plot_results(results, 'SARSA')
	save_results(results, 'SARSA')
	#input('Press ENTER to continue')
	#print('\n\n', 100 * '#', '\n\n')
	#results = task11()
	#plot_results(results, 'Q Learning ER')
	#save_results(results, 'Q Learning ER')
	#input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task12()
	plot_results(results, 'Q Learning ET')
	save_results(results, 'Q Learning ET')
	#input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task14c()
	plot_results(results, 'Double Q Learning')
	save_results(results, 'Double Q Learning')
