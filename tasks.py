import main
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def task1():
	print('TASK 1:')
	print('Interactive text interface to navigate through the ice world environment.')
	R, Q, V, states, actions = main.initialize(init=0)
	main.manual(R)
	print('Task 1 complete.')
	print()


def task2():
	print('TASK 2:')
	print('Random policy implementation.')
	results = {}
	R, Q, V, states, actions = main.initialize(init=0)
	for g in np.arange(0.1, 1.1, 0.4):
		deltas = main.random_policy(R, Q, V, states, actions, gamma=g)
		results.update({'gamma = %.1f' % g : deltas})
		print('\n\n')
	print('Task 2 complete.')
	print()

	return results


def task3():
	print('TASK 3:')
	print('Value iteration for various discount factors.')
	results = {}
	for g in np.arange(0.1, 1.1, 0.4):
		R, Q, V, states, actions = main.initialize(init=0)
		deltas = main.value_iteration(R, Q, V, states, actions, gamma=g)
		results.update({'gamma = %.1f' % g : deltas})
		print('\n\n')
	print('Task 3 complete.')
	print()

	return results


def task4():
	print('TASK 4:')
	print('Howards policy iteration.')
	results = {}
	for g in np.arange(0.1, 1.1, 0.4):
		R, Q, V, states, actions = main.initialize(init=0)
		deltas = main.policy_iteration(R, Q, V, states, actions, gamma=g)
		results.update({'gamma = %.1f' % g : deltas})
	print('Task 4 complete.')
	print()

	return results


def task5():
	print('TASK 5.1:')
	print('Simple policy iteration.')
	results = {}
	for g in np.arange(0.1, 1.1, 0.4):
		R, Q, V, states, actions = main.initialize(init=0)
		deltas = main.simple_policy_iteration(R, Q, V, states, actions, gamma=g)
		results.update({'gamma = %.1f' % g : deltas})
	print('Task 5 complete.')
	print()

	return results


def task71():
	print('TASK 7.1:')
	print('Additional algorithms.')
	print('Optimistic initialization for Q(t=0)')
	results = {}
	for i in [100]:
		for g in np.arange(0.1, 1.1, 0.4):
			R, Q, V, states, actions = main.initialize(init=i)
			deltas = main.value_iteration(R, Q, V, states, actions, gamma=g)
			#main.policy_iteration(R, Q, V, states, actions)
			results.update({'init = %i, gamma = %.1f' % (i, g) : deltas})
		print('\n\n')
	print('Task 7.1 complete.')
	print()

	return results


def task72():
	print('TASK 7.2:')
	print('Additional algorithms.')
	results = {}
	print('Pessimistic initialization for Q(t=0)')
	for i in [-10]:
		for g in np.arange(0.1, 1.1, 0.4):
			R, Q, V, states, actions = main.initialize(init=i)
			deltas = main.value_iteration(R, Q, V, states, actions, gamma=g)
			#main.policy_iteration(R, Q, V, states, actions)
			results.update({'init = %i, gamma = %.1f' % (i, g) : deltas})
			print('\n\n')
	print('Task 7.2 complete.')
	print()

	return results


def plot_deltas(all_deltas, title):

	for name, deltas in all_deltas.items():
			plt.plot(np.arange(1,len(deltas)+1), deltas, label=name)
	plt.title('Convergence of ' + title)
	plt.xlabel('Sweeps over state space')
	plt.ylabel('Delta')
	plt.legend()
	plt.show()



if __name__ == '__main__':
	#all_deltas = {}
	task1()
	input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task2()
	plot_deltas(results, 'Random Policy')
	input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task3()
	plot_deltas(results, 'Value iteration')
	input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task4()
	plot_deltas(results, 'Howard´s Policy Iteration')
	input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task5()
	plot_deltas(results, 'Simple Policy Iteration')
	input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task71()
	plot_deltas(results, 'VI with Optimistic value initialization')
	input('Press ENTER to continue')
	print('\n\n', 100 * '#', '\n\n')
	results = task72()
	plot_deltas(results, 'VI with Pessimistic value initialization')

	#print(all_deltas)
	#plot_deltas(all_deltas)

