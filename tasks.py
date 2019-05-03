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
	R, Q, V, states, actions = main.initialize(init=0)
	main.random_policy(R, Q, V, states, actions, gamma=0.9)
	print('Task 2 complete.')
	print()


def task3():
	print('TASK 3:')
	print('Value iteration for various discount factors.')
	results = {}
	for g in np.arange(0.1, 1.1, 0.4):
		R, Q, V, states, actions = main.initialize(init=0)
		deltas = main.value_iteration(R, Q, V, states, actions, gamma=g)
		results.update({'Value iteration (gamma = %.1f)' % g : deltas})
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
		results.update({'Howards P.I. (gamma = %.1f)' % g : deltas})
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
		results.update({'Simple P.I. (gamma = %.1f)' % g : deltas})
	print('Task 5 complete.')
	print()

	return results


def task7():
	print('TASK 7:')
	print('Additional algorithms.')
	print('Optimistic initialization for Q(t=0)')
	for i in [-10, -1, 0, 1, 10, 100]:
		R, Q, V, states, actions = main.initialize(init=i)
		main.policy_iteration(R, Q, V, states, actions)
	print('Task 7 complete.')
	print()


def plot_deltas(all_deltas):

	for name, deltas in all_deltas.items():
			plt.plot(np.arange(1,len(deltas)+1), deltas, label=name)
	plt.title('Maximum Value function update per iteration')
	plt.xlabel('Iteration')
	plt.ylabel('Delta')
	plt.legend()
	plt.show()



if __name__ == '__main__':
	#all_deltas = {}
	task1()
	input('Press ENTER to continue')
	print('\n', 100 * '#', '\n')
	task2()
	input('Press ENTER to continue')
	print('\n', 100 * '#', '\n')
	results = task3()
	plot_deltas(results)
	input('Press ENTER to continue')
	print('\n', 100 * '#', '\n')
	results = task4()
	plot_deltas(results)
	input('Press ENTER to continue')
	print('\n', 100 * '#', '\n')
	results = task5()
	plot_deltas(results)
	input('Press ENTER to continue')
	print('\n', 100 * '#', '\n')
	task7()
	#print(all_deltas)
	#plot_deltas(all_deltas)

