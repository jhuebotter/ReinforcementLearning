import main
import numpy as np


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
	for g in np.arange(0.1, 1.0, 0.1):
		R, Q, V, states, actions = main.initialize(init=0)
		main.value_iteration(R, Q, V, states, actions, gamma=g)
		print('\n\n')
	print('Task 3 complete.')
	print()


def task4():
	print('TASK 4:')
	print('Howards policy iteration.')
	R, Q, V, states, actions = main.initialize(init=0)
	main.policy_iteration(R, Q, V, states, actions)
	print('Task 4 complete.')
	print()


def task5():
	print('TASK 5.1:')
	print('Simple policy iteration.')
	R, Q, V, states, actions = main.initialize(init=0)
	main.simple_policy_iteration(R, Q, V, states, actions)
	print('Task 5 complete.')
	print()


def task7():
	print('TASK 7:')
	print('Additional algorithms.')
	print('Optimistic initialization for Q(t=0)')
	for i in [-10, -1, 0, 1, 10, 100]:
		R, Q, V, states, actions = main.initialize(init=i)
		main.policy_iteration(R, Q, V, states, actions)
	print('Task 7 complete.')
	print()


if __name__ == '__main__':
	task1()
	input('Press ENTER to continue')
	print('\n', 150 * '#', '\n')
	task2()
	input('Press ENTER to continue')
	print('\n', 150 * '#', '\n')
	task3()
	input('Press ENTER to continue')
	print('\n', 150 * '#', '\n')
	task4()
	input('Press ENTER to continue')
	print('\n', 150 * '#', '\n')
	task5()
	input('Press ENTER to continue')
	print('\n', 150 * '#', '\n')
	task7()

