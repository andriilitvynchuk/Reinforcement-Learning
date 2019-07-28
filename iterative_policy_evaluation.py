import numpy as np 
import matplotlib.pyplot as plt 
from gridworld import standard_grid

SMALL_ENOUGH = 10e-4

def print_values(V, g):
	print('-------------------')
	for i in range(g.width):
		for j in range(g.height):
			v = V.get((i, j), 0)
			if v >= 0:
				print(' %.2f|' %v, end=' ')
			else:
				print('%.2f|' %v, end=' ')
		print('')
	print('-------------------')


def print_policy(P, g):
	print('-------------------')
	for i in range(g.width):
		for j in range(g.height):
			v = P.get((i, j), 0)
			if type(v) != str:
				if v >= 0:
					print(' %.2f|' %v, end=' ')
				else:
					print('%.2f|' %v, end=' ')
			else :
				print(f'{v}|', end = ' ')
		print('')
	print('-------------------')


if __name__ == '__main__':

	grid = standard_grid()
	states = grid.all_states()

	V = {}
	for s in states:
		V[s] = 0
	gamma = 1.0 #discount factor

	while True:
		biggest_change = 0
		for s in states:
			old_v = V[s]
			#only not terminal states, terminal states have V(s) = 0
			if s in grid.actions:

				new_v = 0
				p_a = 1.0 / len(grid.actions[s]) #UNIFORM
				for a in grid.actions[s]:
					#print(s, a)
					grid.set_state(s)
					r = grid.move(a)
					new_v += p_a * (r + gamma * V[grid.current_state()])

				V[s] = new_v
				biggest_change = max(biggest_change, np.abs(old_v - V[s]))

		if biggest_change < SMALL_ENOUGH:
			break
	print('Values for uniformly random actions:')
	print_values(V, grid)
	print('\n')

	### fixed policy
	policy = {(2, 0): 'U',
			  (1, 0): 'U',
			  (0, 0): 'R',
			  (0, 1): 'R',
			  (0, 2): 'R',
			  (1, 2): 'R',
			  (2, 1): 'R',
			  (2, 2): 'R',
			  (2, 3): 'U'}
	print_policy(policy, grid)

	V = {}
	for s in states:
		V[s] = 0
	gamma = 0.9 #discount factor

	while True:
		biggest_change = 0
		for s in states:
			old_v = V[s]
			#only not terminal states, terminal states have V(s) = 0
			if s in policy:
				a = policy[s]
				grid.set_state(s)
				r = grid.move(a)
				V[s] = r + gamma * V[grid.current_state()]
				biggest_change = max(biggest_change, np.abs(old_v - V[s]))

		if biggest_change < SMALL_ENOUGH:
			break
	print('Values for fixed actions:')
	print_values(V, grid)
	print('\n')