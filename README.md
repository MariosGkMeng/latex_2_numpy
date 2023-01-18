# What?
A Python module that converts Latex equations to numpy.

# Why?

1. It is easier to write and **review** a scientific-computing algorithm using the original mathematical syntax, rather than the computer programming expression
2. It can be linked to scientific documentation and even publications, without the need to program something from the beginning
3. It can eliminate the need to learn programming-language specific rules (why should you know which module is needed to use linear algebra operations?)
4. Since it looks prettier, it can "save you some sanity" during work when you need to follow strict deadlines and perform a lot of tasks

## Example

``` python
for j in range(max_length/dt):
	# Take a random action in the interval [-tau_lim,tau_lim] drawn from a uniform probability distribution
	e_greedy__get_random_u = rand < e

	if e_greedy__get_random_u:
		u_j = -tau_lim + 2*tau_lim*rand
	else:          
		Q_a_cond_s = Q_table[I[0]][I[1]][:]
		[~, idx_max_u] = np.max(Q_a_cond_s)
		u_j = p.X_discrete(end, idx_max_u)
	
	u = [u, u_j]

	# Simulate one timestep using forward Euler
	y0, I0 = discrete_S_A([x0, u_j], p)
	x0 += dt*pendulum_dyn(x0,u_j) 
	t0 += dt
	
	# get discrete states
	y, I = discrete_S_A([x0, u_j], p)

	# Store values for plotting
	q = [q, x0[0]]
	dq = [dq, x0[1]]
	t = [t, t0]
	
	# Check episode termination condition
	if (np.cos(x0(1)-pi) < min_cosAngle):
	   # Terminate episode 
	   break	
	
	# Reward
	R = p.reward_function(x0)
	episode_reward = episode_reward + p.gamma*R

	# Q-table update
	max__Q_cond_S1 = max(Q_table[I[0]][I[1]][:])
	Q_t = Q_table(I0[0], I0[1], I0[2]) # current value
	R_upd = R; # episode_reward R

	update = R_upd + p.gamma*max__Q_cond_S1 - Q_t
	Q_t = Q_t + e*update

	Q_table[I[0]][I[1]][I0[2]] = Q_t

```

$\text{for } j = 1,..,\frac{L_{max}}{2}$








