import numpy as np
import time

def dynamics(y, params, F=0.0):
	"""
	y: [x1, x1d, x2, x2d]
	params: dict with keys m1,m2,k1,k2,c1,c2,cd
	F: external force on mass 1 (scalar)
	return: dy/dt as numpy array
	"""
	x1, x1d, x2, x2d = y
	m1 = params["m1"]; m2 = params["m2"]
	k1 = params["k1"]; k2 = params["k2"]
	c1 = params["c1"]; c2 = params["c2"]; cd = params["cd"]

	x1dd = (-k1 * x1 - c1 * x1d + cd * (x2d - x1d) + F) / m1
	x2dd = (-k2 * x2 - c2 * x2d - cd * (x2d - x1d)) / m2

	return np.array([x1d, x1dd, x2d, x2dd], dtype=float)

def _rk4_step(fun, y, t, dt, *args):
	k1 = fun(y, *args)
	k2 = fun(y + 0.5*dt*k1, *args)
	k3 = fun(y + 0.5*dt*k2, *args)
	k4 = fun(y + dt*k3, *args)
	return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate(params, y0, t, F=0.0):
	"""
	Simulate the system using RK4.
	params: dict as above
	y0: initial state array_like [x1, x1d, x2, x2d]
	t: 1D array of times (must be increasing)
	F: scalar constant force
	return: array of shape (len(t), 4)
	"""
	t = np.asarray(t, dtype=float)
	y = np.asarray(y0, dtype=float)
	n = t.size
	sol = np.zeros((n, 4), dtype=float)
	sol[0] = y.copy()
	for i in range(n-1):
		dt = t[i+1] - t[i]
		if dt <= 0:
			raise ValueError("t must be strictly increasing")
		y = _rk4_step(lambda yy, p, ff: dynamics(yy, p, ff), y, t[i], dt, params, F)
		sol[i+1] = y
	return sol

if __name__ == "__main__":
	params = {
		"m1": 1.0, "m2": 1.5,
		"k1": 10.0, "k2": 20.0,
		"c1": 0.5, "c2": 0.6, "cd": 1.2
	}
	y0 = np.array([0.0, 0.0, 0.1, 0.0], dtype=float)
	t = np.linspace(0.0, 10.0, 501)
	F = 1.0

	start = time.time()
	sol = simulate(params, y0, t, F)
	end = time.time()
	print("t[0], state[0]:", t[0], sol[0])
	print("Total time for simulation:", end - start)