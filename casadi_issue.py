import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

opti = ca.Opti()

# System variables
x = opti.variable(2)
u = opti.variable()

# Terminal time
T = opti.variable()

# Dynamics
f = ca.Function("F", [x, u], [ca.vertcat(x[1], u)])

N = 50
dt = T/N # length of a control interval

X = opti.variable(2, N+1)
x1 = X[0, :]
x2 = X[1, :]

# Control
U = opti.variable(1, N)

# Cost
opti.minimize(T)

# RK4
k1 = f(x, u)
k2 = f(x + dt / 2.0 * k1, u)
k3 = f(x + dt / 2.0 * k2, u)
k4 = f(x + dt * k3, u)
xf = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

# Single step time propagation
F_RK4 = ca.Function("F_RK4", [x, u], [xf], ['x[k]', 'u[k]'], ['x[k+1]'])

# Gap-closing shooting constraints
gap_constraints = []
for k in range(N):
    con = (X[:,k+1] == F_RK4(X[:,k], U[k]))
    gap_constraints.append(con)
    opti.subject_to(con)

# constraints
opti.subject_to(opti.bounded(-1, U, 1))

# boundary conditions
opti.subject_to(X[:, 0] == [2, 2])
opti.subject_to(X[:, N] == [0, 0])

opti.subject_to(T > 0) # Time must be positive

# Provide initial guesses for the solver:
opti.set_initial(T, 3)

# Solve the NLP using IPOPT
#opti.solver('ipopt')
opti.solver('ipopt', {'calc_multipliers': True})
#opti.solver('ipopt', {'ipopt.fixed_variable_treatment': 'relax_bounds'})
#opti.solver('ipopt', {'ipopt.fixed_variable_treatment': 'make_constraint'})

#opti.solver('bonmin', {'bonmin': {'calc_lam_x': True}})

#options = {'qpsol': 'qrqp'}
#opti.solver('sqpmethod', options)

sol = opti.solve()

# Create time vector
t = np.linspace(0, sol.value(T), sol.value(T)/sol.value(dt))

# Aggregate multipliers
lambdas = np.array([sol.value(opti.dual(gap)) for gap in gap_constraints])