# TG vortex decay problem
# Test for discrete energy and physical energy
# reproduce Doan-Hoang-Ju-Lan-2025
# do the KH instability experiment

from firedrake import *
import csv
from mpi4py import MPI
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper
from tabulate import tabulate
from ufl.algorithms.ad import expand_derivatives
import numpy as np
import math

def energy_u(u):
    return 0.5 * assemble(inner(u, u) * dx)
    

def real_energy(t):
    k = 8 * pi**2
    u1 = sin(2*pi*x) * cos(2*pi*y) * exp(-nu * k * t)
    u2 = -cos(2 * pi * x) * sin(2 * pi * y) * exp(-nu * k * t)
    u_real = as_vector([u1, u2])
    return 1/4 * np.exp(-16 * pi **2 * float(nu) * float(t))

def scurl(x):
    return x[1].dx(0) - x[0].dx(1)

stage = 1
T = 200/28
nu = Constant(1/2800)

L = 64
mesh = PeriodicUnitSquareMesh(L, L, direction="x")

(x, y) = SpatialCoordinate(mesh)

Vg = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

Z = MixedFunctionSpace([Vg, Q])

# z1
z1 = Function(Z)
z1_test = TestFunction(Z)
z1_prev = Function(Z)

(u1, p1) = split(z1)
(u1t, p1t) = split(z1_test)
(u1p, p1p) = split(z1_prev)

# z2
z2 = Function(Z)
z2_test = TestFunction(Z)
z2_prev = Function(Z)

(u2, p2) = split(z2)
(u2t, p2t) = split(z2_test)
(u2p, p2p) = split(z2_prev)


# time parameters
dt = Constant(1/560)
t = Constant(0)

# Lagrange multiplier
theta = 100.0

# solution
u_sol = Function(Vg, name="Velocity")
p_sol = Function(Q, name="Pressure")

def v_grad(x):
    return as_vector([x.dx(1), -x.dx(0)])

delta0 = 1/28
u_0 = as_vector([tanh((2 * y-1)/delta0), 0])
psi = 0.001 * exp(-(y-0.5)**2/delta0) * (cos(8*pi*x) + cos(20*pi*x))
u_per = v_grad(psi)
u_ex = u_0 + u_per

#initial condition
z1_prev.sub(0).interpolate(u_ex)


# u1 p1
F1 =(
        #u
      inner((u1 - u1p)/dt, u1t) * dx
    + nu * inner(grad(u1), grad(u1t)) * dx
    + inner(grad(p1), u1t) * dx
    #p
    + inner(u1, grad(p1t)) * dx
)

# u2 p2
F2 =(
        #u
      inner(u2/dt, u2t) * dx
    + nu * inner(grad(u2), grad(u2t)) * dx
    + inner(dot(grad(u1p), u1p), u2t) * dx # F(u^n)
    + inner(grad(p2), u2t) * dx
    #p
    + inner(u2, grad(p2t)) * dx
)


lu = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "snes_rtol": 1e-12,
    "snes_atol": 1e-12, 
    "snes_max_it": 100,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

sp = lu

def compute_A(u2):
    return assemble(0.5 * inner(u2, u2) * dx) + theta + assemble(dt * nu * inner(grad(u2), grad(u2)) * dx)

def compute_B(u1, u2, u1p):
    return assemble(-inner(u1 - u1p, u2)* dx - dt * inner(dot(grad(u1p), u1p), u1) * dx)

def compute_C(u1, u1p, p1p, q):
    return assemble(-0.5 * inner(u1 - u1p, u1 - u1p) * dx) - theta * q ** 2

def compute_root(A, B, C):
    # compute the discreminat 
    D = B**2 - 4 * A * C

    x1 = (-B + math.sqrt(D)) / (2 * A)
    x2 = (-B - math.sqrt(D)) / (2 * A)
    
    if x1 > 0 and x2 > 0:
        return min(x1, x2)
    elif x1 > 0:
        return x1
    elif x2 > 0:
        return x2
    else:
        return None 


#bcs = [DirichletBC(Z.sub(0).sub(1), 0, (1, 2))]
bcs = None
pb1 = NonlinearVariationalProblem(F1, z1, bcs)
solver1 = NonlinearVariationalSolver(pb1, solver_parameters = sp)

pb2 = NonlinearVariationalProblem(F2, z2, bcs)
solver2 = NonlinearVariationalSolver(pb2, solver_parameters = sp)

q_init = 1.0
pvd = VTKFile("output/drlm.pvd")
w = Function(Q, name="Vorticity")
w.interpolate(scurl(u_sol))
pvd.write(u_sol, p_sol, w, time=float(t))

while (float(t) < float(T-dt)+1.0e-10):
    t.assign(t+dt)
    if mesh.comm.rank == 0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    solver1.solve()
    (u1, p1) = z1.subfunctions
    
    solver2.solve()
    (u2, p2) = z2.subfunctions
    
    # compute the coefficients
    A = compute_A(u2)
    B = compute_B(u1, u2, u1p)
    C = compute_C(u1, u1p, p1p, q_init)
      
    q = compute_root(A, B, C)
    q_const = Constant(q)
    u_sol.assign(u1 + q_const * u2)
    p_sol.assign(p1 + q_const * p2)
    energy = energy_u(u_sol)
    energyR = real_energy(t)
    diff = energy - energyR

    if mesh.comm.rank == 0:
        print(RED % f"t={float(t)}, energy={energy}, realEnergy={energyR}, diff={diff}")
    w.interpolate(scurl(u_sol))
    pvd.write(u_sol, p_sol, w, time=float(t))
    z1_prev.assign(z1)
    z2_prev.assign(z2)
    q_init = q

     
         

