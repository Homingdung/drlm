# Wang-Wang-Li-2026
# drlm for MHD (u, p, B)
# energy stability test

from firedrake import *
import csv
from mpi4py import MPI
from irksome import GaussLegendre, Dt, MeshConstant, TimeStepper
from tabulate import tabulate
from ufl.algorithms.ad import expand_derivatives
import numpy as np
import math



L = 4
mesh = UnitCubeMesh(L, L, L)

(x, y, z0) = SpatialCoordinate(mesh)

Vg = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

Z = MixedFunctionSpace([Vg, Q, Vg])

# z1
z1 = Function(Z)
z1_test = TestFunction(Z)
z1_prev = Function(Z)

(u1, p1, B1) = split(z1)
(u1t, p1t, B1t) = split(z1_test)
(u1p, p1p, B1p) = split(z1_prev)

# z2
z2 = Function(Z)
z2_test = TestFunction(Z)
z2_prev = Function(Z)

(u2, p2, B2) = split(z2)
(u2t, p2t, B2t) = split(z2_test)
(u2p, p2p, B2p) = split(z2_prev)

# time parameters
dt = Constant(0.02)
t = Constant(0)
T = 1.0

# Lagrange multiplier
theta = 1.0

# solution
u_sol = Function(Vg, name="Velocity")
p_sol = Function(Q, name="Pressure")
B_sol = Function(Vg, name="MagneticField")

#initial condition
ux = y **2 * (1 - y) * z0 **2 * (1 - z0)
uy = x**2 * (1-x) * z0 **2 
uz = x **2 * (1-x) * y **2 * (1-y)

Bx = y **2 * (1 - z0) **2 
By = (1-x) **2 * z0 ** 2
Bz = x **2 * (1-y) ** 2

u_ex = as_vector([ux, uy, uz]) 
B_ex = as_vector([Bx, By, Bz]) 

z1_prev.sub(0).interpolate(u_ex)
z1_prev.sub(2).interpolate(B_ex)

gamma = Constant(100)
nu = Constant(0.01)
eta = Constant(0.01)
S = Constant(1)

# u1 p1 B1
F1 =(
        #u
      inner((u1 - u1p)/dt, u1t) * dx
    + nu * inner(grad(u1), grad(u1t)) * dx
    + inner(grad(p1), u1t) * dx
    #p
    + inner(u1, grad(p1t)) * dx
    # B
    + inner((B1 - B1p)/dt, B1t) * dx
    + eta * inner(curl(B1), curl(B1t)) * dx
    + gamma * inner(div(B1), div(B1t)) * dx
)

# u2 p2 B2
F2 =(
        #u
      inner(u2/dt, u2t) * dx
    + nu * inner(grad(u2), grad(u2t)) * dx
    + inner(dot(grad(u1p), u1p), u2t) * dx # F(u^n)
    + S * inner(cross(B1p, curl(B1p)), u2t) * dx
    
    + inner(grad(p2), u2t) * dx
    #p
    + inner(u2, grad(p2t)) * dx
    #B
    + inner(B2/dt, B2t) * dx
    + eta * inner(curl(B2), curl(B2t)) * dx
    - inner(curl(cross(u1p, B1p)), B2t) * dx
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

def compute_A(u2, B2):
    T1 = assemble(0.5 * inner(u2, u2) * dx) +assemble(S * 0.5 * inner(B2, B2) * dx) 
    T2 = theta/float(dt) + assemble(nu * inner(grad(u2), grad(u2)) * dx) 
    T3 = assemble(S * eta * inner(curl(B2), curl(B2))* dx)
    return T1 + T2 + T3

def compute_B(u1, u2, u1p, B1, B2, B1p):
    T1 = assemble(-inner(u1/dt, u2)* dx) - assemble(S/dt * inner(B1, B2) * dx) + assemble(1/dt * inner(u1p, u2) * dx) 
    T2 = - assemble(inner(dot(grad(u1p), u1p), u1) * dx) + assemble(S * inner(cross(B1p, curl(B1p)), u1) * dx) 
    T3 = assemble(S/dt * inner(B1p, B2)*dx) + assemble(S * inner(curl(cross(u1p, B1p)), B1) * dx)
    return T1 + T2 + T3

def compute_C(u1, u1p, p1p, B1, B1p, q):
    T1 = assemble(-0.5 * inner(u1 - u1p, u1 - u1p) * dx) -assemble(0.5 * S/dt * inner(B1-B1p, B1-B1p) * dx)  
    T2 = - theta * q ** 2/float(dt)
    return T1 + T2

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

def energy_uB(u, B):
    return 0.5 * assemble(inner(u, u) * dx) + assemble(0.5 * S * inner(B, B) * dx)

    

bcs = [

    DirichletBC(Z.sub(0), 0, "on_boundary"),
    DirichletBC(Z.sub(2), 0, "on_boundary"),
]

pb1 = NonlinearVariationalProblem(F1, z1, bcs)
solver1 = NonlinearVariationalSolver(pb1, solver_parameters = sp)

pb2 = NonlinearVariationalProblem(F2, z2, bcs)
solver2 = NonlinearVariationalSolver(pb2, solver_parameters = sp)

q_init = 0.0
pvd = VTKFile("output/drlm.pvd")
pvd.write(u_sol, p_sol, B_sol, time=float(t))

while (float(t) < float(T-dt)+1.0e-10):
    t.assign(t+dt)
    if mesh.comm.rank == 0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    solver1.solve()
    (u1, p1, B1) = z1.subfunctions
    
    solver2.solve()
    (u2, p2, B2) = z2.subfunctions
    
    # compute the coefficients
    A = compute_A(u2, B2)
    B = compute_B(u1, u2, u1p, B1, B2, B1p)
    C = compute_C(u1, u1p, p1p, B1, B1p, q_init)
      
    q = compute_root(A, B, C)
    q_const = Constant(q)
    u_sol.assign(u1 + q_const * u2)
    p_sol.assign(p1 + q_const * p2)
    B_sol.assign(B1 + q_const * B2)

    energy = energy_uB(u_sol, B_sol)

    if mesh.comm.rank == 0:
        print(RED % f"t={float(t)}, energy={energy}")
    pvd.write(u_sol, p_sol, B_sol, time=float(t))
    z1_prev.assign(z1)
    z2_prev.assign(z2)
    q_init = q

     
         

