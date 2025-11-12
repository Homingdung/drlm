# helicity preserving scheme

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

#(u, p, w, A, B)
Z = MixedFunctionSpace([Vg, Q, Vg, Vg, Vg])

# z1
z1 = Function(Z)
z1_test = TestFunction(Z)
z1_prev = Function(Z)

(u1, p1, w1, A1, B1) = split(z1)
(u1t, p1t, w1t, A1t, B1t) = split(z1_test)
(u1p, p1p, w1p, A1p, B1p) = split(z1_prev)

# z2
z2 = Function(Z)
z2_test = TestFunction(Z)
z2_prev = Function(Z)

(u2, p2, w2, A2, B2) = split(z2)
(u2t, p2t, w2t, A2t,  B2t) = split(z2_test)
(u2p, p2p, w2p, A2p, B2p) = split(z2_prev)

# time parameters
dt = Constant(0.02)
t = Constant(0)
T = 1.0

# Lagrange multiplier
theta = 100.0

# solution
u_sol = Function(Vg, name="Velocity")
p_sol = Function(Q, name="Pressure")
B_sol = Function(Vg, name="MagneticField")
w_sol = Function(Vg, name="Vorticity")
A_sol = Function(Vg, name="MagneticPotential")


#initial condition
#ux = y **2 * (1 - y) * z0 **2 * (1 - z0)
#uy = x**2 * (1-x) * z0 **2 
#uz = x **2 * (1-x) * y **2 * (1-y)

#Bx = y **2 * (1 - z0) **2 
#By = (1-x) **2 * z0 ** 2
#Bz = x **2 * (1-y) ** 2

#u_ex = as_vector([ux, uy, uz]) 
#B_ex = as_vector([Bx, By, Bz]) 
# initial condition, Mao-Xi-2025
def g(x):
    return 32 * x**3 * (x - 1) ** 3

phi_ex = as_vector([y*g(x)*g(y) * g(z0), -x*g(x)*g(y)*g(z0), g(x)*g(y)*g(z0)])
u_ex = curl(phi_ex)
A_ex = as_vector([10 * y*g(x) * g(y) * g(z0), -10 * x*g(x)*g(y)*g(z0), 10 * g(x)*g(y)*g(z0)])
P_ex = sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z0)
B_ex = curl(A_ex)
w_ex = curl(u_ex)

z1_prev.sub(0).interpolate(u_ex)    
z1_prev.sub(1).interpolate(P_ex)    
z1_prev.sub(2).interpolate(w_ex)    
z1_prev.sub(3).interpolate(A_ex)
z1_prev.sub(4).interpolate(B_ex)

gamma = Constant(100)
nu = Constant(0)
eta = Constant(0)
s = Constant(1)

# u1 p1, w1, A1, B1
F1 =(
        #u
      inner((u1 - u1p)/dt, u1t) * dx
    + nu * inner(curl(u1), curl(u1t)) * dx
    + inner(grad(p1), u1t) * dx
    + gamma * inner(div(u1), div(u1t)) * dx
    #p
    + inner(u1, grad(p1t)) * dx
    #w
    + inner(w1, w1t) * dx
    - inner(curl(u1), w1t) * dx
    # A
    + inner((A1 - A1p)/dt, A1t) * dx
    + eta * inner(curl(B1), A1t) * dx
    # B
    + inner(B1, B1t) * dx
    - inner(curl(A1), B1t) * dx
    + gamma * inner(div(B1), div(B1t)) * dx
)

# u2, p2, w2, A2, B2
F2 =(
        #u
      inner(u2/dt, u2t) * dx
    + nu * inner(curl(u2), curl(u2t)) * dx
    + inner(cross(w1p, u1p), u2t) * dx # Linearized term
    - s * inner(cross(curl(B1p), B1p), u2t) * dx # Linearized term
    + gamma * inner(div(u2), div(u2t)) * dx

    + inner(grad(p2), u2t) * dx
    #p
    + inner(u2, grad(p2t)) * dx
    #w
    + inner(w2, w2t) * dx
    - inner(curl(u2), w2t) * dx
    #A
    + inner(A2/dt, A2t) * dx
    + eta * inner(curl(B2), A2t) * dx
    - inner(cross(u1p, B1p), A2t) * dx #linearized term
    #B
    + inner(B2, B2t) * dx
    - inner(curl(A2), B2t) * dx
    + gamma * inner(div(B2), div(B2t)) * dx
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

def compute_A(u2, B2, u1p, w1p, B1p):
    T1 = assemble(0.5 * inner(u2, u2) * dx)/float(dt) + assemble(s * 0.5 * inner(B2, B2) * dx)/float(dt) 
    T2 = theta/float(dt) + assemble(nu * inner(curl(u2), curl(u2))*dx) 
    T3 = assemble(s * eta * inner(curl(B2), curl(B2)) * dx)
    return T1 + T2 + T3

def compute_B(u1, u2, u1p, B1, B2, B1p):
    T1 = assemble(-inner(u1/dt, u2)* dx) - assemble(s/dt * inner(B1, B2) * dx) 
    T2 = -assemble(inner(cross(w1p, u1p), u1)*dx) + assemble(s * inner(cross(curl(B1p), B1p), u1) * dx)
    T3 = assemble(s * inner(curl(cross(u1p, B1p)), B1)*dx) 
    return T1 + T2 + T3

def compute_C(u1, u1p, p1p, B1, B1p, q):
    T1 = assemble(-0.5/dt * inner(u1 - u1p, u1 - u1p) * dx) -assemble(0.5 * s/dt * inner(B1-B1p, B1-B1p) * dx)  
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
    return 0.5 * assemble(inner(u, u) * dx) + assemble(0.5 * s * inner(B, B) * dx)

def compute_cross_helicity(u, B):  
    return assemble(inner(u, B) * dx)

def compute_helicity(A, B):
    return assemble(inner(A, B) * dx)

def compute_div(u):
    return norm(div(u), "L2")


bcs = [
    DirichletBC(Z.sub(0), u_ex, "on_boundary"),
    DirichletBC(Z.sub(2), w_ex, "on_boundary"),
    DirichletBC(Z.sub(3), A_ex, "on_boundary"),
    DirichletBC(Z.sub(4), B_ex, "on_boundary"),
]

pb1 = NonlinearVariationalProblem(F1, z1, bcs)
solver1 = NonlinearVariationalSolver(pb1, solver_parameters = sp)

pb2 = NonlinearVariationalProblem(F2, z2, bcs)
solver2 = NonlinearVariationalSolver(pb2, solver_parameters = sp)

q_init = 1.0
pvd = VTKFile("output/drlm.pvd")
pvd.write(u_sol, p_sol, w_sol, A_sol, B_sol, time=float(t))

data_filename = "data.csv"
fieldnames = ["t", "energy", "helicity_m", "helicity_c", "divu", "divB"]
if mesh.comm.rank == 0:
    with open(data_filename, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()  


while (float(t) < float(T-dt)+1.0e-10):
    t.assign(t+dt)
    if mesh.comm.rank == 0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    solver1.solve()
    (u1, p1, w1, A1, B1) = z1.subfunctions
    
    solver2.solve()
    (u2, p2, w2, A2, B2) = z2.subfunctions
    
    # compute the coefficients
    A = compute_A(u2, B2, u1p, w1p, B1p)
    B = compute_B(u1, u2, u1p, B1, B2, B1p)
    C = compute_C(u1, u1p, p1p, B1, B1p, q_init)
      
    q = compute_root(A, B, C)
    q_const = Constant(q)

    u_sol.assign(u1 + q_const * u2)
    p_sol.assign(p1 + q_const * p2)
    B_sol.assign(B1 + q_const * B2)
    w_sol.assign(w1 + q_const * w2)
    A_sol.assign(A1 + q_const * A2)

    energy = energy_uB(u_sol, B_sol)
    helicity_c = compute_cross_helicity(u_sol, B_sol)
    helicity_m = compute_helicity(A_sol, B_sol)
    #helicity_f = compute_helicity(u_sol, w_sol)
    divu = compute_div(u_sol)
    divB = compute_div(B_sol)

    if mesh.comm.rank == 0:
        row = {
        "t": float(t),
        "energy": float(energy),
        "helicity_m": float(helicity_m),
        "helicity_c": float(helicity_c),
        "divu": float(divu),
        "divB": float(divB),
        }
        with open(data_filename, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames = fieldnames)
            writer.writerow(row)
    
    if mesh.comm.rank == 0:
        print(RED % f"t={float(t)}, energy={energy}, crossHelicity={helicity_c}, magneticHelicity={helicity_m}, divu={divu}, divB={divB}")
    pvd.write(u_sol, p_sol, w_sol, A_sol, B_sol, time=float(t))

    z1_prev.assign(z1)
    z2_prev.assign(z2)
    q_init = q

     
         

