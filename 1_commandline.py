
# coding: utf-8

# In[ ]:

import numpy as np
import sympy
import fipy as fp
import matplotlib.pyplot as plt
import os
import sys

c, rho_s, c_alpha, c_beta = sympy.symbols("c_var rho_s c_alpha c_beta")
f_0 = rho_s * (c - c_alpha)**2 * (c_beta - c)**2

sympy.diff(f_0, c, 2)

# command format:
# python 1a_commandline.py 100 2.0 a
# where nx = 100
# and dx = 2.0
# for benchmark problem 1a
nx = int(sys.argv[1])
dx = float(sys.argv[2])
problem = sys.argv[3]

if (problem == "a"):
    mesh = fp.PeriodicGrid2D(nx=nx, ny=nx, dx=dx, dy=dx)
elif (problem == "b"):
    mesh = fp.Grid2D(nx=nx, ny=nx, dx=dx, dy=dx)
elif (problem == "c"):
    mesh = fp.Grid2D(dx=0.5, dy=0.5, nx=40, ny=200) + (fp.Grid2D(dx=0.5, dy=0.5, nx=200, ny=40) + [[-40],[100]])

c_alpha = 0.3
c_beta = 0.7
kappa = 2.0
M = 5.0
c_0 = 0.5
epsilon = 0.01
rho_s = 5.0

# solution variable
c_var = fp.CellVariable(mesh=mesh, name=r"$c$", hasOld=True)

# array of sample c-values: used in f versus c plot
vals = np.linspace(-.1, 1.1, 1000)

x , y = np.array(mesh.x), np.array(mesh.y)

# initial value for square and T domains
c_var[:] = c_0 + epsilon * (np.cos(0.105 * x) * np.cos(0.11 * y) + (np.cos(0.13 * x) * np.cos(0.087 * y))**2 + np.cos(0.025 * x - 0.15 * y) * np.cos(0.07 * x - 0.02 * y))

# bulk free energy density
def f_0(c):
    return rho_s*((c - c_alpha)**2)*((c_beta-c)**2)
def f_0_var(c_var):
    return 2*rho_s*((c_alpha - c_var)**2 + 4*(c_alpha - c_var)*(c_beta - c_var) + (c_beta - c_var)**2)
# free energy
def f(c):
    return (f_0(c)+ .5*kappa*(c.grad.mag)**2)

# plot free energy density versus c
def plotf_c():
    plt.figure(1)
    plt.xlabel('c')
    plt.ylabel('f_0')
    plt.plot(vals, f_0(vals))
    plt.show()
    
# save elapsed time and free energy at each data point
time_data = []
cvar_data = []
f_data = []
# checks whether a folder for the pickles from this simulation exists
# if not, creates one in the home directory
file_dir = "/data/and9/surf-research/Anushka/1{0}{1}".format(problem, nx)
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
file_name = "/data/and9/surf-research/Anushka/1{0}{1}/1{0}{1}".format(problem, nx)

def save_data(time, cvar, f):
    time_data.append(time)
    cvar_data.append(np.array(cvar.value))
    f_data.append(f.value)
    
    np.savez(file_name, time = time_data, c_var = cvar_data, f = f_data)

# solver equation    
eqn = fp.TransientTerm(coeff=1.) == fp.DiffusionTerm(M * f_0_var(c_var)) - fp.DiffusionTerm((M, kappa))

elapsed = 0.0
steps = 0
dt = 0.01
total_sweeps = 2
tolerance = 1e-1

# controls on how long the simulation runs: steps, duration, or both
total_steps = 20
duration = 4600

c_var.updateOld()
from fipy.solvers.pysparse import LinearLUSolver as Solver
solver = Solver()
print "Starting Solver."
while elapsed <= duration:
    res0 = eqn.sweep(c_var, dt=dt, solver=solver)
    #record the volume integral of the free energy 
    # equivalent to the average value of the free energy for any cell,
    # multiplied by the number of cells and the area of each cell
    # (since this is a 2D domain)
       
    
    for sweeps in range(total_sweeps):
        res = eqn.sweep(c_var, dt=dt, solver=solver)
        
    if res < res0 * tolerance:
          
        # anything in this loop will only be executed 100 times
        if ((steps%10)==0):
            print steps
            print elapsed
            print "Saving data"
            save_data(elapsed, c_var, f(c_var).cellVolumeAverage*mesh.numberOfCells*(dx**2))
            
        steps += 1
        elapsed += dt
        dt *= 1.1
        c_var.updateOld()
    else:
        dt *= 0.8
        c_var[:] = c_var.old

# simulation ends
print 'steps reached:', steps


