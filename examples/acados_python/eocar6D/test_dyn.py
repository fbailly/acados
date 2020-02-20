from acados_template import AcadosModel
from casadi import MX, Function, vertcat, mtimes, inv
import sys,os
from ctypes import *
import numpy as np
import biorbd as brbd
m        = brbd.Model("/home/fbailly/devel/models/simple.bioMod")

M = 10
I = np.eye(3)
Iinv = I

def so3(x):
  so3 = np.zeros((3,3))
  so3[0,1] = -x[2]
  so3[0,2] =  x[1]
  so3[1,0] =  x[2]
  so3[1,2] = -x[0]
  so3[2,0] = -x[1]
  so3[2,1] =  x[0]
  return so3



# Compute dynamics
def forw_dyn(x,u):
  accLin = u[:3]/M-so3(x[9:]).dot(x[6:9]) - np.array([[0],[0],[9.81]])
  accAng = Iinv.dot(u[3:]-so3(x[9:]).dot(I.dot(x[9:])))
  return np.vstack([accLin, accAng])

x0 = np.ones((12,1))
x0[3]  = 0
x0[4]  = 0
x0[5]  = 0
# x0[9]  = 0.5
# x0[10] = 0.5
# x0[11] = 0.5
u0 = np.zeros((6,1))
print(f"my forward dyn :\n {forw_dyn(x0,u0)}")
print(f"brbd forward dyn :\n {m.ForwardDynamics(x0[:6].squeeze(),x0[6:].squeeze(),u0.squeeze()).to_array()}")
print()