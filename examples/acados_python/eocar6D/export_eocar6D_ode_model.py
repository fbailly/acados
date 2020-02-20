from acados_template import AcadosModel
from casadi import MX, Function, vertcat, mtimes, inv
import sys,os
from ctypes import *
import numpy as np


def export_eocar6D_ode_model():

    model_name = 'eocar6D_ode'
    M = 1
    I = MX.eye(3)
    Iinv = I
    def so3(x):
      so3 = MX(3,3)
      so3[0,1] = -x[2]
      so3[0,2] =  x[1]
      so3[1,0] =  x[2]
      so3[1,2] = -x[0]
      so3[2,0] = -x[1]
      so3[2,1] =  x[0]
      return so3


    # Declare model variables
    x = MX.sym('x', 12)
    u = MX.sym('u', 6)
    xDot = MX.sym('xDot', 12)

    # Compute dynamics
    accLin = u[:3]/M-mtimes(so3(x[9:]), x[6:9]) - np.array([0,0,M*9.81])
    accAng = mtimes(Iinv, u[3:]-mtimes(so3(x[9:]), mtimes(I, x[9:])))
    acc    = vertcat(accLin, accAng)
    f_cas = Function('f_cas', [x, u], [vertcat(x[6:], acc)])

    f_expl = f_cas(x, u)

    f_impl = xDot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xDot
    model.u = u
    model.p =[]
    model.name = model_name

    return model 

