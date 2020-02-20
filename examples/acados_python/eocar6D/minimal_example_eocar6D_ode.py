from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from export_eocar6D_ode_model import export_eocar6D_ode_model
import numpy as np
import os
import scipy.linalg
import matplotlib.pyplot as plt

os.environ["ACADOS_SOURCE_DIR"] = "/home/fbailly/miniconda3/envs/python_acados"

# create ocp object to formulate the OCP
ocp = AcadosOcp()

# set model
model = export_eocar6D_ode_model()
ocp.model = model
ocp.solver_options.model_external_shared_lib_dir     = os.getcwd()+"/test_external_lib/build"
ocp.solver_options.model_external_shared_lib_name    = "external_ode_casadi"

Tf = 2
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
ny_e = nx
N = 100
x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
xT = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# set dimensions
ocp.dims.nx    = nx
ocp.dims.ny    = ny
ocp.dims.ny_e  = ny_e
ocp.dims.nu    = nu
ocp.dims.N     = N

# set cost module
ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.cost_type_e = 'LINEAR_LS'

Q = 2*np.eye(nx)
R = 100*np.eye(nu)

ocp.cost.W = scipy.linalg.block_diag(Q, R)

ocp.cost.W_e = Q

ocp.cost.Vx = np.zeros((ny, nx))

Vu = np.zeros((ny, nu))
Vu[nx:,:] = 1.0
ocp.cost.Vu = Vu

ocp.cost.Vx_e = np.zeros((nx,nx))

ocp.cost.yref  = np.zeros((ny, ))
ocp.cost.yref_e = np.ones((ny_e, ))

# set constraints
Fmax = 10
ocp.constraints.x0 = x0
ocp.dims.nbx_0 = nx
ocp.constraints.constr_type = 'BGH'
ocp.constraints.lbu = -Fmax*np.ones(nu,)
ocp.constraints.ubu = Fmax*np.ones(nu,)
ocp.constraints.idxbu = np.array(range(nu))
ocp.dims.nbu   = nu

# terminal constraints
ocp.constraints.Jbx_e  = np.eye(nx)
ocp.constraints.ubx_e  = xT
ocp.constraints.lbx_e  = xT
ocp.constraints.idxbx_e = np.array(range(nx))
ocp.dims.nbx_e = nx


#path constraints
ocp.constraints.Jbx   = np.eye(nx)
ocp.constraints.ubx   = 2*np.ones(nx,)
ocp.constraints.lbx   = -2*np.ones(nx,)
ocp.constraints.idxbx = np.array(range(nx))
ocp.dims.nbx   = nx


ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI
# ocp.solver_options.qp_solver_iter_max = 1000
# ocp.solver_options.sim_method_newton_iter = 5
# ocp.solver_options.sim_method_num_stages = 4
# ocp.solver_options.sim_method_num_steps = 2
# ocp.solver_options.nlp_solver_max_iter = 200
# ocp.solver_options.nlp_solver_step_length = 1.0


# set prediction horizon
ocp.solver_options.tf = Tf
ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI

ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

# initial guess
t_traj = np.linspace(0, Tf, N+1)
x_traj = np.linspace(x0,xT,N+1)
u_traj = np.ones((N,6))+np.random.rand(N,6)*1e-6
for n in range(N+1):
  ocp_solver.set(n, 'x', x_traj[n,:])
for n in range(N):
  ocp_solver.set(n, 'u', u_traj[n])

simX = np.ndarray((N+1, nx))
simU = np.ndarray((N, nu))

status = ocp_solver.solve()

if status != 0:
    raise Exception('acados returned status {}. Exiting.'.format(status))

# get solution
# get solution
stat_fields = ['time_tot', 'time_lin', 'time_qp', 'time_qp_solver_call', 'time_reg', 'sqp_iter']
for field in stat_fields:
  print(f"{field} : {ocp_solver.get_stats(field)}")
for i in range(N):
    simX[i,:] = ocp_solver.get(i, "x")
    simU[i,:] = ocp_solver.get(i, "u")
simX[N,:] = ocp_solver.get(N, "x")

print(simX)
plt.plot(simX[:,:nu],'o',label='opt_sol')
# plt.plot(x_traj[:,:nu],'x',label='init_sol')
plt.legend()
plt.title('position')
plt.figure()
plt.plot(simX[:,nu:],'o',label='opt_sol')
# plt.plot(x_traj[:,nu:],'x',label='init_sol')
plt.legend()
plt.title('velocity')
plt.figure()
plt.plot(simU,'o',label='opt_sol')
# plt.plot(u_traj,'x',label='init_traj')
plt.legend()
plt.title('control')
plt.show(block=True)
print()