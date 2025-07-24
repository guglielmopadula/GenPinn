import numpy as np
import ufl
from petsc4py import PETSc
from tqdm import trange
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner
from dolfinx.io import XDMFFile
from mpi4py import MPI
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from petsc4py.PETSc import ScalarType
import time
from dolfinx.fem import FunctionSpace
import pyvista
from mpi4py import MPI
from dolfinx import mesh
import meshio
import basix
from basix import CellType, ElementFamily, LagrangeVariant
from pygem import FFD
latent_ffd=np.load("latent_ffd.npy")

for i in trange(600):
    rabbit=meshio.read("data/Stanford_Bunny_3d.mesh")
    nodes=rabbit.points
    parameter=latent_ffd[i]
    ffd=FFD([3,3,3])
    parameter_x=parameter[:27]
    parameter_y=parameter[27:54]
    parameter_z=parameter[54:]
    ffd.array_mu_x=parameter_x.reshape(3,3,3)
    ffd.array_mu_y=parameter_y.reshape(3,3,3)
    ffd.array_mu_z[:,:,1:]=parameter_z.reshape(3,3,2)
    nodes=ffd(nodes)
    elem=rabbit.cells_dict["tetra"]
    t = 0  # Start time
    T = 1  # End time
    num_steps = 100  # Number of time steps
    dt = (T - t) / num_steps  # Time step size
    gdim = 3
    shape = "tetrahedron"
    degree = 1
    lagrange = basix.create_element(
        ElementFamily.P, CellType.tetrahedron, 1)
    rabbit_mesh=mesh.create_mesh(MPI.COMM_WORLD,elem,nodes,lagrange)
    V = fem.functionspace(rabbit_mesh, ("Lagrange", 1))
    xdmf = io.XDMFFile(rabbit_mesh.comm, "data/full_param_{}.xdmf".format(i), "w")
    xdmf.write_mesh(rabbit_mesh)
    def initial_condition(x):
        return 0*x[0]+0.5

    class BC:
        def __init__(self,  t):
            self.t = t

        def __call__(self, x):
            return 0.5 + 0*x[0] + 0.5 * self.t

    u_bc = fem.Function(V)
    bc_fun=BC(0.)
    u_bc.interpolate(bc_fun)
    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(initial_condition)
    # Create boundary condition
    fdim = rabbit_mesh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        rabbit_mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(u_bc, fem.locate_dofs_topological(V, fdim, boundary_facets))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - u_n  * v * ufl.dx
    a = fem.form(ufl.lhs(F))
    L = fem.form(ufl.rhs(F))
    A = assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = create_vector(L)
    uh = fem.Function(V)
    solver = PETSc.KSP().create(rabbit_mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    for n in trange(num_steps):
        # Update Diriclet boundary condition
        bc_fun.t += dt
        u_bc.interpolate(bc_fun)
        # Update the right hand side reusing the initial vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, L)
        # Apply Dirichlet boundary condition to the vector
        apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])
        # Solve linear problem
        solver.solve(b, uh.vector)
        uh.x.scatter_forward()
        # Update solution at previous time step (u_n)
        u_n.x.array[:] = uh.x.array
        xdmf.write_function(uh, bc_fun.t)
    xdmf.close()