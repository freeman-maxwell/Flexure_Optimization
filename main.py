import numpy as np
import scipy as sc
import scipy.interpolate
import matplotlib.pyplot as plt
import math
import pygmsh
import dolfinx
from mpi4py import MPI
import ufl
from petsc4py.PETSc import ScalarType
import pyvista

def in2m(x):
    x_np = np.asarray(x)
    return 0.0254 * x_np

flexure = { # all units in m
    'total_length': in2m(2.5),
    'foot_length': in2m(0.37),
    'center_length': in2m(0.3125),
    'center_buffer_length': in2m(0.15),
    'thickness': in2m(1 / 40)
}

material = {
    # Aluminum H32-5052
    "E": 70.3E9, # Young's Modulus, in Pa
    "nu": 0.33, # Poisson's Ratio
    "G": 70.3E9 / (2 * (1 + 0.33)), # Shear Modulus
    "Yield": 193E6 #Yield Strength, in Pa
}


def generate_flexure_mesh(pts, mesh_density=0.01):

    def generate_flexure_function(pts, plot=False):
        flexure_length = flexure['total_length'] - flexure['foot_length'] - flexure['center_length'] - flexure['center_buffer_length']
        
        # Create the flexure from the chosen parameters
        dx = flexure_length / (len(pts) + 1)
        x_pts_flexure = np.append([flexure['foot_length']], np.linspace(flexure['foot_length'] + dx, flexure['foot_length'] + flexure_length, len(pts)))
        y_pts_flexure = np.append([0], pts)

        spline = sc.interpolate.CubicSpline(x_pts_flexure, y_pts_flexure, bc_type='clamped')

        def flexure_function(x_data):
            y = []
            for x in x_data:
                if 0 <= x <= flexure['foot_length']:
                    y.append(0)
                elif flexure['foot_length'] < x < flexure['foot_length'] + flexure_length:
                    y.append(spline(x))
                elif flexure['foot_length'] + flexure_length <= x:
                    y.append(pts[-1])
            return np.array(y)

        if plot:
            x = np.linspace(0, flexure['total_length'], 100)
            y = flexure_function(x)
            plt.plot(x, y)
            plt.title('Flexure Shape')
            plt.xlabel('X (in)')
            plt.ylabel('Y (in)')
            plt.show()

        return flexure_function


    def get_normals(x, y, length):
        x_n = []
        y_n = []
        for idx in range(len(x) - 1):
            x0, y0, xa, ya = x[idx], y[idx], x[idx + 1], y[idx + 1]
            dx, dy = xa - x0, ya - y0
            norm = math.hypot(dx, dy) * 1 / length
            dx /= norm
            dy /= norm
            x_n.append(x0 - dy)
            y_n.append(y0 + dx)
        return x_n, y_n


    def remove_loops(x_data, y_data, N=200):
        y_new = []
        x_new = np.linspace(0, flexure['total_length'], N)
        for index, x in enumerate(x_new):
            point_indices = []
            for i in range(len(x_data)-1):
                if (x_data[i] <= x <= x_data[i + 1]) or (x_data[i] >= x >= x_data[i + 1]):
                    point_indices.append(i)
            y = []
            for i in point_indices:
                m = (y_data[i+1] - y_data[i]) / (x_data[i+1] - x_data[i])
                y.append(m * (x - x_data[i]) + y_data[i])
            if index == N-1:
                y_new.append(y_new[-1])
            else:
                y_new.append(max(y))
        return x_new, y_new

    func = generate_flexure_function(pts, plot=False)

    x1 = np.array(np.linspace(0, flexure['total_length'], 200))
    y1 = func(x1)
    x2, y2 = get_normals(x1, y1, flexure['thickness'])
    x3, y3 = remove_loops(x2, y2, 200)

    bottom_edge = np.transpose([x1, y1])
    top_edge = np.transpose([x3, y3])[::-1]
    contour = np.concatenate((bottom_edge, top_edge))

    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(
            contour,
            mesh_size=mesh_density,
        )
        pymesh = geom.generate_mesh()

    # Extract the vertices and cells from the pygmsh mesh
    vertices = np.array(pymesh.points)[:,:2]
    cells = np.array(pymesh.cells_dict["triangle"])  # Assuming you have triangles

    ufl_tri = ufl.Mesh(ufl.VectorElement("Lagrange", ufl.triangle, 1))

    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, vertices, ufl_tri)

    return mesh, pymesh

def display_mesh(mesh):
    # Extract the vertices and cells from the pygmsh mesh
    vertices = np.array(mesh.points)
    cells = np.array(mesh.cells_dict["triangle"])  # Assuming you have triangles

    x = vertices[:, 0]
    y = vertices[:, 1]

    # Plot the mesh using matplotlib
    plt.figure(figsize=(8, 6))
    plt.triplot(x, y, cells)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Pygmsh Mesh")
    plt.gca().set_aspect("equal")
    plt.show()

def run_fea(mesh, order=2):

    # Create the vector-valued function space
    element = ufl.VectorElement('Lagrange',mesh.ufl_cell(),degree=1,dim=2)
    V = dolfinx.fem.FunctionSpace(mesh, element)

    # Define the test and trial functions on the function space
    du, dv = ufl.TestFunctions(V)
    u, v = ufl.TrialFunctions(V)
    x = ufl.SpatialCoordinate(mesh)

    # Identify points to be fixed (Dirichlet Boundary Condition)
    def fixed_end(x):
        return x[0] < flexure['foot_length']
    
    left_facets = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim - 1, fixed_end)
    dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, left_facets)

    # bc = dolfinx.fem.dirichletbc(value=ScalarType([0,0]), dofs=dofs, V=V)

    u_bc = dolfinx.fem.Function(V)
    with u_bc.vector.localForm() as loc:
        loc.set(0.0)

    bcs = [dolfinx.fem.dirichletbc(u_bc, dofs)]
        
    dx = ufl.Measure("dx",domain=mesh)
    right_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, lambda x : np.isclose(x[0], flexure['total_length']))
    mt = dolfinx.mesh.meshtags(mesh, 1, right_facets, 1)
    ds = ufl.Measure("ds", subdomain_data=mt)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    E = material["E"] 
    nu = material["nu"]
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    # this is for plane-stress
    lmbda = 2*mu*lmbda/(lmbda+2*mu)

    def eps(u):
        # Strain
        return ufl.sym(ufl.grad(u))

    def sigma(eps):
        # Stress
        return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(2)

    def a(u,v):
        # The bilinear form of the weak formulation
        return ufl.inner(sigma(eps(u)), eps(v)) * dx 

    def L(v): 
        # The linear form of the weak formulation
        # Volume force
        b = dolfinx.fem.Constant(mesh, ScalarType([0.0, 0.0]))

        # Surface force on the top
        f = dolfinx.fem.Constant(mesh,ScalarType([0.0, 0.0]))
        return ufl.dot(b, v) * dx + ufl.dot(f, v) * ds(1)
    
    problem = dolfinx.fem.petsc.LinearProblem(a(u,v), L(v), bcs=bcs, 
                                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    uh.name = "displacement"

    # Write result to png/XDMF file 
    pyvista.start_xvfb()
    from dolfinx.plot import create_vtk_mesh

    # Create plotter and pyvista grid
    p = pyvista.Plotter(off_screen=True)
    topology, cell_types, x = create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)

    # Attach vector values to grid and warp grid by vector

    vals = np.zeros((x.shape[0], 3))
    vals[:,:len(uh)] = uh.x.array.reshape((x.shape[0], len(uh)))
    grid["u"] = vals
    actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("u", factor=1.5)
    actor_1 = p.add_mesh(warped,opacity=0.8)
    p.view_xy()
    fig_array = p.screenshot(f"meshes/component.png")

    '''from pathlib import Path

    Path("output").mkdir(parents=True, exist_ok=True)    
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "meshes/result.xdmf", "w") as file:
            file.write_mesh(uh.function_space.mesh)
            file.write_function(uh)'''

pts = in2m([0.1, -0.2, 0.1])
mesh, pymesh = generate_flexure_mesh(pts, mesh_density=in2m(0.01))
# display_mesh(pymesh)
run_fea(mesh, order=2)

