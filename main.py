import numpy as np
import scipy as sc
import scipy.interpolate
import matplotlib.pyplot as plt
import math
from shapely.geometry import Polygon
import pygmsh
import pyvista as pv
import dolfinx.plot as plot
import dolfinx
from mpi4py import MPI
import meshio
import sys
import gmsh
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from pathlib import Path
import ufl

def in2m(x):
    return 0.0254*x

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
    "v": 0.33, # Poisson's Ratio
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

    # Extract the x and y coordinates of the vertices
    x, y = vertices[:, 0], vertices[:, 1]

    # Plot the mesh using matplotlib
    plt.figure(figsize=(8, 6))
    plt.triplot(x, y, cells)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Pygmsh Mesh")
    plt.gca().set_aspect("equal")
    plt.show()

class fem:
    def __init__(self):
        self.flag = 0
    def define_function_space():
        # Define the second-order Lagrange (quadratic) element
        element = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)

        # Create the vector-valued function space
        V = dolfinx.FunctionSpace(mesh, element)

        # Define the test and trial functions on the function space
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

def fem_setup(file):
    xdmf_file = dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'meshes/mesh.xdmf', "r")
    mesh = xdmf_file.read_mesh(name="Grid")
    mesh.topology.create_connectivity_all()


pts = [0.1, -0.2, 0.3, -0.4, 0.5]
mesh, pymesh = generate_flexure_mesh(pts)


