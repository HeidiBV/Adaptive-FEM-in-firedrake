#%%
from firedrake import inner,grad,dx,ds,TrialFunction, \
    TestFunction,solve,FacetNormal,avg,jump,CellDiameter,CellVolume, \
    sqrt,dS,assemble,dot,UnitSquareMesh,FunctionSpace,DirichletBC, \
    SpatialCoordinate,Mesh,Function,tripcolor,triplot,trisurf, \
    And,Or,conditional,pi,sin,cos,div

import firedrake as fd
import cmasher as cmr
import pygmsh

from matplotlib import pyplot as plt, animation
from matplotlib.gridspec import GridSpec
from IPython.display import Image
import numpy as np
from matplotlib.path import Path
from scipy.interpolate import griddata

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 14, 'figure.titlesize': 16,'figure.dpi': 75})

#%% title Defining the bilinear and linear form
def a(u, v, c):
    return (inner(grad(u), grad(v)) + inner(c*u, v)) * dx

def L(v, f, g, neumann_sides):
    l = 0 if f is None else inner(f, v) * dx
    l += 0 if g is None else inner(g, v) * ds(neumann_sides)
    return l

def solve_system(a, L, bc, k, c, V, x, y, f = None, g = None, neumann_sides = None):
    u = TrialFunction(V)
    v = TestFunction(V)

    uh = Function(V)
    solve(a(u,v,c) == L(v, f, g, neumann_sides), uh, solver_parameters={'ksp_type': 'cg', 'pc_type': 'lu'}, bcs = [bc])
    return uh
#%% title Definition of f and u

"""We test our solution by creating an f that solves u exactly."""

def test_fun(x,y):
    c = 1
    return -2*((x-1)*x+(y-1)*y)+c*(x*y*(x-1)*(y-1))

def u_exact_fun(x,y):
    return x*y*(x-1)*(y-1)

#%% title Global indicator $\eta$
def get_eta_global(mesh, u, c, f = None, g = None, neumann_sides = None):
    n = FacetNormal(mesh) # normal vector on each edge
    h = CellDiameter(mesh) # diameter of each element

    r = h * (- c*u + div(grad(u)) + (0 if f is None else f))
    R = - sqrt(avg(h)) * jump(grad(u), n)

    integrand = r**2 * dx + R**2 * dS

    if g is not None:
        R_neumann = sqrt(h) * (g - dot(grad(u), n))
        integrand += R_neumann**2 * ds(neumann_sides)

    eta = assemble(integrand)
    return eta

#%% title Sharp and Relaible
def sharp(N,k,f,exact_fun):
    c = 1
    mesh = UnitSquareMesh(N,N)

    V = FunctionSpace(mesh, "CG", k)
    bc = DirichletBC(V, 0, "on_boundary")
    x, y = SpatialCoordinate(mesh)

    f = None if f is None else Function(V).interpolate(f(x,y))

    uh = solve_system(a, L, bc, k, c, V, x, y, f = f)

    e_grad = grad(exact_fun(x,y)-uh)
    error_energy = np.sqrt(assemble(e_grad**2*dx))

    a_post_err = np.sqrt(get_eta_global(mesh, uh, c, f = f))

    return uh,error_energy,a_post_err

def convergence(k,test_fun,exact_fun):
  n_vals = np.array([10,20,40,80])
  a_post_err_arr = np.zeros(len(n_vals))
  error_arr = np.zeros(len(n_vals))

  for i,N in enumerate(n_vals):
      _,error,a_post_err = sharp(N,k,test_fun,exact_fun)
      a_post_err_arr[i] = a_post_err
      error_arr[i] = error

  return n_vals,a_post_err_arr,error_arr

def plot_convergence(test_fun,exact_fun):
    n_k = 3
    vals_apost = np.zeros((n_k,4))
    vals_error = np.zeros((n_k,4))
    n_vals = np.array([10,20,40,80])

    for i in range(n_k):
        _,_,vals_apost[i,:] = convergence(i+1,test_fun,exact_fun)
        _,vals_error[i,:],_ = convergence(i+1,test_fun,exact_fun)

    plt.figure(figsize=(20,6))
    for i in range(n_k):
        rate_a_post = np.polyfit(np.log(n_vals),np.log(vals_apost[i]),1)
        rate_error = np.polyfit(np.log(n_vals),np.log(vals_error[i]),1)
        plt.subplot(1,3,i+1)
        plt.loglog(n_vals,vals_apost[i],label=f'a posteriori error rate {-np.round(rate_a_post[0],2)}')
        plt.loglog(n_vals,vals_error[i],label=f'energy norm error {-np.round(rate_error[0],2)}')
        plt.title(f'convergence for k = {i+1}')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
        plt.grid('on')
    plt.show()
#%% convergence plot
plot_convergence(test_fun,u_exact_fun)

#%% title OurMesh class

class OurMesh:
    """
    A wrapper for all the data concerning the mesh
    Attributes:
        filename (str): The name of the file where the mesh is saved.
        vertices (list of list): The list of vertices defining the domain polygon.
        mesh_size (float): The global mesh size.
        mesh (firedrake.Mesh): The generated mesh.
        refined_points (dict): A dictionary with points as keys and their refinement level (h_refine) as values.

    Methods:
        refine_mesh(self, points, h_refine):
            Refines the mesh around specific points with the given refinement level.
        plot(self):
            Plots the generated mesh using firedrake's triplot function.
    """
    def __init__(self,
                 filename = "mesh.msh",
                 mesh_size = 0.1,
                 vertices = [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    ]):

        self.filename = filename
        self.vertices = vertices
        self.mesh_size = mesh_size

        with pygmsh.geo.Geometry() as geom:
            poly = geom.add_polygon(vertices, mesh_size = mesh_size)
            # add labels
            for i, edge in enumerate(poly.curves):
                geom.add_physical(edge, label=f"{i+1}")
            geom.add_physical(poly.surface, label="0")

            geom.generate_mesh(dim=2)
            pygmsh.write(self.filename)

        self.mesh = Mesh(self.filename)
        #dict with point as key, refined_h as value
        self.refined_points = {}
        #min/max x and y of the polygon
        vertices_array = np.array(self.vertices)
        self.min_x, self.max_x = np.min(vertices_array[:,0]), np.max(vertices_array[:,0])
        self.min_y, self.max_y = np.min(vertices_array[:,1]), np.max(vertices_array[:,1])

    def refine_mesh(self, points, h_refine):
        with pygmsh.geo.Geometry() as geom:
            poly = geom.add_polygon(self.vertices, mesh_size = self.mesh_size)

            # add labels
            for i, edge in enumerate(poly.curves):
                geom.add_physical(edge, label=f"{i+1}")
            geom.add_physical(poly.surface, label="0")

            # add points to dict with refined points
            for i in range(len(points)):
                self.refined_points[points[i]] = h_refine[i]

            # add refined points to the mesh
            for point in self.refined_points:
                added_point = geom.add_point(point, self.refined_points[point])
                geom.in_surface(added_point, poly.surface)

            geom.generate_mesh()
            pygmsh.write(self.filename)
            self.mesh = fd.Mesh(self.filename)

    def polygon_mask(self,nx,ny):
        x = np.linspace(self.min_x, self.max_x, nx)
        y = np.linspace(self.min_y, self.max_y, ny)
        x, y = np.meshgrid(x, y)
        x, y = x.flatten(), y.flatten()

        points = np.vstack((x,y)).T

        path = Path(self.vertices)
        mask = path.contains_points(points).reshape((ny,nx))
        return mask

    def _plot_refinement(self, ax, cmap):
        """
        x, y = zip(*self.refined_points.keys())
        nbins=300
        k = gaussian_kde([x,y])
        xi, yi = np.mgrid[self.min_x:self.max_x:nbins*1j, self.min_y:self.max_y:nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        zi = np.ma.masked_where(~self.polygon_mask(len(xi), len(yi)).ravel(), zi, copy=False)

        # Make the plot
        ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=cmap)

        ######
        """
        f_x, f_y = get_element_coordinates(self.mesh)
        DG0 = FunctionSpace(self.mesh, "Discontinuous Lagrange", 0)
        h = Function(DG0).interpolate(CellDiameter(self.mesh)).dat.data

        grid_x, grid_y = np.mgrid[self.min_x:self.max_x:200j, self.min_y:self.max_y:200j]
        Z = griddata(list(zip(f_x, f_y)), h, (grid_x, grid_y), method='nearest')
        mask = self.polygon_mask(len(grid_x), len(grid_y))
        Z[~mask] = np.nan
        ax.contourf(grid_x, grid_y, Z, cmap=cmap)

    def plot(self, ax=None, plot_refined_points = True, cmap = cmr.bubblegum, figsize=(10,10)):
        if ax is None: fig, ax = plt.subplots(1,1,figsize=figsize)
        if plot_refined_points and self.refined_points!={}: self._plot_refinement(ax, cmap)
        triplot(self.mesh, axes=ax)
        plt.gca().set_aspect("equal")
        if ax is None: plt.show()

#%% title Get element coordinates
def get_element_coordinates(mesh):
    DG0 = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    x,y = SpatialCoordinate(mesh)
    f_x = Function(DG0).interpolate(x).dat.data
    f_y = Function(DG0).interpolate(y).dat.data
    return f_x, f_y

#%% title Plotting function for the solution and the mesh
def plot_solution_and_mesh(u, mesh,angles = (30,20) ,plot_refined_points = True, plot="surface", data = None, cmap1=cmr.tropical, cmap2 = cmr.get_sub_cmap("cmr.gothic", 0.4, 0.95), cmap3 = cmr.get_sub_cmap("cmr.swamp_r", 0.05, 0.75)):
    # make figure
    if isinstance(data, tuple): #animate
        fig = plt.figure(figsize=(14,14))
        gs = GridSpec(2, 2)
        local_etas, f_xs, f_ys, meshes, global_etas = data
        animate = True
    else:
        fig = plt.figure(figsize=(14,14))
        gs = GridSpec(2, 2)
        global_etas = data
        animate = False

    iterations = len(global_etas)
    # plotting the solution
    if plot=="surface":
        ax0 = fig.add_subplot(gs[0,0], projection='3d')
        trisurf(u, axes = ax0, cmap=cmap1)
        ax0.view_init(angles[0],angles[1])
    else:
        ax0 = fig.add_subplot(gs[0,0])
        tripcolor(u, axes = ax0, cmap=cmap1)
    ax0.set_title("Numerical solution")

    # plotting the mesh
    ax1 = fig.add_subplot(gs[0,1])
    mesh.plot(ax1, plot_refined_points, cmap = cmap2)
    ax1.set_title("Final mesh")

    # plotting global etas
    ax3 = fig.add_subplot(gs[1,1] if animate else gs[1,:])
    ax3.plot(np.arange(iterations), global_etas, label=r"$\eta^2$")
    ax3.plot(np.arange(iterations), [np.sum(local_eta) for local_eta in local_etas], '--', label=r"$\sum_i \eta_i^2$")
    plt.legend()
    ax3.set_title(r"Global $\eta^2$")

    # animating local etas
    if animate:
        ax2 = fig.add_subplot(gs[1,0])
        ax2.set_xlim([mesh.min_x, mesh.max_x])
        ax2.set_ylim([mesh.min_y, mesh.max_y])
        #vmin = np.min(local_etas[0])
        #vmax = np.max(local_etas[0])

        grid_x, grid_y = np.mgrid[mesh.min_x:mesh.max_x:200j, mesh.min_y:mesh.max_y:200j]

        def animate(i):
            f_x, f_y = f_xs[i], f_ys[i]

            Z = griddata(list(zip(f_x, f_y)), local_etas[i], (grid_x, grid_y), method='nearest')
            mask = mesh.polygon_mask(len(grid_x), len(grid_y))
            Z[~mask] = np.nan

            ax2.contourf(grid_x, grid_y, Z, cmap=cmap3) #, vmin=vmin, vmax=vmax)

            triplot(meshes[i], axes = ax2)
            plt.gca().set_aspect("equal")
            ax2.set_title(rf"Local $\eta_i^2$, iteration {i}")

        plt.tight_layout()
        anim = animation.FuncAnimation(fig, animate, interval=1500, frames=iterations)
        anim.save('fd_sol.gif', writer='pillow')
        #Displays animation
        with open('fd_sol.gif','rb') as file:
            display(Image(file.read()))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

#%% title Localization of indicator
def get_eta_local(mesh, u, c, f = None, g = None, neumann_sides = None):
    DG0 = fd.FunctionSpace(mesh, "Discontinuous Lagrange", 0)

    n = FacetNormal(mesh) # normal vector on each facet/edge
    h = CellDiameter(mesh) # diameter of each element
    area = Function(DG0).interpolate(CellVolume(mesh)).dat.data # area of each triangle

    # define r and R

    r = h * (- c*u + div(grad(u)) + (0 if f is None else f))
    R = - avg(sqrt(h)) * jump(grad(u), n)

    # project R into DG0

    R2_DG0 = Function(DG0)
    X = TestFunction(DG0)
    Eq = R2_DG0 * X * dx - R**2 * avg(X) * dS 
    if g is not None:
        R_neumann = sqrt(h) * (g - dot(grad(u), n))
        Eq -= R_neumann**2 * X * ds(neumann_sides)

    solve(Eq==0, R2_DG0, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    # project r into DG0
    r2_DG0 = fd.Function(DG0).interpolate(r**2)

    # calculating the norms by integration (multiply by area) and taking the square root
    eta_local = (r2_DG0.dat.data + R2_DG0.dat.data) * area

    return eta_local
#%% title Deciding which points to refine, and by how much
def get_points_to_refine(mesh, eta_local, local_tol = 0.1, h_min = 0.01):
    # the points we want to refine are those above the tolerance, which have not
    # yet reached the minimum element size
    DG0 = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    h = Function(DG0).interpolate(CellDiameter(mesh)).dat.data
    indices_to_refine = (eta_local > local_tol) * (h > h_min)

    if np.sum(indices_to_refine) == 0: #no points to refine, return empty lists early
        return [], []

    h = h[indices_to_refine]

    f_x, f_y = get_element_coordinates(mesh)
    points_to_refine_x = f_x[indices_to_refine]
    points_to_refine_y = f_y[indices_to_refine]

    points_to_refine = list(zip(points_to_refine_x, points_to_refine_y))

    # deciding to what degree each point should be refined
    max_factor = 0.9 # minimum refinement factor (new h = factor*h)
    min_factor = 0.1 # maximum refinemnet factor

    if len(points_to_refine) == 1: # only one point to refine
        mesh_refinements = np.maximum(h*(max_factor + min_factor)/2, h_min) # refining with the average of max_factor and min_factor
        return points_to_refine, mesh_refinements

    etas_inv = 1/eta_local[indices_to_refine]
    # we now transform the factors to the interval [min_factor, max_factor]
    factors = min_factor + (max_factor-min_factor)/(np.max(etas_inv)-np.min(etas_inv)) * (etas_inv - np.min(etas_inv))
    mesh_refinements = np.maximum(factors * h, h_min) # h cannot be smaller than h_min

    return points_to_refine, mesh_refinements

#%% title Adaptive solver
def adaptive_solver(a, L, mesh, c, k, f = None, g = None, neumann_sides = None, improvement_tol = 0.9, maxiter = 10, DirichletBC_where = "on_boundary", h_min = 0.01, animate=True, improvement_factor=0.1):
    iter = 0
    improvement = 0
    points_to_refine = []
    global_etas = []
    if animate:
        local_etas, f_xs, f_ys, meshes = [], [], [], []
    while (iter <= maxiter and len(points_to_refine) != 0 and improvement < improvement_tol) or iter == 0:
        if iter > 0: mesh.refine_mesh(points_to_refine, mesh_refinements)

        # define new function space and interpolate f
        V = FunctionSpace(mesh.mesh, "CG", k)
        bc = DirichletBC(V, 0, DirichletBC_where)
        x, y = SpatialCoordinate(mesh.mesh)
        F = None if f is None else Function(V).interpolate(f(x,y))
        G = None if g is None else Function(V).interpolate(g(x,y))

        # solve system over new mesh
        u  = solve_system(a, L, bc, k, c, V, x, y, F, G, neumann_sides)

        # refine the mesh
        eta_global = get_eta_global(mesh.mesh, u, c, F, G, neumann_sides)
        eta_local = get_eta_local(mesh.mesh, u, c, F, G, neumann_sides)
        global_etas.append(eta_global)

        if animate:
            f_x, f_y = get_element_coordinates(mesh.mesh)
            local_etas.append(eta_local)
            f_xs.append(f_x)
            f_ys.append(f_y)
            meshes.append(mesh.mesh)

        if iter == 0:
            eta_global0 = eta_global
            eta_global_goal = eta_global0 * improvement_factor
        else:
            improvement = global_etas[iter]/global_etas[iter-1]

        local_tol = eta_global_goal / mesh.mesh.num_cells()
        points_to_refine, mesh_refinements = get_points_to_refine(mesh.mesh, eta_local, local_tol, h_min) 
        iter += 1

    if animate:
        data = (local_etas, f_xs, f_ys, meshes, global_etas)
        return u, data
    else:
        return u, global_etas


#%% title Example 1: Discontinuous function
"""# Adaptive solving over the unit square"""

def f(x,y):
    cond1 = And(And(x >= 0.45, x <= 0.55), And(y >= 0.45, y <= 0.55))
    cond2 = Or(And(And(x >= 0.4, x < 0.45), And(y >= 0.4, y < 0.45)), And(And(x > 0.55, x <= 0.6), And(y > 0.55, y <= 0.6)))
    return conditional(cond1, 10, conditional(cond2, 1, 0))

# initialize mesh and parameters
mesh = OurMesh(mesh_size=0.2)
c = 1 #@param
k = 1 #@param
maxiter = 4 #@param
h_min = 0.01 #@param
improvement_factor = 0.1 #@param

# solve
u, data = adaptive_solver(a, L, mesh, c, k, f = f, h_min=h_min, maxiter=maxiter, improvement_factor = improvement_factor)

plot_solution_and_mesh(u, mesh, data = data)
#%% title Example 2: Sinusoidal function
def f(x, y):
    return sin(16*pi*x)*sin(2*pi*y)

# initialize mesh and parameters
mesh = OurMesh(mesh_size=0.2)
c = 1 #@param
k = 1 #@param
maxiter = 4 #@param
h_min = 0.01 #@param
improvement_factor = 0.1 #@param

# solve
u, data = adaptive_solver(a, L, mesh, c, k, f=f,
                          h_min=h_min, maxiter=maxiter,
                          improvement_factor = improvement_factor)

plot_solution_and_mesh(u, mesh, data = data)
#%% title Solving the problem
"""# L-shaped domain"""
def g(x,y):
    factor = 1/(sqrt((x-1)**2+(y-1)**2))
    return factor * 2*pi*cos(2*pi*sqrt((x-1)**2+(y-1)**2))

neumann_sides = (1,2,5,6)

# initialize mesh and parameters
mesh = OurMesh(vertices = [[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]],mesh_size=0.3)
c = 1 #@param
k = 1 #@param
maxiter = 4 #@param
h_min = 0.01 #@param
improvement_factor = 0.1 #@param

# solve
u, data  = adaptive_solver(a, L, mesh, c, k, g = g, neumann_sides = neumann_sides,
                                    DirichletBC_where=[3,4], h_min = h_min, maxiter=maxiter,
                                    improvement_factor = improvement_factor)

plot_solution_and_mesh(u, mesh, data=data)
