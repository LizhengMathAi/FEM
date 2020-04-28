import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils


def example_0(n=16, num_refine=3):
    """
    u = f, \boldsymbol{x} \in \Omega

    check items:
    * FiniteElement.gram_p1
    * FiniteElement.integer_p1
    * LinearSystem.node_mul_node
    * LinearSystem.node_mul_func
    """

    def func_u(x):
        return np.cos(x[0] + x[1])

    func_f = func_u

    # Start to solve
    mesh = utils.SquareMesh(n=n)

    gram_tensor = mesh.gram_p1()
    mat = mesh.node_mul_node(gram_tensor)
    nn = mesh.vertices.__len__()
    mat = coo_matrix((mat.data, (mat.idx[0], mat.idx[1])), shape=(nn, nn))

    integer_tensor = mesh.integer_p1(func_f, num_refine=num_refine)
    rhs = mesh.node_mul_func(integer_tensor)

    coeff = spsolve(mat, rhs)

    # Check L2 error
    error = mesh.p1_error()
    print(error)

    # Show figure
    fig = plt.figure(figsize=(10, 6))

    xx = mesh.vertices[:, 0]
    yy = mesh.vertices[:, 1]
    u_val = func_u(mesh.vertices.T)
    u_h = coeff

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title("z = u_val(x, y)\nz = u_h(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[0]")
    ax.plot_trisurf(xx, yy, u_val, alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h, alpha=0.5)

    plt.show()


def example_1(n=16, num_refine=4):
    """
    u = f, \boldsymbol{x} \in \Omega

    check items:
    * FiniteElement.gram_s0
    * FiniteElement.integer_s0
    * LinearSystem.surface_mul_surface
    * LinearSystem.surface_mul_func
    """

    def func_u(x):
        return np.cos(x[0] + x[1])

    func_f = func_u

    # Start to solve
    mesh = utils.SquareMesh(n=n)

    gram_tensor = mesh.gram_s0()
    mat = mesh.surface_mul_surface(gram_tensor)
    nt = mesh.triangles.__len__()
    mat = coo_matrix((mat.data, (mat.idx[0], mat.idx[1])), shape=(nt, nt))

    integer_tensor = mesh.integer_s0(func_f, num_refine=num_refine)
    rhs = mesh.surface_mul_func(integer_tensor)

    coeff = spsolve(mat, rhs)

    # Check L2 error
    error = mesh.s0_error()
    print(error)

    # Show figure
    fig = plt.figure(figsize=(10, 6))

    centers = np.mean(mesh.tri_tensor, axis=1)
    xx = centers[:, 0]
    yy = centers[:, 1]
    u_val = func_u(centers.T)
    u_h = coeff

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title("z = u_val(x, y)\nz = u_h(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[0]")
    ax.plot_trisurf(xx, yy, u_val, alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h, alpha=0.5)

    plt.show()


def example_2(n=16, num_refine=3):
    """
    u = f, \boldsymbol{x} \in \Omega
    u = g, \boldsymbol{x} \in \partial \Omega

    check items:
    * IsotropicMesh.inner_node_ids
    * IsotropicMesh.bound_node_ids
    * FiniteElement.gram_p1
    * FiniteElement.integer_p1
    * LinearSystem.node_mul_node
    * LinearSystem.node_mul_func
    """

    def func_u(x):
        return np.cos(x[0] + x[1])

    func_f = func_u
    func_g = func_u

    # Start to solve
    mesh = utils.SquareMesh(n=n)

    gram_tensor = mesh.gram_p1()
    mat = mesh.node_mul_node(gram_tensor)

    mat_1 = mat[mesh.inner_node_ids, mesh.inner_node_ids]
    inner_nn = mesh.inner_node_ids.__len__()
    mat_1 = coo_matrix((mat_1.data, (mat_1.idx[0], mat_1.idx[1])), shape=(inner_nn, inner_nn))

    mat_2 = mat[mesh.inner_node_ids, mesh.bound_node_ids]
    bound_nn = mesh.bound_node_ids.__len__()
    mat_2 = coo_matrix((mat_2.data, (mat_2.idx[0], mat_2.idx[1])), shape=(inner_nn, bound_nn))

    bound_vertices = mesh.vertices[mesh.bound_node_ids]
    rhs_1 = func_g(bound_vertices.T)

    integer_tensor = mesh.integer_p1(func_f, num_refine=num_refine)
    rhs = mesh.node_mul_func(integer_tensor)
    rhs_2 = rhs[mesh.inner_node_ids]

    coeff = spsolve(mat_1, rhs_2 - mat_2@rhs_1)

    # Check L2 error
    error = mesh.p1_error()
    print(error)

    # Show figure
    fig = plt.figure(figsize=(10, 6))

    xx = mesh.vertices[:, 0]
    yy = mesh.vertices[:, 1]
    u_val = func_u(mesh.vertices.T)
    u_h = func_g(mesh.vertices.T)
    u_h[mesh.inner_node_ids] = coeff

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title("z = u_val(x, y)\nz = u_h(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[0]")
    ax.plot_trisurf(xx, yy, u_val, alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h, alpha=0.5)

    plt.show()


def example_3(n=16, num_refine=3):
    """
    -\Delta u = f, \boldsymbol{x} \in \Omega
    u = g, \boldsymbol{x} \in \partial \Omega

    check items:
    * IsotropicMesh.inner_node_ids
    * IsotropicMesh.bound_node_ids
    * FiniteElement.gram_p1
    * FiniteElement.integer_p1
    * LinearSystem.node_mul_node
    * LinearSystem.node_mul_func
    """

    def func_u(x):
        return np.cos(x[0] + x[1])

    func_f = func_u
    func_g = func_u

    # Start to solve
    mesh = utils.SquareMesh(n=n)

    gram_tensor = mesh.gram_grad_p1()
    mat = mesh.node_mul_node(gram_tensor)

    mat_1 = mat[mesh.inner_node_ids, mesh.inner_node_ids]
    inner_nn = mesh.inner_node_ids.__len__()
    mat_1 = coo_matrix((mat_1.data, (mat_1.idx[0], mat_1.idx[1])), shape=(inner_nn, inner_nn))

    mat_2 = mat[mesh.inner_node_ids, mesh.bound_node_ids]
    bound_nn = mesh.bound_node_ids.__len__()
    mat_2 = coo_matrix((mat_2.data, (mat_2.idx[0], mat_2.idx[1])), shape=(inner_nn, bound_nn))

    bound_vertices = mesh.vertices[mesh.bound_node_ids]
    rhs_1 = func_g(bound_vertices.T)

    integer_tensor = mesh.integer_p1(func_f, num_refine=num_refine)
    rhs = mesh.node_mul_func(integer_tensor)
    rhs_2 = rhs[mesh.inner_node_ids]

    coeff = spsolve(mat_1, rhs_2 - mat_2@rhs_1)

    # Check L2 error
    error = mesh.p1_error()
    print(error)

    # Show figure
    fig = plt.figure(figsize=(10, 6))

    xx = mesh.vertices[:, 0]
    yy = mesh.vertices[:, 1]
    u_val = func_u(mesh.vertices.T)
    u_h = func_g(mesh.vertices.T)
    u_h[mesh.inner_node_ids] = coeff

    extend_xx = np.mean(mesh.tri_tensor[:, :, 0], axis=1)
    extend_yy = np.mean(mesh.tri_tensor[:, :, 1], axis=1)
    extend_u_val = func_u(np.vstack([extend_xx, extend_yy]))
    extend_u_h = (u_h[mesh.triangles[:, 0]] + u_h[mesh.triangles[:, 1]] + u_h[mesh.triangles[:, 2]]) / 3

    xx = np.hstack([xx, extend_xx])
    yy = np.hstack([yy, extend_yy])
    u_val = np.hstack([u_val, extend_u_val])
    u_h = np.hstack([u_h, extend_u_h])

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title("z = u_val(x, y)\nz = u_h(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[0]")
    ax.plot_trisurf(xx, yy, u_val, alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h, alpha=0.5)

    plt.show()


def example_4(n=16, num_refine=3):
    """
    \boldsymbol{u} = \boldsymbol{f}, \boldsymbol{x} \in \Omega
    <\boldsymbol{u}, \boldsymbol{n}> = 0, \boldsymbol{x} \in \partial \Omega

    check items:
    * IsotropicMesh.inner_node_ids
    * FiniteElement.gram_p1
    * FiniteElement.gram_p1_rt0
    * FiniteElement.gram_rt0
    * FiniteElement.integer_p1
    * FiniteElement.integer_rt0
    * LinearSystem.node_mul_node
    * LinearSystem.node_mul_edge
    * LinearSystem.edge_mul_edge
    * LinearSystem.node_mul_func
    * LinearSystem.edge_mul_func
    """

    def func_u(x):
        ux = np.sin(np.pi * x[0]) * np.cos(x[0] + x[1])
        uy = np.sin(np.pi * x[1]) * np.cos(x[0] + x[1])
        return np.stack([ux, uy], axis=0)

    func_f = func_u

    # Start to solve
    class Mesh(utils.SquareMesh, utils.RT0):
        def __init__(self, n):
            super().__init__(n)
            self.build()

    mesh = Mesh(n)

    # list the nodes and edges
    node_ids = mesh.inner_node_ids
    nn = mesh.inner_node_ids.__len__()

    edge = [[i, i + mesh.n + 2] for i in range(mesh.n)]
    edge += [[(i + 1) * (mesh.n + 1) - 2, (i + 1) * (mesh.n + 1) - 2 + mesh.n + 2] for i in range(mesh.n)]
    edge += [[(mesh.n + 1) * (mesh.n - 1) + i, (mesh.n + 1) * (mesh.n - 1) + i + mesh.n + 2] for i in range(mesh.n)]
    edge += [[(i + 1) * (mesh.n + 1) - 2 - (mesh.n - 1), (i + 1) * (mesh.n + 1) + 1] for i in range(mesh.n)]

    edge += [[i, i + mesh.n + 1]for i in range(1, mesh.n)]
    edge += [[(i + 1) * (mesh.n + 1) - 2, (i + 1) * (mesh.n + 1) - 2 + 1]for i in range(1, mesh.n)]
    edge += [[(mesh.n + 1) * (mesh.n - 1) + i, (mesh.n + 1) * (mesh.n - 1) + i + mesh.n + 1] for i in range(1, mesh.n)]
    edge += [[i * (mesh.n + 1), i * (mesh.n + 1) + 1]for i in range(1, mesh.n)]

    edge = [e[0] * (mesh.n + 1) ** 2 + e[1] for e in edge]
    edge_ids = sorted(list(set(edge)))
    ne = edge_ids.__len__()

    # compute mat
    gram_p1 = mesh.gram_p1()
    mat_1 = mesh.node_mul_node(gram_p1)
    mat_1 = mat_1[node_ids, node_ids]
    mat_1 = coo_matrix((mat_1.data, (mat_1.idx[0], mat_1.idx[1])), shape=(nn, nn))

    gram_p1_rt0 = mesh.gram_p1_rt0(gram_p1=gram_p1)
    mat_2 = mesh.node_mul_edge(gram_p1_rt0[:, :, :, 0])
    mat_2 = mat_2[node_ids, edge_ids]
    mat_2 = coo_matrix((mat_2.data, (mat_2.idx[0], mat_2.idx[1])), shape=(nn, ne))

    mat_3 = mat_2.T

    mat_4 = mesh.node_mul_edge(gram_p1_rt0[:, :, :, 1])
    mat_4 = mat_4[node_ids, edge_ids]
    mat_4 = coo_matrix((mat_4.data, (mat_4.idx[0], mat_4.idx[1])), shape=(nn, ne))

    mat_5 = mat_4.T

    gram_rt0 = mesh.gram_rt0(gram_p1=gram_p1)
    mat_6 = mesh.edge_mul_edge(gram_rt0)
    mat_6 = mat_6[edge_ids, edge_ids]
    mat_6 = coo_matrix((mat_6.data, (mat_6.idx[0], mat_6.idx[1])), shape=(ne, ne))

    data = np.hstack([
        mat_1.data,
        mat_2.data,
        mat_1.data,
        mat_4.data,
        mat_3.data,
        mat_5.data,
        mat_6.data])
    row = np.hstack([
        mat_1.row,
        mat_2.row,
        mat_1.row + nn,
        mat_4.row + nn,
        mat_3.row + nn * 2,
        mat_5.row + nn * 2,
        mat_6.row + nn * 2])
    col = np.hstack([
        mat_1.col,
        mat_2.col + nn * 2,
        mat_1.col + nn,
        mat_4.col + nn * 2,
        mat_3.col,
        mat_5.col + nn,
        mat_6.col + nn * 2])
    mat = coo_matrix((data, (row, col)), shape=(2 * nn + ne, 2 * nn + ne))

    # compute rhs
    integer_tensor = mesh.integer_p1(lambda x: func_f(x)[0], num_refine=num_refine)
    rhs_1 = mesh.node_mul_func(integer_tensor)
    rhs_1 = rhs_1[node_ids]

    integer_tensor = mesh.integer_p1(lambda x: func_f(x)[1], num_refine=num_refine)
    rhs_2 = mesh.node_mul_func(integer_tensor)
    rhs_2 = rhs_2[node_ids]

    integer_tensor = mesh.integer_rt0(func_f, num_refine=num_refine)
    rhs_3 = mesh.edge_mul_func(integer_tensor)
    rhs_3 = rhs_3[edge_ids]

    rhs = np.hstack([rhs_1, rhs_2, rhs_3])

    coeff = spsolve(mat, rhs)

    # Check L2 error
    error = mesh.p1_error()
    print(error)

    # Show figure
    fig = plt.figure(figsize=(10, 6))

    xx = mesh.vertices[node_ids, 0]
    yy = mesh.vertices[node_ids, 1]
    u_val = func_u(np.vstack([xx, yy]))
    u_h = np.vstack([coeff[:nn], coeff[nn:2*nn]])

    anchor = [[i / mesh.n, 0] for i in range(mesh.n)]
    anchor += [[1, i / mesh.n] for i in range(mesh.n)]
    anchor += [[i / mesh.n, 1] for i in range(1, mesh.n + 1)]
    anchor += [[0, i / mesh.n] for i in range(1, mesh.n + 1)]
    anchor = np.array(anchor)

    grid = np.array([
        [1/2, 0], [0, 1/2], [1/2, 1], [1, 1/2],
        [1/3, 1/4],
        [3/4, 1/4],
        [1/4, 1/3],
        [2/3, 3/4],
        [1/4, 3/4],
        [3/4, 2/3]
    ]) / mesh.n

    extend_xy = np.vstack([grid + a.reshape(1, -1) for a in anchor])

    extend_xx = extend_xy[:, 0]
    extend_yy = extend_xy[:, 1]

    extend_u_val = func_u(extend_xy.T)

    rt0_coeff = [coeff[edge_ids.index(idx)] if idx in edge_ids else 0
                 for idx in mesh.undirected_graph.flatten_idx]
    rt0_coeff = np.array(rt0_coeff)
    extend_u_h = mesh.rt0_interpolation(extend_xy.T, rt0_coeff)

    xx = np.hstack([xx, extend_xx])
    yy = np.hstack([yy, extend_yy])
    u_val = np.hstack([u_val, extend_u_val])
    u_h = np.hstack([u_h, extend_u_h])

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title("z = u_val(x, y)[0]\nz = u_h(x, y)[0]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[0]")
    ax.plot_trisurf(xx, yy, u_val[0], alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h[0], alpha=0.5)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("z = u_val(x, y)[1]\nz = u_h(x, y)[1]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[1]")
    ax.plot_trisurf(xx, yy, u_val[1], alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h[1], alpha=0.5)

    plt.show()


def example_5(n=16, num_refine=3):
    """
    \boldsymbol{u} + \nabla p = \boldsymbol{g}, \boldsymbol{x} \in \Omega
    \mathrm{div} \boldsymbol{u} = q, \boldsymbol{x} \in \Omega
    p = 0, \boldsymbol{x} \in \partial \Omega

    check items:
    * IsotropicMesh.inner_node_ids
    * FiniteElement.gram_p1
    * FiniteElement.gram_p1_grad_p1
    * FiniteElement.integer_p1
    * LinearSystem.node_mul_node
    * LinearSystem.node_mul_func
    """
    def func_u(x):
        ux = np.sin(np.pi * x[0]) * np.cos(x[0] + x[1])
        uy = np.sin(np.pi * x[1]) * np.cos(x[0] + x[1])
        return np.stack([ux, uy], axis=0)

    def func_p(x): return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    # func_f = func_u
    def func_g(x, eps=1e-6):
        # \boldsymbol{u} + \nabla p = \boldsymbol{g}
        ux, uy = func_u(x)
        px = (func_p([x[0] + eps, x[1]]) - func_p([x[0] - eps, x[1]])) / (2 * eps)
        py = (func_p([x[0], x[1] + eps]) - func_p([x[0], x[1] - eps])) / (2 * eps)
        return np.array([ux + px, uy + py])

    def func_q(x, eps=1e-6):
        # div \boldsymbol{u} = q
        u0x = (func_u([x[0] + eps, x[1]])[0] - func_u([x[0] - eps, x[1]])[0]) / (2 * eps)
        u1y = (func_u([x[0], x[1] + eps])[1] - func_u([x[0], x[1] - eps])[1]) / (2 * eps)
        return u0x + u1y

    # Start to solve
    mesh = utils.SquareMesh(n=n)

    nn = mesh.vertices.__len__()
    inner_node_ids = mesh.inner_node_ids
    inner_nn = inner_node_ids.__len__()

    # \phi_{\boldsymbol{u}}: P_1^2 space
    # \phi_p: P_1 space

    # compute mat of (\phi_{\boldsymbol{u}}, \phi_{\boldsymbol{u}})
    gram_p1 = mesh.gram_p1()
    mat_1 = mesh.node_mul_node(gram_p1)
    mat_1 = coo_matrix((mat_1.data, (mat_1.idx[0], mat_1.idx[1])), shape=(nn, nn))

    data = np.hstack([mat_1.data, mat_1.data])
    row = np.hstack([mat_1.row, mat_1.row + nn])
    col = np.hstack([mat_1.col, mat_1.col + nn])
    mat_uu = coo_matrix((data, (row, col)), shape=(2 * nn, 2 * nn))

    # compute mat of (div \phi_{\boldsymbol{u}}, \phi_p)
    gram_p1_grad_p1 = mesh.gram_p1_grad_p1()
    mat_1 = mesh.node_mul_node(gram_p1_grad_p1[:, :, :, 0])
    mat_1 = mat_1[inner_node_ids, :]
    mat_1 = coo_matrix((mat_1.data, (mat_1.idx[1], mat_1.idx[0])), shape=(nn, inner_nn))

    mat_2 = mesh.node_mul_node(gram_p1_grad_p1[:, :, :, 1])
    mat_2_inner = mat_2[inner_node_ids, :]
    mat_2_inner = coo_matrix((mat_2_inner.data, (mat_2_inner.idx[1], mat_2_inner.idx[0])), shape=(nn, inner_nn))

    data = np.hstack([mat_1.data, mat_2_inner.data])
    row = np.hstack([mat_1.row, mat_2_inner.row + nn])
    col = np.hstack([mat_1.col, mat_2_inner.col])
    mat_up = coo_matrix((data, (row, col)), shape=(2 * nn, inner_nn))

    # compute mat of (\phi_p, \mathrm{div} \phi_{\boldsymbol{u}})
    mat_pu = mat_up.T

    # --- mat ---

    data = np.hstack([mat_uu.data, -mat_up.data, mat_pu.data])
    row = np.hstack([mat_uu.row, mat_up.row, mat_pu.row + 2 * nn])
    col = np.hstack([mat_uu.col, mat_up.col + 2 * nn, mat_pu.col])
    mat = coo_matrix((data, (row, col)), shape=[2 * nn + inner_nn, 2 * nn + inner_nn])

    # compute rhs of (\phi_{\boldsymbol{u}}, \boldsymbol{g})
    integer_tensor = mesh.integer_p1(lambda x: func_g(x)[0], num_refine=num_refine)
    rhs_1 = mesh.node_mul_func(integer_tensor)

    integer_tensor = mesh.integer_p1(lambda x: func_g(x)[1], num_refine=num_refine)
    rhs_2 = mesh.node_mul_func(integer_tensor)

    rhs_ug = np.hstack([rhs_1, rhs_2])

    # compute rhs of (\phi_p, q)
    integer_tensor = mesh.integer_p1(func_q, num_refine=num_refine)
    rhs_pq = mesh.node_mul_func(integer_tensor)
    rhs_pq = rhs_pq[inner_node_ids]

    # --- rhs ---
    rhs = np.hstack([rhs_ug, rhs_pq])

    coeff = spsolve(mat, rhs)

    # Check L2 error
    error = mesh.p1_error()
    print(error)

    # Show figure
    fig = plt.figure(figsize=(15, 6))

    xx = mesh.vertices[:, 0]
    yy = mesh.vertices[:, 1]
    u_val = func_u(np.vstack([xx, yy]))
    u_h = np.vstack([coeff[:nn], coeff[nn:2*nn]])

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_title("z = u_val(x, y)[0]\nz = u_h(x, y)[0]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[0]")
    ax.plot_trisurf(xx, yy, u_val[0], alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h[0], alpha=0.5)

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.set_title("z = u_val(x, y)[0]\nz = u_h(x, y)[1]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[1]")
    ax.plot_trisurf(xx, yy, u_val[1], alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h[1], alpha=0.5)

    p_val = func_p(np.vstack([xx, yy]))
    p_h = np.zeros_like(p_val)
    p_h[inner_node_ids] = coeff[(2 * nn):]

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.set_title("z = p_val(x, y)\nz = p_h(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("p")
    ax.plot_trisurf(xx, yy, p_val, alpha=0.5)
    ax.plot_trisurf(xx, yy, p_h, alpha=0.5)

    plt.show()


def example_6(n=16, num_refine=3):
    """
    \boldsymbol{u} = \boldsymbol{f}, \boldsymbol{x} \in \Omega
    <\boldsymbol{u}, \boldsymbol{n}> = 0, \boldsymbol{x} \in \partial \Omega

    check items:
    * IsotropicMesh.inner_node_ids
    * FiniteElement.gram_p1
    * FiniteElement.integer_p1
    * LinearSystem.node_mul_node
    """

    def func_u(x):
        ux = np.sin(np.pi * x[0]) * np.cos(x[0] + x[1])
        uy = np.sin(np.pi * x[1]) * np.cos(x[0] + x[1])
        return np.stack([ux, uy], axis=0)

    func_f = func_u

    # coarse_mesh
    def_rt0_list = [
        utils.MultiSquareMesh.DefRT0([
            np.array([
                [0., 0.],
                [.5, 0.],
                [0., .5]
            ])
        ]),
        utils.MultiSquareMesh.DefRT0([
            np.array([
                [1., 0.],
                [1., .5],
                [.5, 0.]
            ])
        ]),
        utils.MultiSquareMesh.DefRT0([
            np.array([
                [1., 1.],
                [.5, 1.],
                [1., .5]
            ])
        ]),
        utils.MultiSquareMesh.DefRT0([
            np.array([
                [0., 1.],
                [0., .5],
                [.5, 1.]
            ])
        ]),
        #        *
        #       /|\
        #      / | \
        #     *-----*
        utils.MultiSquareMesh.DefRT0([
            np.array([
                [0.25, 0.],
                [0.5, 0.],
                [0.5, 0.25]
            ]),
            np.array([
                [0.75, 0.],
                [0.5, 0.25],
                [0.5, 0.]
            ])
        ]),
        #        *
        #       /|
        #      / |
        #     *--|
        #      \ |
        #       \|
        #        *
        utils.MultiSquareMesh.DefRT0([
            np.array([
                [1., 0.25],
                [1., 0.5],
                [0.75, 0.5]
            ]),
            np.array([
                [1., 0.75],
                [0.75, 0.5],
                [1., 0.5]
            ])
        ]),
        #     *-----*
        #      \ | /
        #       \|/
        #        *
        utils.MultiSquareMesh.DefRT0([
            np.array([
                [0.75, 1.],
                [0.5, 1.],
                [0.5, 0.75]
            ]),
            np.array([
                [0.25, 1.],
                [0.5, 0.75],
                [0.5, 1.]
            ])
        ]),
        #     *
        #     |\
        #     | \
        #     |--*
        #     | /
        #     |/
        #     *
        utils.MultiSquareMesh.DefRT0([
            np.array([
                [0., 0.75],
                [0., 0.5],
                [0.25, 0.5]
            ]),
            np.array([
                [0., 0.25],
                [0.25, 0.5],
                [0., 0.5]
            ])
        ])
    ]

    ne = def_rt0_list.__len__()

    # fine mesh
    class Mesh(utils.MultiSquareMesh, utils.RT0):
        def __init__(self, n):
            super().__init__(n)
            self.build()

    mesh = Mesh(n)

    node_ids = mesh.inner_node_ids
    nn = mesh.inner_node_ids.__len__()

    # compute mat
    # P_1 X P_1
    gram_p1 = mesh.gram_p1()
    mat_11 = mesh.node_mul_node(gram_p1)
    mat_11 = mat_11[node_ids, node_ids]
    mat_11 = coo_matrix((mat_11.data, (mat_11.idx[0], mat_11.idx[1])), shape=(nn, nn))

    mat_22 = mat_11

    # P_1^2 X RT0
    gram_p1_def_rt0 = mesh.gram_p1_def_rt0(def_rt0_list, num_refine=num_refine)
    mat_13 = gram_p1_def_rt0[node_ids, :, 0]
    mat_13 = coo_matrix((mat_13.data, (mat_13.idx[0], mat_13.idx[1])), shape=(nn, ne))

    mat_23 = gram_p1_def_rt0[node_ids, :, 1]
    mat_23 = coo_matrix((mat_23.data, (mat_23.idx[0], mat_23.idx[1])), shape=(nn, ne))

    # RT0 X P_1^2
    mat_31 = mat_13.T
    mat_32 = mat_23.T

    # RT0 X RT0
    mat_33 = mesh.gram_def_rt0(def_rt0_list, num_refine=num_refine)
    mat_33 = coo_matrix((mat_33.data, (mat_33.idx[0], mat_33.idx[1])), shape=(ne, ne))

    # --- mat ---
    data = np.hstack([
        mat_11.data, mat_13.data,
        mat_22.data, mat_23.data,
        mat_31.data, mat_32.data, mat_33.data
    ])
    row = np.hstack([
        mat_11.row, mat_13.row,
        mat_22.row + nn, mat_23.row + nn,
        mat_31.row + 2 * nn, mat_32.row + 2 * nn, mat_33.row + 2 * nn
    ])
    col = np.hstack([
        mat_11.col, mat_13.col + 2 * nn,
        mat_22.col + nn, mat_23.col + 2 * nn,
        mat_31.col, mat_32.col + nn, mat_33.col + 2 * nn
    ])
    mat = coo_matrix((data, (row, col)), shape=(2 * nn + ne, 2 * nn + ne))

    print(nn, ne, mat.shape)
    print("rank(mat)", np.linalg.matrix_rank(mat.toarray()))

    # compute rhs
    # P_1^2 X \boldsymbol{f}
    integer_tensor = mesh.integer_p1(lambda x: func_f(x)[0], num_refine=num_refine)
    rhs_1 = mesh.node_mul_func(integer_tensor)
    rhs_1 = rhs_1[node_ids]

    integer_tensor = mesh.integer_p1(lambda x: func_f(x)[1], num_refine=num_refine)
    rhs_2 = mesh.node_mul_func(integer_tensor)
    rhs_2 = rhs_2[node_ids]

    # RT0 X \boldsymbol{f}
    rhs_3 = mesh.integer_def_rt0(def_rt0_list, func=func_f, num_refine=num_refine)

    # --- rhs ---
    rhs = np.hstack([rhs_1, rhs_2, rhs_3])

    coeff = spsolve(mat, rhs)

    coeff_u_p12x = coeff[:nn]
    coeff_u_p12y = coeff[nn:2*nn]
    coeff_u_rt0 = coeff[2*nn:]

    # Check L2 error
    error = mesh.p1_error()
    print(error)

    # Show figure
    fig = plt.figure(figsize=(15, 6))

    xx = mesh.vertices[:, 0]
    yy = mesh.vertices[:, 1]
    xy = np.vstack([xx, yy])
    u_val = func_u(xy)
    u_h = np.zeros_like(u_val)
    u_h[:, node_ids] = np.vstack([coeff_u_p12x, coeff_u_p12y])
    for i in range(ne):
        # rt0_i = lambda x: coarse_mesh.rt0(x, edge_ids[i] // coarse_nn, edge_ids[i] % coarse_nn)
        u_h += coeff_u_rt0[i] * def_rt0_list[i].value(xy)

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_title("z = u_val(x, y)[0]\nz = u_h(x, y)[0]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[0]")
    ax.plot_trisurf(xx, yy, u_val[0], alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h[0], alpha=0.5)

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.set_title("z = u_val(x, y)[1]\nz = u_h(x, y)[1]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[1]")
    ax.plot_trisurf(xx, yy, u_val[1], alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h[1], alpha=0.5)

    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("mesh")

    for vertex_ids in mesh.triangles:
        ax.fill(mesh.vertices[vertex_ids, 0], mesh.vertices[vertex_ids, 1], alpha=0.15, c='cyan')

    for def_rt0 in def_rt0_list:
        for tri_vertices in def_rt0.tri_vertices_list:
            ax.fill(tri_vertices[:, 0], tri_vertices[:, 1], alpha=0.45, c='magenta')
            rt0, = ax.plot(tri_vertices[[1, 2], 0], tri_vertices[[1, 2], 1], c='r')

    ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1])
    p12 = ax.scatter(mesh.vertices[mesh.inner_node_ids, 0], mesh.vertices[mesh.inner_node_ids, 1])

    fig.legend((p12, rt0), ("P_1^2", "RT0"))

    plt.show()


def example_7(n=10, num_refine=3):
    """
    \boldsymbol{u} = \boldsymbol{f}, \boldsymbol{x} \in \Omega
    <\boldsymbol{u}, \boldsymbol{n}> = 0, \boldsymbol{x} \in \partial \Omega

    check items:
    * IsotropicMesh.inner_node_inds
    * FiniteElement.gram_p1
    * FiniteElement.integer_p1
    * LinearSystem.node_mul_node
    """

    def func_u(x):
        # ux = np.sin(np.pi * x[0]) * np.cos(x[0] + x[1]) * (1 + x[1]) / 3
        # uy = np.sin(np.pi * x[1]) * np.cos(x[0] + x[1]) * (1 + x[0]) / 3
        ux = np.sin(np.pi * x[0]) * np.cos(x[0] + x[1])
        uy = np.sin(np.pi * x[1]) * np.cos(x[0] + x[1])
        return np.stack([ux, uy], axis=0)

    func_f = func_u

    def func_omega(x):
        return -np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    def func_outer_normal(x, eps=1e-6):
        # \boldsymbol{n} = \frac{\nabla omega}{\| \nabla omega \|}
        px = (func_omega([x[0] + eps, x[1]]) - func_omega([x[0] - eps, x[1]])) / (2 * eps)
        py = (func_omega([x[0], x[1] + eps]) - func_omega([x[0], x[1] - eps])) / (2 * eps)

        nx = px / np.sqrt(np.square(px) + np.square(py))
        ny = py / np.sqrt(np.square(px) + np.square(py))

        return np.array([nx, ny])

    def func_tangent(x, eps=1e-6):
        nx, ny = func_outer_normal(x, eps=eps)
        return np.array([-ny, nx])

    # mesh
    mesh = utils.MultiSquareMesh(n=n)

    inner_ids = mesh.inner_node_ids
    inner_nn = inner_ids.__len__()

    bound_ids = []
    for ind in mesh.bound_node_ids:
        if mesh.vertices[ind, 0] not in [0, 1]:
            bound_ids.append(ind)
        if mesh.vertices[ind, 1] not in [0, 1]:
            bound_ids.append(ind)

    bound_nn = bound_ids.__len__()

    # compute mat
    # P_1 X P_1
    gram_p1 = mesh.gram_p1()
    mat_11 = mesh.node_mul_node(gram_p1)
    mat_11 = mat_11[inner_ids, inner_ids]
    mat_11 = coo_matrix((mat_11.data, (mat_11.idx[0], mat_11.idx[1])), shape=(inner_nn, inner_nn))

    mat_22 = mat_11

    # P_1^2 X ST1
    node_ids = mesh.triangles.reshape(-1)
    vertices = mesh.vertices[node_ids, :].T
    tx, ty = func_tangent(vertices)
    tangents = np.stack([tx.reshape(-1, 3), ty.reshape(-1, 3)], axis=2)
    gram_p1_st1 = np.einsum("tij,tjd->tijd", gram_p1, tangents)

    mat_13 = mesh.node_mul_node(gram_p1_st1[:, :, :, 0])
    mat_13 = mat_13[inner_ids, bound_ids]
    mat_13 = coo_matrix((mat_13.data, (mat_13.idx[0], mat_13.idx[1])), shape=(inner_nn, bound_nn))

    mat_23 = mesh.node_mul_node(gram_p1_st1[:, :, :, 1])
    mat_23 = mat_23[inner_ids, bound_ids]
    mat_23 = coo_matrix((mat_23.data, (mat_23.idx[0], mat_23.idx[1])), shape=(inner_nn, bound_nn))

    # ST1 X P_1^2
    mat_31 = mat_13.T
    mat_32 = mat_23.T

    # ST1 X ST1
    gram_st1 = np.einsum("tij,tid,tjd->tij", gram_p1, tangents, tangents)
    mat_33 = mesh.node_mul_node(gram_st1)
    mat_33 = mat_33[bound_ids, bound_ids]
    mat_33 = coo_matrix((mat_33.data, (mat_33.idx[0], mat_33.idx[1])), shape=(bound_nn, bound_nn))

    # --- mat ---
    data = np.hstack([
        mat_11.data, mat_13.data,
        mat_22.data, mat_23.data,
        mat_31.data, mat_32.data, mat_33.data
    ])
    row = np.hstack([
        mat_11.row, mat_13.row,
        mat_22.row + inner_nn, mat_23.row + inner_nn,
        mat_31.row + 2 * inner_nn, mat_32.row + 2 * inner_nn, mat_33.row + 2 * inner_nn
    ])
    col = np.hstack([
        mat_11.col, mat_13.col + 2 * inner_nn,
        mat_22.col + inner_nn, mat_23.col + 2 * inner_nn,
        mat_31.col, mat_32.col + inner_nn, mat_33.col + 2 * inner_nn
    ])
    mat = coo_matrix((data, (row, col)), shape=(2 * inner_nn + bound_nn, 2 * inner_nn + bound_nn))

    print(inner_nn, bound_nn, mat.shape)
    print("rank(mat)", np.linalg.matrix_rank(mat.toarray()))

    # compute rhs
    # P_1^2 X \boldsymbol{f}
    integer_tensor_1 = mesh.integer_p1(lambda x: func_f(x)[0], num_refine=num_refine)
    rhs_1 = mesh.node_mul_func(integer_tensor_1)
    rhs_1 = rhs_1[inner_ids]

    integer_tensor_2 = mesh.integer_p1(lambda x: func_f(x)[1], num_refine=num_refine)
    rhs_2 = mesh.node_mul_func(integer_tensor_2)
    rhs_2 = rhs_2[inner_ids]

    # ST1 X \boldsymbol{f}
    integer_tensor_3 = tangents[:, :, 0] * integer_tensor_1 + tangents[:, :, 1] * integer_tensor_2
    rhs_3 = mesh.node_mul_func(integer_tensor_3)
    rhs_3 = rhs_3[bound_ids]

    # --- rhs ---
    rhs = np.hstack([rhs_1, rhs_2, rhs_3])

    coeff = spsolve(mat, rhs)

    coeff_u_p12x = coeff[:inner_nn]
    coeff_u_p12y = coeff[inner_nn:2*inner_nn]
    coeff_u_st1 = coeff[2*inner_nn:]

    # Check L2 error
    error = mesh.p1_error()
    print(error)

    # Show figure
    fig = plt.figure(figsize=(15, 6))

    xx = mesh.vertices[np.hstack([inner_ids, bound_ids]), 0]
    yy = mesh.vertices[np.hstack([inner_ids, bound_ids]), 1]
    xy = np.vstack([xx, yy])
    u_val = func_u(xy)
    u_h = np.zeros_like(u_val)
    u_h[:, :inner_nn] = np.vstack([coeff_u_p12x, coeff_u_p12y])
    tangents = func_tangent(mesh.vertices[bound_ids, :].T)
    u_h[:, inner_nn:] = np.einsum('i,di->di', coeff_u_st1, tangents)

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_title("z = u_val(x, y)[0]\nz = u_h(x, y)[0]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[0]")
    ax.plot_trisurf(xx, yy, u_val[0], alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h[0], alpha=0.5)

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.set_title("z = u_val(x, y)[1]\nz = u_h(x, y)[1]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[1]")
    ax.plot_trisurf(xx, yy, u_val[1], alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h[1], alpha=0.5)

    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("mesh")

    for vertex_ids in mesh.triangles:
        ax.fill(mesh.vertices[vertex_ids, 0], mesh.vertices[vertex_ids, 1], alpha=0.15, c='cyan')

    ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1])
    p12 = ax.scatter(mesh.vertices[inner_ids, 0], mesh.vertices[inner_ids, 1])
    st1 = ax.scatter(mesh.vertices[bound_ids, 0], mesh.vertices[bound_ids, 1])

    fig.legend((p12, st1), ("P_1^2", "ST1"))

    plt.show()


def example_8(n=16, num_refine=3):
    """
    \boldsymbol{u} + \nabla p = \boldsymbol{g}, \boldsymbol{x} \in \Omega
    \mathrm{div} \boldsymbol{u} = q, \boldsymbol{x} \in \Omega
    <\boldsymbol{u}, \boldsymbol{n}> = 0, \boldsymbol{x} \in \partial \Omega
    \iint_{\Omega} p = r


    Finite Element Space:
    \phi_{\boldsymbol{u}}: P_1^2 + ST1
    \phi_p: P_1

    check items:
    * IsotropicMesh.inner_node_inds
    * FiniteElement.gram_p1
    * FiniteElement.gram_p1_grad_p1
    * FiniteElement.integer_p1
    * LinearSystem.node_mul_node
    * LinearSystem.node_mul_func
    """
    def func_u(x):
        ux = np.sin(np.pi * x[0]) * np.cos(x[0] + x[1])
        uy = np.sin(np.pi * x[1]) * np.cos(x[0] + x[1])
        return np.stack([ux, uy], axis=0)

    def func_p(x):
        # return np.sin(x[0] - x[1])
        return np.sin(x[0] - x[1]) + np.sin(np.pi * x[0])

    def func_g(x, eps=1e-6):
        # \boldsymbol{u} + \nabla p = \boldsymbol{g}
        ux, uy = func_u(x)
        px = (func_p([x[0] + eps, x[1]]) - func_p([x[0] - eps, x[1]])) / (2 * eps)
        py = (func_p([x[0], x[1] + eps]) - func_p([x[0], x[1] - eps])) / (2 * eps)
        return np.array([ux + px, uy + py])

    def func_q(x, eps=1e-6):
        # div \boldsymbol{u} = q
        u0x = (func_u([x[0] + eps, x[1]])[0] - func_u([x[0] - eps, x[1]])[0]) / (2 * eps)
        u1y = (func_u([x[0], x[1] + eps])[1] - func_u([x[0], x[1] - eps])[1]) / (2 * eps)
        return u0x + u1y

    def func_omega(x):
        return -np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    def func_outer_normal(x, eps=1e-6):
        # \boldsymbol{n} = \frac{\nabla omega}{\| \nabla omega \|}
        px = (func_omega([x[0] + eps, x[1]]) - func_omega([x[0] - eps, x[1]])) / (2 * eps)
        py = (func_omega([x[0], x[1] + eps]) - func_omega([x[0], x[1] - eps])) / (2 * eps)

        nx = px / np.sqrt(np.square(px) + np.square(py))
        ny = py / np.sqrt(np.square(px) + np.square(py))

        return np.array([nx, ny])

    def func_tangent(x, eps=1e-6):
        nx, ny = func_outer_normal(x, eps=eps)
        return np.array([-ny, nx])

    # mesh
    mesh = utils.MultiSquareMesh(n=n)

    inner_ids = mesh.inner_node_ids
    inner_nn = inner_ids.__len__()

    bound_ids = []
    for ind in mesh.bound_node_ids:
        if mesh.vertices[ind, 0] not in [0, 1]:
            bound_ids.append(ind)
        if mesh.vertices[ind, 1] not in [0, 1]:
            bound_ids.append(ind)

    print(bound_ids)

    bound_nn = bound_ids.__len__()

    nn = mesh.vertices.__len__()

    # compute mat of (\phi_{\boldsymbol{u}}, \phi_{\boldsymbol{u}})
    # P_1 X P_1
    gram_p1 = mesh.gram_p1()
    mat_11 = mesh.node_mul_node(gram_p1)
    mat_11 = mat_11[inner_ids, inner_ids]
    mat_11 = coo_matrix((mat_11.data, (mat_11.idx[0], mat_11.idx[1])), shape=(inner_nn, inner_nn))

    mat_22 = mat_11

    # P_1^2 X ST1
    node_ids = mesh.triangles.reshape(-1)
    vertices = mesh.vertices[node_ids, :].T
    tx, ty = func_tangent(vertices)
    tangents = np.stack([tx.reshape(-1, 3), ty.reshape(-1, 3)], axis=2)  # [NT, 3, 2]
    gram_p1_st1 = np.einsum("tij,tjd->tijd", gram_p1, tangents)

    mat_13 = mesh.node_mul_node(gram_p1_st1[:, :, :, 0])
    mat_13 = mat_13[inner_ids, bound_ids]
    mat_13 = coo_matrix((mat_13.data, (mat_13.idx[0], mat_13.idx[1])), shape=(inner_nn, bound_nn))

    mat_23 = mesh.node_mul_node(gram_p1_st1[:, :, :, 1])
    mat_23 = mat_23[inner_ids, bound_ids]
    mat_23 = coo_matrix((mat_23.data, (mat_23.idx[0], mat_23.idx[1])), shape=(inner_nn, bound_nn))

    # ST1 X P_1^2
    mat_31 = mat_13.T
    mat_32 = mat_23.T

    # ST1 X ST1
    gram_st1 = np.einsum("tij,tid,tjd->tij", gram_p1, tangents, tangents)
    mat_33 = mesh.node_mul_node(gram_st1)
    mat_33 = mat_33[bound_ids, bound_ids]
    mat_33 = coo_matrix((mat_33.data, (mat_33.idx[0], mat_33.idx[1])), shape=(bound_nn, bound_nn))

    # --- mat_uu ---
    data = np.hstack([
        mat_11.data, mat_13.data,
        mat_22.data, mat_23.data,
        mat_31.data, mat_32.data, mat_33.data
    ])
    row = np.hstack([
        mat_11.row, mat_13.row,
        mat_22.row + inner_nn, mat_23.row + inner_nn,
        mat_31.row + 2 * inner_nn, mat_32.row + 2 * inner_nn, mat_33.row + 2 * inner_nn
    ])
    col = np.hstack([
        mat_11.col, mat_13.col + 2 * inner_nn,
        mat_22.col + inner_nn, mat_23.col + 2 * inner_nn,
        mat_31.col, mat_32.col + inner_nn, mat_33.col + 2 * inner_nn
    ])
    mat_uu = coo_matrix((data, (row, col)), shape=(2 * inner_nn + bound_nn, 2 * inner_nn + bound_nn))

    print("mat_uu:", mat_uu.shape)
    print("rank(mat_uu):", np.linalg.matrix_rank(mat_uu.toarray()))

    # compute mat of (div \phi_{\boldsymbol{u}}, \phi_p)
    # div P_1^2 X P_1
    gram_p1_grad_p1 = mesh.gram_p1_grad_p1()
    mat_1 = mesh.node_mul_node(gram_p1_grad_p1[:, :, :, 0])
    mat_1 = mat_1[np.hstack([inner_ids, bound_ids]), inner_ids]
    mat_1 = coo_matrix((mat_1.data, (mat_1.idx[1], mat_1.idx[0])), shape=(inner_nn, inner_nn + bound_nn))

    mat_2 = mesh.node_mul_node(gram_p1_grad_p1[:, :, :, 1])
    mat_2 = mat_2[np.hstack([inner_ids, bound_ids]), inner_ids]
    mat_2 = coo_matrix((mat_2.data, (mat_2.idx[1], mat_2.idx[0])), shape=(inner_nn, inner_nn + bound_nn))

    # div ST1 X P_1
    gram_p1_div_st1 = np.einsum("tijd,tjd->tij", gram_p1_grad_p1, tangents)
    mat_3 = mesh.node_mul_node(gram_p1_div_st1)
    mat_3 = mat_3[np.hstack([inner_ids, bound_ids]), bound_ids]
    mat_3 = coo_matrix((mat_3.data, (mat_3.idx[1], mat_3.idx[0])), shape=(bound_nn, inner_nn + bound_nn))

    # --- mat_up ---
    data = np.hstack([
        mat_1.data,
        mat_2.data,
        mat_3.data
    ])
    row = np.hstack([
        mat_1.row,
        mat_2.row + inner_nn,
        mat_3.row + 2 * inner_nn
    ])
    col = np.hstack([
        mat_1.col,
        mat_2.col,
        mat_3.col
    ])
    mat_up = coo_matrix((data, (row, col)), shape=(2 * inner_nn + bound_nn, inner_nn + bound_nn))

    print("mat_up:", mat_up.shape)
    print("rank(mat_up):", np.linalg.matrix_rank(mat_up.toarray()))

    # compute mat of (\phi_p, \mathrm{div} \phi_{\boldsymbol{u}})
    # --- mat_pu ---
    mat_pu = mat_up.T

    # compute mat of (\phi_p, 1)
    gram_p0_p1 = mesh.gram_p0_p1()
    mat_p = mesh.node_mul_node(gram_p0_p1)
    diag_idx = []
    for i, idx in enumerate(mat_p.flatten_idx):
        if (idx // nn) == (idx % nn):
            diag_idx.append(i)
    data = mat_p.data[diag_idx]
    row = np.zeros_like(data, dtype=np.int)
    col = np.arange(nn)
    mat_p = utils.SparseTensor((data, (row, col)), shape=(1, nn))
    mat_p = mat_p[:, np.hstack([inner_ids, bound_ids])]
    mat_p = coo_matrix((mat_p.data, (mat_p.idx[0], mat_p.idx[1])), shape=(1, inner_nn + bound_nn))

    # *** mat ***
    data = np.hstack([
        mat_uu.data, -mat_up.data,
        mat_pu.data,
        mat_p.data
    ])
    row = np.hstack([
        mat_uu.row, mat_up.row,
        mat_pu.row + 2 * inner_nn + bound_nn,
        mat_p.row + 3 * inner_nn + 2 * bound_nn
    ])
    col = np.hstack([
        mat_uu.col, mat_up.col + 2 * inner_nn + bound_nn,
        mat_pu.col,
        mat_p.col + 2 * inner_nn + bound_nn
    ])
    mat = coo_matrix((data, (row, col)), shape=[3 * inner_nn + 2 * bound_nn + 1, 3 * inner_nn + 2 * bound_nn])

    print("mat:", mat.shape)
    print("rank(mat):", np.linalg.matrix_rank(mat.toarray()))

    # compute rhs of (\phi_{\boldsymbol{u}}, \boldsymbol{g})
    # P_1^2 X \boldsymbol{f}
    integer_tensor_1 = mesh.integer_p1(lambda x: func_g(x)[0], num_refine=num_refine)
    rhs_1 = mesh.node_mul_func(integer_tensor_1)
    rhs_1 = rhs_1[inner_ids]

    integer_tensor_2 = mesh.integer_p1(lambda x: func_g(x)[1], num_refine=num_refine)
    rhs_2 = mesh.node_mul_func(integer_tensor_2)
    rhs_2 = rhs_2[inner_ids]

    # ST1 X \boldsymbol{f}
    integer_tensor_3 = tangents[:, :, 0] * integer_tensor_1 + tangents[:, :, 1] * integer_tensor_2
    rhs_3 = mesh.node_mul_func(integer_tensor_3)
    rhs_3 = rhs_3[bound_ids]

    # --- rhs_ug ---
    rhs_ug = np.hstack([rhs_1, rhs_2, rhs_3])

    # compute rhs of (\phi_p, q)
    # P_1 X func
    # --- rhs_pq ---
    integer_tensor = mesh.integer_p1(func_q, num_refine=num_refine)
    rhs_1 = mesh.node_mul_func(integer_tensor)
    rhs_1 = rhs_1[np.hstack([inner_ids, bound_ids])]
    rhs_pq = rhs_1

    # *** rhs ***
    r = mat_p @ func_p(mesh.vertices[np.hstack([inner_ids, bound_ids])].T)
    print("r:", r)
    rhs = np.hstack([rhs_ug, rhs_pq, r])
    # rhs = np.hstack([rhs_ug, rhs_pq])

    print(mat.shape, rhs_ug.shape, rhs_pq.shape, r.shape)
    coeff = spsolve(mat.T @ mat, mat.T @ rhs)

    coeff_u_p12x = coeff[:inner_nn]
    coeff_u_p12y = coeff[inner_nn:2*inner_nn]
    coeff_u_st1 = coeff[2*inner_nn:(2*inner_nn + bound_nn)]
    coeff_p_p1 = coeff[(2*inner_nn + bound_nn):]

    print("mat_p@coeff_p_p1:", mat_p@coeff_p_p1)

    # Show figure
    fig = plt.figure(figsize=(12, 12))

    xx = mesh.vertices[np.hstack([inner_ids, bound_ids]), 0]
    yy = mesh.vertices[np.hstack([inner_ids, bound_ids]), 1]
    xy = np.vstack([xx, yy])
    u_val = func_u(xy)

    # nxx =
    txy = np.mean(mesh.tri_tensor, axis=1).T

    coeff_u_x = np.zeros(shape=(nn, ), dtype=np.float)
    coeff_u_x[inner_ids] = coeff_u_p12x
    coeff_u_x[bound_ids] = coeff_u_st1 * func_tangent(mesh.vertices.T)[0, bound_ids]
    u_x = np.mean(coeff_u_x[mesh.triangles.reshape(-1)].reshape(-1, 3), axis=1)

    coeff_u_y = np.zeros(shape=(nn, ), dtype=np.float)
    coeff_u_y[inner_ids] = coeff_u_p12y
    coeff_u_y[bound_ids] = coeff_u_st1 * func_tangent(mesh.vertices.T)[1, bound_ids]
    u_y = np.mean(coeff_u_y[mesh.triangles.reshape(-1)].reshape(-1, 3), axis=1)

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_title("z = u_val(x, y)[0]\nz = u_h(x, y)[0]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[0]")
    ax.plot_trisurf(xx, yy, u_val[0], alpha=0.5)
    ax.plot_trisurf(txy[0], txy[1], u_x, alpha=0.5)

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.set_title("z = u_val(x, y)[1]\nz = u_h(x, y)[1]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[1]")
    ax.plot_trisurf(xx, yy, u_val[1], alpha=0.5)
    ax.plot_trisurf(txy[0], txy[1], u_y, alpha=0.5)

    xx = mesh.vertices[np.hstack([inner_ids, bound_ids]), 0]
    yy = mesh.vertices[np.hstack([inner_ids, bound_ids]), 1]
    xy = np.vstack([xx, yy])
    p_val = func_p(xy)
    p_h = coeff_p_p1

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.set_title("z = p_val(x, y)\nz = p_h(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("p")
    ax.plot_trisurf(xx, yy, p_val, alpha=0.5)
    ax.plot_trisurf(xx, yy, p_h, alpha=0.5)

    ax = fig.add_subplot(2, 2, 4)
    ax.set_title("mesh")

    for vertex_ids in mesh.triangles:
        ax.fill(mesh.vertices[vertex_ids, 0], mesh.vertices[vertex_ids, 1], alpha=0.15, c='cyan')

    ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1])
    p12 = ax.scatter(mesh.vertices[inner_ids, 0], mesh.vertices[inner_ids, 1])
    st1 = ax.scatter(mesh.vertices[bound_ids, 0], mesh.vertices[bound_ids, 1])

    fig.legend((p12, st1), ("P_1^2", "ST1"))

    plt.show()


def example_10(n=13, num_refine=3):
    """
    \boldsymbol{u} + \nabla p = \boldsymbol{g}, \boldsymbol{x} \in \Omega
    \mathrm{div} \boldsymbol{u} = q, \boldsymbol{x} \in \Omega
    <\boldsymbol{u}, \boldsymbol{n}> = 0, \boldsymbol{x} \in \partial \Omega
    q = r, \boldsymbol{x} \in \partial \Omega

    check items:
    * IsotropicMesh.inner_node_ids
    * FiniteElement.gram_p1
    * FiniteElement.gram_p1_grad_p1
    * FiniteElement.integer_p1
    * LinearSystem.node_mul_node
    * LinearSystem.node_mul_func
    """
    def func_u(x):
        ux = np.sin(np.pi * x[0]) * np.cos(x[0] + x[1])
        uy = np.sin(np.pi * x[1]) * np.cos(x[0] + x[1])
        return np.stack([ux, uy], axis=0)

    def func_p(x):
        # return np.zeros_like(x[0])
        # return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        return np.sin(x[0] - x[1])

    # func_f = func_u
    def func_g(x, eps=1e-6):
        # \boldsymbol{u} + \nabla p = \boldsymbol{g}
        ux, uy = func_u(x)
        px = (func_p([x[0] + eps, x[1]]) - func_p([x[0] - eps, x[1]])) / (2 * eps)
        py = (func_p([x[0], x[1] + eps]) - func_p([x[0], x[1] - eps])) / (2 * eps)
        return np.array([ux + px, uy + py])

    def func_q(x, eps=1e-6):
        # div \boldsymbol{u} = q
        u0x = (func_u([x[0] + eps, x[1]])[0] - func_u([x[0] - eps, x[1]])[0]) / (2 * eps)
        u1y = (func_u([x[0], x[1] + eps])[1] - func_u([x[0], x[1] - eps])[1]) / (2 * eps)
        return u0x + u1y

    def func_r(x): return func_p(x)

    # Mesh
    class Mesh(utils.LinearSystem, utils.FiniteElement):
        def __init__(self, n):
            # generate origin vertices
            x = np.linspace(0, 1, n + 1, endpoint=True)
            y = np.linspace(0, 1, n + 1, endpoint=True)
            X, Y = np.meshgrid(x[4:-5], y[4:-5])
            fine_vertices = np.vstack((X.reshape(-1), Y.reshape(-1))).T

            H = 1 / float((n + 1) // 2)
            X, Y = np.meshgrid(x[0::2], y[0::2])
            coarse_vertices = np.vstack((X.reshape(-1), Y.reshape(-1))).T
            coarse_vertices = [[x, y] for x, y in coarse_vertices if min(min(x, y), min(1-x, 1-y)) <= 2 * H]
            coarse_vertices = np.array(coarse_vertices)

            self.vertices = np.vstack([fine_vertices, coarse_vertices])
            # self.vertices = coarse_vertices

            tri_mesh = Delaunay(self.vertices)
            self.triangles = tri_mesh.simplices
            self.neighbors = tri_mesh.neighbors

    mesh = Mesh(n=n)
    # mesh = utils.SquareMesh(n=n)

    nn = mesh.vertices.__len__()

    inner_node_ids = mesh.inner_node_ids
    inner_nn = inner_node_ids.__len__()

    bound_node_ids = mesh.bound_node_ids
    bound_nn = bound_node_ids.__len__()

    edge_ids = mesh.get_bound_rt0_ids()
    ne = edge_ids.__len__()

    # \phi_{\boldsymbol{u}}: P_1^2 space + TR0 space
    # \phi_p: P_1 space

    # compute mat of (\phi_{\boldsymbol{u}}, \phi_{\boldsymbol{u}})
    # --- P_1 X P_1 ---
    gram_p1 = mesh.gram_p1()
    mat_1 = mesh.node_mul_node(gram_p1)
    mat_1 = mat_1[inner_node_ids, inner_node_ids]
    mat_1 = coo_matrix((mat_1.data, (mat_1.idx[0], mat_1.idx[1])), shape=(inner_nn, inner_nn))

    # --- P_1^2 X RT0 ---
    gram_p1_rt0 = mesh.gram_p1_rt0(gram_p1=gram_p1)
    mat_2 = mesh.node_mul_edge(gram_p1_rt0[:, :, :, 0])
    mat_2 = mat_2[inner_node_ids, edge_ids]
    mat_2 = coo_matrix((mat_2.data, (mat_2.idx[0], mat_2.idx[1])), shape=(inner_nn, ne))

    mat_3 = mesh.node_mul_edge(gram_p1_rt0[:, :, :, 1])
    mat_3 = mat_3[inner_node_ids, edge_ids]
    mat_3 = coo_matrix((mat_3.data, (mat_3.idx[0], mat_3.idx[1])), shape=(inner_nn, ne))

    # --- RT0 X RT0 ---
    gram_rt0 = mesh.gram_rt0(gram_p1=gram_p1)
    mat_4 = mesh.edge_mul_edge(gram_rt0)
    mat_4 = mat_4[edge_ids, edge_ids]
    mat_4 = coo_matrix((mat_4.data, (mat_4.idx[0], mat_4.idx[1])), shape=(ne, ne))

    # === mat_uu ===
    data = np.hstack([
        mat_1.data,
        mat_2.data,
        mat_1.data,
        mat_3.data,
        mat_2.data,
        mat_3.data,
        mat_4.data
    ])
    row = np.hstack([
        mat_1.row,
        mat_2.row,
        mat_1.row + inner_nn,
        mat_3.row + inner_nn,
        mat_2.col + 2 * inner_nn,
        mat_3.col + 2 * inner_nn,
        mat_4.row + 2 * inner_nn
    ])
    col = np.hstack([
        mat_1.col,
        mat_2.col + 2 * inner_nn,
        mat_1.col + inner_nn,
        mat_3.col + 2 * inner_nn,
        mat_2.row,
        mat_3.row + inner_nn,
        mat_4.col + 2 * inner_nn
    ])
    mat_uu = coo_matrix((data, (row, col)), shape=(2 * inner_nn + ne, 2 * inner_nn + ne))
    print(inner_nn, ne, mat_uu.shape)
    print("rank(mat_1):", np.linalg.matrix_rank(mat_1.toarray()))
    print("rank(mat_uu):", np.linalg.matrix_rank(mat_uu.toarray()))

    # compute mat of (div \phi_{\boldsymbol{u}}, \phi_p)
    # --- div P_1^2 X P_1 ---
    gram_p1_grad_p1 = mesh.gram_p1_grad_p1()
    mat_1 = mesh.node_mul_node(gram_p1_grad_p1[:, :, :, 0])
    mat_1 = mat_1[inner_node_ids, inner_node_ids]
    mat_1 = coo_matrix((mat_1.data, (mat_1.idx[1], mat_1.idx[0])), shape=(inner_nn, inner_nn))

    mat_2 = mesh.node_mul_node(gram_p1_grad_p1[:, :, :, 1])
    mat_2 = mat_2[inner_node_ids, inner_node_ids]
    mat_2 = coo_matrix((mat_2.data, (mat_2.idx[1], mat_2.idx[0])), shape=(inner_nn, inner_nn))

    # --- div RT0 X P_1 ---
    gram_p1_div_rt0 = mesh.gram_p1_div_rt0()
    mat_3 = mesh.node_mul_edge(gram_p1_div_rt0)
    mat_3 = mat_3[inner_node_ids, edge_ids]
    mat_3 = coo_matrix((mat_3.data, (mat_3.idx[1], mat_3.idx[0])), shape=(ne, inner_nn))

    # === mat_up ===
    data = np.hstack([
        mat_1.data,
        mat_2.data,
        mat_3.data
    ])
    row = np.hstack([
        mat_1.row,
        mat_2.row + inner_nn,
        mat_3.row + 2 * inner_nn
    ])
    col = np.hstack([
        mat_1.col,
        mat_2.col,
        mat_3.col
    ])
    mat_up = coo_matrix((data, (row, col)), shape=(2 * inner_nn + ne, inner_nn))

    print("mat_up:", mat_up.shape)
    print("rank(mat_up):", np.linalg.matrix_rank(mat_up.toarray()))

    # compute mat of (\phi_p, \mathrm{div} \phi_{\boldsymbol{u}})
    # === mat_pu ===
    mat_pu = mat_up.T

    # *** mat ***
    data = np.hstack([mat_uu.data, -mat_up.data, mat_pu.data])
    row = np.hstack([mat_uu.row, mat_up.row, mat_pu.row + 2 * inner_nn + ne])
    col = np.hstack([mat_uu.col, mat_up.col + 2 * inner_nn + ne, mat_pu.col])
    mat = coo_matrix((data, (row, col)), shape=[3 * inner_nn + ne, 3 * inner_nn + ne])

    print(inner_nn, nn, mat_uu.shape, mat_up.shape, mat_uu.shape, mat.shape)
    print("rank(mat_uu):", np.linalg.matrix_rank(mat_uu.toarray()))
    print("rank(mat):", np.linalg.matrix_rank(mat.toarray()))

    # compute rhs of (\phi_{\boldsymbol{u}}, \boldsymbol{g})
    # --- P_1^2 X func ---
    integer_tensor = mesh.integer_p1(lambda x: func_g(x)[0], num_refine=num_refine)
    rhs_1 = mesh.node_mul_func(integer_tensor)
    rhs_1 = rhs_1[inner_node_ids]

    integer_tensor = mesh.integer_p1(lambda x: func_g(x)[1], num_refine=num_refine)
    rhs_2 = mesh.node_mul_func(integer_tensor)
    rhs_2 = rhs_2[inner_node_ids]

    # --- RT0 X func ---
    integer_tensor = mesh.integer_rt0(func_g, num_refine=num_refine)
    rhs_3 = mesh.edge_mul_func(integer_tensor)
    rhs_3 = rhs_3[edge_ids]

    # === rhs_ug ===
    rhs_ug = np.hstack([rhs_1, rhs_2, rhs_3])

    # compute rhs of (\phi_p, q)
    # --- P_1 X func ---
    integer_tensor = mesh.integer_p1(func_q, num_refine=num_refine)
    rhs_1 = mesh.node_mul_func(integer_tensor)

    # === rhs_pq ===
    rhs_pq = rhs_1[inner_node_ids]

    # *** rhs ***
    # --- div P1_2^2 X P_1 ---
    mat_1 = mesh.node_mul_node(gram_p1_grad_p1[:, :, :, 0])
    mat_1 = mat_1[bound_node_ids, inner_node_ids]
    mat_1 = coo_matrix((mat_1.data, (mat_1.idx[1], mat_1.idx[0])), shape=(inner_nn, bound_nn))

    mat_2 = mesh.node_mul_node(gram_p1_grad_p1[:, :, :, 0])
    mat_2 = mat_2[bound_node_ids, inner_node_ids]
    mat_2 = coo_matrix((mat_2.data, (mat_2.idx[1], mat_2.idx[0])), shape=(inner_nn, bound_nn))

    # --- div RT0 X P_1 ---
    mat_3 = mesh.node_mul_edge(gram_p1_div_rt0)
    mat_3 = mat_3[bound_node_ids, edge_ids]
    mat_3 = coo_matrix((mat_3.data, (mat_3.idx[1], mat_3.idx[0])), shape=(ne, bound_nn))

    bound_p = func_r(mesh.vertices[bound_node_ids, :].T)

    print((mat_1@bound_p).shape)
    print((mat_2@bound_p).shape)
    print((mat_3@bound_p).shape)
    rhs = np.hstack([rhs_ug + np.hstack([mat_1@bound_p, mat_2@bound_p, mat_3@bound_p]), rhs_pq])
    coeff = spsolve(mat, rhs)

    coeff_u_p1x = coeff[:inner_nn]
    coeff_u_p1y = coeff[inner_nn:2*inner_nn]
    coeff_u_rt0 = coeff[2*inner_nn:(2*inner_nn + ne)]
    coeff_p_p1 = coeff[(2*inner_nn + ne):]

    # Check L2 error
    # error = mesh.p1_error()
    # print(error)

    # Show figure
    fig = plt.figure(figsize=(12, 12))

    xx = mesh.vertices[inner_node_ids, 0]
    yy = mesh.vertices[inner_node_ids, 1]
    xy = np.vstack([xx, yy])
    u_val = func_u(xy)
    p12_h = np.vstack([coeff_u_p1x, coeff_u_p1y])
    # rt0_h = (coeff_u_rt0[0] * rt0_ws(xy)
    #          + coeff_u_rt0[1] * rt0_se(xy)
    #          + coeff_u_rt0[2] * rt0_en(xy)
    #          + coeff_u_rt0[3] * rt0_nw(xy))
    # u_h = p12_h + rt0_h
    u_h = p12_h

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_title("z = u_val(x, y)[0]\nz = u_h(x, y)[0]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[0]")
    ax.plot_trisurf(xx, yy, u_val[0], alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h[0], alpha=0.5)

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.set_title("z = u_val(x, y)[0]\nz = u_h(x, y)[1]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[1]")
    ax.plot_trisurf(xx, yy, u_val[1], alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h[1], alpha=0.5)

    xx = mesh.vertices[:, 0]
    yy = mesh.vertices[:, 1]
    xy = np.vstack([xx, yy])
    p_val = func_p(xy)
    p_h = func_r(xy)
    p_h[inner_node_ids] = coeff_p_p1

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.set_title("z = p_val(x, y)\nz = p_h(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("p")
    ax.plot_trisurf(xx, yy, p_val, alpha=0.5)
    ax.plot_trisurf(xx, yy, p_h, alpha=0.5)

    ax = fig.add_subplot(2, 2, 4)
    ax.set_title("mesh")

    for vertex_ids in mesh.triangles:
        ax.fill(mesh.vertices[vertex_ids, 0], mesh.vertices[vertex_ids, 1], alpha=0.15, c='cyan')
    # for vertices in [ws_vertices, se_vertices, en_vertices, nw_vertices]:
    #     rt0, = ax.fill(vertices[:, 0], vertices[:, 1], alpha=0.35, c='orange')

    for idx in edge_ids:
        start_node = idx // nn
        end_node = idx % nn
        rt0, = ax.plot(mesh.vertices[[start_node, end_node], 0], mesh.vertices[[start_node, end_node], 1], c='r')

    ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1])
    p12 = ax.scatter(mesh.vertices[mesh.inner_node_ids, 0], mesh.vertices[mesh.inner_node_ids, 1])

    fig.legend((p12, rt0, p12), ("P_1^2", "RT0", "P_1"))

    plt.show()


def example_11(n=19, num_refine=3):
    """
    \boldsymbol{u} + \nabla p = \boldsymbol{g}, \boldsymbol{x} \in \Omega
    \mathrm{div} \boldsymbol{u} = q, \boldsymbol{x} \in \Omega
    <\boldsymbol{u}, \boldsymbol{n}> = 0, \boldsymbol{x} \in \partial \Omega

    check items:
    * IsotropicMesh.inner_node_ids
    * FiniteElement.gram_p1
    * FiniteElement.gram_p1_grad_p1
    * FiniteElement.integer_p1
    * LinearSystem.node_mul_node
    * LinearSystem.node_mul_func
    """
    def func_u(x):
        ux = np.sin(np.pi * x[0]) * np.cos(x[0] + x[1])
        uy = np.sin(np.pi * x[1]) * np.cos(x[0] + x[1])
        return np.stack([ux, uy], axis=0)

    def func_p(x):
        # return np.zeros_like(x[0])
        # return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        # return np.exp(np.cos(np.pi * x[0]) + np.cos(np.pi * x[1]))
        return np.sin(x[0] - x[1])

    # func_f = func_u
    def func_g(x, eps=1e-6):
        # \boldsymbol{u} + \nabla p = \boldsymbol{g}
        ux, uy = func_u(x)
        px = (func_p([x[0] + eps, x[1]]) - func_p([x[0] - eps, x[1]])) / (2 * eps)
        py = (func_p([x[0], x[1] + eps]) - func_p([x[0], x[1] - eps])) / (2 * eps)
        return np.array([ux + px, uy + py])

    def func_q(x, eps=1e-6):
        # div \boldsymbol{u} = q
        u0x = (func_u([x[0] + eps, x[1]])[0] - func_u([x[0] - eps, x[1]])[0]) / (2 * eps)
        u1y = (func_u([x[0], x[1] + eps])[1] - func_u([x[0], x[1] - eps])[1]) / (2 * eps)
        return u0x + u1y

    # Mesh
    class Mesh(utils.LinearSystem, utils.FiniteElement):
        def __init__(self, n):
            # generate origin vertices
            x = np.linspace(0, 1, n + 1, endpoint=True)
            y = np.linspace(0, 1, n + 1, endpoint=True)
            X, Y = np.meshgrid(x[4:-5], y[4:-5])
            fine_vertices = np.vstack((X.reshape(-1), Y.reshape(-1))).T

            H = 1 / float((n + 1) // 2)
            X, Y = np.meshgrid(x[0::2], y[0::2])
            coarse_vertices = np.vstack((X.reshape(-1), Y.reshape(-1))).T
            coarse_vertices = [[x, y] for x, y in coarse_vertices if min(min(x, y), min(1-x, 1-y)) <= 2 * H]
            coarse_vertices = np.array(coarse_vertices)

            self.vertices = np.vstack([fine_vertices, coarse_vertices])
            # self.vertices = coarse_vertices

            tri_mesh = Delaunay(self.vertices)
            self.triangles = tri_mesh.simplices
            self.neighbors = tri_mesh.neighbors

    # mesh = Mesh(n=n)
    mesh = utils.SquareMesh(n=n)

    nn = mesh.vertices.__len__()

    inner_node_ids = mesh.inner_node_ids
    inner_nn = inner_node_ids.__len__()

    bound_node_ids = mesh.bound_node_ids
    bound_nn = bound_node_ids.__len__()

    edge_ids = mesh.get_bound_rt0_ids()
    ne = edge_ids.__len__()

    # \phi_{\boldsymbol{u}}: P_1^2 space + TR0 space
    # \phi_p: P_1 space

    # compute mat of (\phi_{\boldsymbol{u}}, \phi_{\boldsymbol{u}})
    # --- P_1 X P_1 ---
    gram_p1 = mesh.gram_p1()
    mat_1 = mesh.node_mul_node(gram_p1)
    mat_1 = mat_1[inner_node_ids, inner_node_ids]
    mat_1 = coo_matrix((mat_1.data, (mat_1.idx[0], mat_1.idx[1])), shape=(inner_nn, inner_nn))

    # --- P_1^2 X RT0 ---
    gram_p1_rt0 = mesh.gram_p1_rt0(gram_p1=gram_p1)
    mat_2 = mesh.node_mul_edge(gram_p1_rt0[:, :, :, 0])
    mat_2 = mat_2[inner_node_ids, edge_ids]
    mat_2 = coo_matrix((mat_2.data, (mat_2.idx[0], mat_2.idx[1])), shape=(inner_nn, ne))

    mat_3 = mesh.node_mul_edge(gram_p1_rt0[:, :, :, 1])
    mat_3 = mat_3[inner_node_ids, edge_ids]
    mat_3 = coo_matrix((mat_3.data, (mat_3.idx[0], mat_3.idx[1])), shape=(inner_nn, ne))

    # --- RT0 X RT0 ---
    gram_rt0 = mesh.gram_rt0(gram_p1=gram_p1)
    mat_4 = mesh.edge_mul_edge(gram_rt0)
    mat_4 = mat_4[edge_ids, edge_ids]
    mat_4 = coo_matrix((mat_4.data, (mat_4.idx[0], mat_4.idx[1])), shape=(ne, ne))

    # === mat_uu ===
    data = np.hstack([
        mat_1.data,
        mat_2.data,
        mat_1.data,
        mat_3.data,
        mat_2.data,
        mat_3.data,
        mat_4.data
    ])
    row = np.hstack([
        mat_1.row,
        mat_2.row,
        mat_1.row + inner_nn,
        mat_3.row + inner_nn,
        mat_2.col + 2 * inner_nn,
        mat_3.col + 2 * inner_nn,
        mat_4.row + 2 * inner_nn
    ])
    col = np.hstack([
        mat_1.col,
        mat_2.col + 2 * inner_nn,
        mat_1.col + inner_nn,
        mat_3.col + 2 * inner_nn,
        mat_2.row,
        mat_3.row + inner_nn,
        mat_4.col + 2 * inner_nn
    ])
    mat_uu = coo_matrix((data, (row, col)), shape=(2 * inner_nn + ne, 2 * inner_nn + ne))
    print(inner_nn, ne, mat_uu.shape)
    print("rank(mat_1):", np.linalg.matrix_rank(mat_1.toarray()))
    print("rank(mat_uu):", np.linalg.matrix_rank(mat_uu.toarray()))

    # compute mat of (div \phi_{\boldsymbol{u}}, \phi_p)
    # --- div P_1^2 X P_1 ---
    gram_p1_grad_p1 = mesh.gram_p1_grad_p1()
    mat_1 = mesh.node_mul_node(gram_p1_grad_p1[:, :, :, 0])
    mat_1 = mat_1[:, inner_node_ids]
    mat_1 = coo_matrix((mat_1.data, (mat_1.idx[1], mat_1.idx[0])), shape=(inner_nn, nn))

    mat_2 = mesh.node_mul_node(gram_p1_grad_p1[:, :, :, 1])
    mat_2 = mat_2[:, inner_node_ids]
    mat_2 = coo_matrix((mat_2.data, (mat_2.idx[1], mat_2.idx[0])), shape=(inner_nn, nn))

    # --- div RT0 X P_1 ---
    gram_p1_div_rt0 = mesh.gram_p1_div_rt0()
    mat_3 = mesh.node_mul_edge(gram_p1_div_rt0)
    mat_3 = mat_3[:, edge_ids]
    mat_3 = coo_matrix((mat_3.data, (mat_3.idx[1], mat_3.idx[0])), shape=(ne, nn))

    # === mat_up ===
    data = np.hstack([
        mat_1.data,
        mat_2.data,
        mat_3.data
    ])
    row = np.hstack([
        mat_1.row,
        mat_2.row + inner_nn,
        mat_3.row + 2 * inner_nn
    ])
    col = np.hstack([
        mat_1.col,
        mat_2.col,
        mat_3.col
    ])
    mat_up = coo_matrix((data, (row, col)), shape=(2 * inner_nn + ne, nn))

    print("mat_up:", mat_up.shape)
    print("rank(mat_up):", np.linalg.matrix_rank(mat_up.toarray()))

    # compute mat of (\phi_p, \mathrm{div} \phi_{\boldsymbol{u}})
    # === mat_pu ===
    mat_pu = mat_up.T

    # *** mat ***
    data = np.hstack([mat_uu.data, -mat_up.data, mat_pu.data])
    row = np.hstack([mat_uu.row, mat_up.row, mat_pu.row + 2 * inner_nn + ne])
    col = np.hstack([mat_uu.col, mat_up.col + 2 * inner_nn + ne, mat_pu.col])
    mat = coo_matrix((data, (row, col)), shape=[2 * inner_nn + ne + nn, 2 * inner_nn + ne + nn])

    print(inner_nn, nn, mat_uu.shape, mat_up.shape, mat_uu.shape, mat.shape)
    print("rank(mat_uu):", np.linalg.matrix_rank(mat_uu.toarray()))
    print("rank(mat):", np.linalg.matrix_rank(mat.toarray()))

    # compute rhs of (\phi_{\boldsymbol{u}}, \boldsymbol{g})
    # --- P_1^2 X func ---
    integer_tensor = mesh.integer_p1(lambda x: func_g(x)[0], num_refine=num_refine)
    rhs_1 = mesh.node_mul_func(integer_tensor)
    rhs_1 = rhs_1[inner_node_ids]

    integer_tensor = mesh.integer_p1(lambda x: func_g(x)[1], num_refine=num_refine)
    rhs_2 = mesh.node_mul_func(integer_tensor)
    rhs_2 = rhs_2[inner_node_ids]

    # --- RT0 X func ---
    integer_tensor = mesh.integer_rt0(func_g, num_refine=num_refine)
    rhs_3 = mesh.edge_mul_func(integer_tensor)
    rhs_3 = rhs_3[edge_ids]

    # === rhs_ug ===
    rhs_ug = np.hstack([rhs_1, rhs_2, rhs_3])

    # compute rhs of (\phi_p, q)
    # --- P_1 X func ---
    # === rhs_pq ===
    integer_tensor = mesh.integer_p1(func_q, num_refine=num_refine)
    rhs_pq = mesh.node_mul_func(integer_tensor)

    # *** rhs ***
    rhs = np.hstack([rhs_ug, rhs_pq])
    coeff = spsolve(mat, rhs)

    coeff_u_p1x = coeff[:inner_nn]
    coeff_u_p1y = coeff[inner_nn:2*inner_nn]
    coeff_u_rt0 = coeff[2*inner_nn:(2*inner_nn + ne)]
    coeff_p_p1 = coeff[(2*inner_nn + ne):]

    # Check L2 error
    # error = mesh.p1_error()
    # print(error)

    # Show figure
    fig = plt.figure(figsize=(12, 12))

    xx = mesh.vertices[inner_node_ids, 0]
    yy = mesh.vertices[inner_node_ids, 1]
    xy = np.vstack([xx, yy])
    u_val = func_u(xy)
    p12_h = np.vstack([coeff_u_p1x, coeff_u_p1y])
    # rt0_h = (coeff_u_rt0[0] * rt0_ws(xy)
    #          + coeff_u_rt0[1] * rt0_se(xy)
    #          + coeff_u_rt0[2] * rt0_en(xy)
    #          + coeff_u_rt0[3] * rt0_nw(xy))
    # u_h = p12_h + rt0_h
    u_h = p12_h

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_title("z = u_val(x, y)[0]\nz = u_h(x, y)[0]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[0]")
    ax.plot_trisurf(xx, yy, u_val[0], alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h[0], alpha=0.5)

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.set_title("z = u_val(x, y)[0]\nz = u_h(x, y)[1]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u[1]")
    ax.plot_trisurf(xx, yy, u_val[1], alpha=0.5)
    ax.plot_trisurf(xx, yy, u_h[1], alpha=0.5)

    xx = mesh.vertices[:, 0]
    yy = mesh.vertices[:, 1]
    xy = np.vstack([xx, yy])
    p_val = func_p(xy)
    p_h = coeff_p_p1

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.set_title("z = p_val(x, y)\nz = p_h(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("p")
    ax.plot_trisurf(xx, yy, p_val, alpha=0.5)
    ax.plot_trisurf(xx, yy, p_h, alpha=0.5)

    ax = fig.add_subplot(2, 2, 4)
    ax.set_title("mesh")

    for vertex_ids in mesh.triangles:
        ax.fill(mesh.vertices[vertex_ids, 0], mesh.vertices[vertex_ids, 1], alpha=0.15, c='cyan')

    for idx in edge_ids:
        start_node = idx // nn
        end_node = idx % nn
        rt0, = ax.plot(mesh.vertices[[start_node, end_node], 0], mesh.vertices[[start_node, end_node], 1], c='r')

    ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1])
    p12 = ax.scatter(mesh.vertices[mesh.inner_node_ids, 0], mesh.vertices[mesh.inner_node_ids, 1])

    fig.legend((p12, rt0, p12), ("P_1^2", "RT0", "P_1"))

    plt.show()


example_0()
example_1()
example_2()
example_3()
example_4()
example_5()
example_6()
example_7()
example_8()
