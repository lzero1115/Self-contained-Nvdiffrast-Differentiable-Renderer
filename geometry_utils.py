import numpy as np
import igl
import torch
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix,cKDTree
from pytorch3d.ops.knn import knn_points
from torch.nn.functional import l1_loss
import scipy.optimize

def laplacian_cot(verts, faces): # negative semi-definite
    """
    Compute the cotangent laplacian

    Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """

    # V = sum(V_n), F = sum(F_n)
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # Add the diagonal indices
    vals = torch.sparse.sum(L, dim=0).to_dense()
    indices = torch.arange(V, device='cuda')
    idx = torch.stack([indices, indices], dim=0)
    L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L # COO Sparse matrix
    return L

def laplacian_uniform(verts, faces): # simple unweighted D-A
    """
    Compute the uniform laplacian

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device='cuda', dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()


def remove_duplicates(v, f):
    """
    Generate a mesh representation with no duplicates and
    return it along with the mapping to the original mesh layout.
    """

    unique_verts, inverse = torch.unique(v, dim=0, return_inverse=True)
    new_faces = inverse[f.long()]
    return unique_verts, new_faces, inverse

def average_edge_length(verts, faces):
    """
    Compute the average length of all edges in a given mesh.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    return (A + B + C).sum() / faces.shape[0] / 3

def mass_voronoi(verts, faces): # voronois mass value, the order is the same as faces
    """
    Compute the area of the Voronoi cell around each vertex in the mesh.
    http://www.alecjacobson.com/weblog/?p=863

    params
    ------

    v: vertex positions
    f: triangle indices
    """
    # Compute edge lengths
    l0 = (verts[faces[:,1]] - verts[faces[:,2]]).norm(dim=1)
    l1 = (verts[faces[:,2]] - verts[faces[:,0]]).norm(dim=1)
    l2 = (verts[faces[:,0]] - verts[faces[:,1]]).norm(dim=1)
    l = torch.stack((l0, l1, l2), dim=1)

    # Compute cosines of the corners of the triangles
    cos0 = (l[:,1].square() + l[:,2].square() - l[:,0].square())/(2*l[:,1]*l[:,2])
    cos1 = (l[:,2].square() + l[:,0].square() - l[:,1].square())/(2*l[:,2]*l[:,0])
    cos2 = (l[:,0].square() + l[:,1].square() - l[:,2].square())/(2*l[:,0]*l[:,1])
    cosines = torch.stack((cos0, cos1, cos2), dim=1)

    # Convert to barycentric coordinates
    barycentric = cosines * l
    barycentric = barycentric / torch.sum(barycentric, dim=1)[..., None]

    # Compute areas of the faces using Heron's formula
    areas = 0.25 * ((l0+l1+l2)*(l0+l1-l2)*(l0-l1+l2)*(-l0+l1+l2)).sqrt()

    # Compute the areas of the sub triangles
    tri_areas = areas[..., None] * barycentric

    # Compute the area of the quad
    cell0 = 0.5 * (tri_areas[:,1] + tri_areas[:, 2])
    cell1 = 0.5 * (tri_areas[:,2] + tri_areas[:, 0])
    cell2 = 0.5 * (tri_areas[:,0] + tri_areas[:, 1])
    cells = torch.stack((cell0, cell1, cell2), dim=1)

    # Different formulation for obtuse triangles
    # See http://www.alecjacobson.com/weblog/?p=874
    cells[:,0] = torch.where(cosines[:,0]<0, 0.5*areas, cells[:,0])
    cells[:,1] = torch.where(cosines[:,0]<0, 0.25*areas, cells[:,1])
    cells[:,2] = torch.where(cosines[:,0]<0, 0.25*areas, cells[:,2])

    cells[:,0] = torch.where(cosines[:,1]<0, 0.25*areas, cells[:,0])
    cells[:,1] = torch.where(cosines[:,1]<0, 0.5*areas, cells[:,1])
    cells[:,2] = torch.where(cosines[:,1]<0, 0.25*areas, cells[:,2])

    cells[:,0] = torch.where(cosines[:,2]<0, 0.25*areas, cells[:,0])
    cells[:,1] = torch.where(cosines[:,2]<0, 0.25*areas, cells[:,1])
    cells[:,2] = torch.where(cosines[:,2]<0, 0.5*areas, cells[:,2])

    # Sum the quad areas to get the voronoi cell
    # voronoi_areas = torch.zeros_like(verts).scatter_add_(0, faces, cells).sum(dim=1)
    # return torch.diag_embed(voronoi_areas)
    return torch.zeros_like(verts).scatter_add_(0, faces, cells).sum(dim=1)

def massmatrix_voronoi(verts, faces): # COO sparse voroni mass matrix
    voronoi_areas = mass_voronoi(verts, faces)
    # Create a dense diagonal matrix
    dense_diag_matrix = torch.diag_embed(voronoi_areas)
    # Convert the dense diagonal matrix to a sparse format
    # sparse_diag_matrix = dense_diag_matrix.to_sparse()
    # Alternatively, create the sparse diagonal matrix directly
    indices = torch.arange(len(voronoi_areas)).unsqueeze(0).repeat(2, 1)
    values = voronoi_areas
    sparse_diag_matrix_direct = torch.sparse_coo_tensor(indices, values, (len(voronoi_areas), len(voronoi_areas)))
    return sparse_diag_matrix_direct

def compute_face_normals(verts, faces):
    """
    Compute per-face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    """
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)

    v = [verts.index_select(1, fi[0]),
                 verts.index_select(1, fi[1]),
                 verts.index_select(1, fi[2])]

    c = torch.cross(v[1] - v[0], v[2] - v[0])
    n = c / torch.norm(c, dim=0)
    return n

def safe_acos(x):
    return torch.acos(x.clamp(min=-1, max=1))

def compute_vertex_normals(verts, faces, face_normals):
    """
    Compute per-vertex normals from face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    face_normals : torch.Tensor
        Per-face normals
    """
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)
    normals = torch.zeros_like(verts)

    v = [verts.index_select(1, fi[0]),
             verts.index_select(1, fi[1]),
             verts.index_select(1, fi[2])]

    for i in range(3):
        d0 = v[(i + 1) % 3] - v[i]
        d0 = d0 / torch.norm(d0)
        d1 = v[(i + 2) % 3] - v[i]
        d1 = d1 / torch.norm(d1)
        d = torch.sum(d0*d1, 0)
        face_angle = safe_acos(torch.sum(d0*d1, 0))
        nn =  face_normals * face_angle
        for j in range(3):
            normals[j].index_add_(0, fi[i], nn[j])
    return (normals / torch.norm(normals, dim=0)).transpose(0, 1)

def compute_face_areas(V,F):
    # Ensure F is a long tensor
    F = F.long()

    # Get the vertices of each face
    v0 = V[F[:, 0], :]
    v1 = V[F[:, 1], :]
    v2 = V[F[:, 2], :]

    edge1, edge2 = v1 - v0, v2 - v0
    cross_product = torch.cross(edge1, edge2, dim=1)
    areas = 0.5 * torch.norm(cross_product, dim=1)
    return areas

def rand_barycentric_coords(num_samples):
    uv = torch.rand(2, num_samples, device = 'cuda')
    u, v = uv[0], uv[1]
    u_sqrt = torch.sqrt(u)
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    return w0, w1, w2


def uniform_sampling(V, F, n_samples):
    # more reliable sampling... inspired by pytorch3d sample_points_from_meshes method
    F = F.long()
    areas = compute_face_areas(V, F)
    total_area = areas.sum()
    if total_area == 0:
        raise ValueError("Total area of the mesh is zero, cannot sample points.")

    # Multinomial sampling of face indices
    sample_face_idxs = torch.multinomial(areas, n_samples, replacement=True)

    # Get vertices of the sampled faces
    v0, v1, v2 = V[F[sample_face_idxs, 0], :], V[F[sample_face_idxs, 1], :], V[F[sample_face_idxs, 2], :]

    # Generate barycentric coordinates
    w0, w1, w2 = rand_barycentric_coords(n_samples)

    # Compute the sampled points using barycentric coordinates
    samples = w0[:, None] * v0 + w1[:, None] * v1 + w2[:, None] * v2

    return samples


# Function to uniformly sample points on a mesh uniformly, not stable!!!!!!
def custom_uniform_sampling(V, F, n_samples):
    # Calculate areas of the triangles
    F = F.long()
    areas = compute_face_areas(V,F)
    cumulative_areas = torch.cumsum(areas, dim=0)
    total_area = cumulative_areas[-1]

    # Ensure cumulative_areas and total_area are on the same device
    cumulative_areas = cumulative_areas.to(V.device)
    total_area = total_area.to(V.device)

    # samples = torch.zeros((n_samples, 3), dtype=torch.float32)
    #
    # for i in range(n_samples):
    #     # Randomly select a triangle weighted by its area
    #     r = torch.rand(1, device=V.device).item() * total_area.item()
    #     face_index = torch.searchsorted(cumulative_areas, torch.tensor([r], device=V.device)).item()
    #
    #     # Get the vertices of the chosen triangle
    #     v0 = V[F[face_index, 0], :]
    #     v1 = V[F[face_index, 1], :]
    #     v2 = V[F[face_index, 2], :]
    #
    #     # Generate uniform random barycentric coordinates
    #     u = torch.rand(1, device=V.device).item()
    #     v = torch.rand(1, device=V.device).item()
    #     sqrt_u = torch.sqrt(torch.tensor(u))
    #     barycentric_coords = [1 - sqrt_u.item(), sqrt_u.item() * (1 - v), sqrt_u.item() * v]
    #
    #     # Convert barycentric coordinates to Cartesian coordinates
    #     sample = barycentric_coords[0] * v0 + barycentric_coords[1] * v1 + barycentric_coords[2] * v2
    #     samples[i, :] = sample

    # Generate random numbers for sampling faces
    random_values = torch.rand(n_samples, device=V.device) * total_area
    # Find which faces to sample
    face_indices = torch.searchsorted(cumulative_areas, random_values)
    # Get the vertices of the chosen triangles
    v0 = V[F[face_indices, 0], :]
    v1 = V[F[face_indices, 1], :]
    v2 = V[F[face_indices, 2], :]

    # Generate uniform random barycentric coordinates
    u = torch.rand(n_samples, device=V.device)
    v = torch.rand(n_samples, device=V.device)
    sqrt_u = torch.sqrt(u)
    w = 1 - sqrt_u
    v = sqrt_u * (1 - v)
    u = sqrt_u * v

    # Compute the sampled points using barycentric coordinates
    samples = w.unsqueeze(1) * v0 + u.unsqueeze(1) * v1 + v.unsqueeze(1) * v2

    return samples


def compute_bounding_box(vertices):
    """
        Compute the bounding box of a given mesh.

        Parameters:
        vertices (torch.Tensor): A tensor of shape (N, 3) representing the vertices of the mesh.

        Returns:
        min_coords (torch.Tensor): A tensor of shape (3,) representing the minimum x, y, z coordinates of the bounding box.
        max_coords (torch.Tensor): A tensor of shape (3,) representing the maximum x, y, z coordinates of the bounding box.
        """
    if not isinstance(vertices, torch.Tensor):
        raise TypeError("Vertices should be a torch.Tensor")

    if vertices.ndimension() != 2 or vertices.size(1) != 3:
        raise ValueError("Vertices should be of shape (N, 3)")

    # Calculate min and max coordinates
    min_coords = torch.min(vertices, dim=0)[0]
    max_coords = torch.max(vertices, dim=0)[0]

    return min_coords, max_coords

def mesh_rescale(vertices, scale_factors):
    """
        Rescale the mesh based on specified scale factors applied to the lengths of the bounding box.

        Parameters:
        vertices (torch.Tensor): A tensor of shape (N, 3) representing the vertices of the mesh.
        scale_factors (torch.Tensor): A tensor of shape (3,) representing the scale factors to apply to the bounding box lengths.

        Returns:
        rescaled_vertices (torch.Tensor): A tensor of shape (N, 3) representing the rescaled vertices.
        """
    if not isinstance(scale_factors, torch.Tensor):
        raise TypeError("Scale factors should be a torch.Tensor")

    if scale_factors.size(0) != 3:
        raise ValueError("Scale factors should be of shape (3,)")

    min_coords, max_coords = compute_bounding_box(vertices)

    # Center the mesh vertices
    center = (min_coords + max_coords) / 2
    centered_vertices = vertices - center

    # Original bounding box size
    original_size = max_coords - min_coords

    # Rescale the mesh vertices
    rescaled_vertices = centered_vertices * scale_factors

    # Translate back to original position if needed
    rescaled_vertices += center

    return rescaled_vertices

# as isometric as possible loss
def aiap_loss(x_canonical, x_deformed, n_neighbors=5):
    """
    Computes the as-isometric-as-possible loss between two sets of points, which measures the discrepancy
    between their pairwise distances.

    Parameters
    ----------
    x_canonical : array-like, shape (n_points, n_dims)
        The canonical (reference) point set, where `n_points` is the number of points
        and `n_dims` is the number of dimensions.
    x_deformed : array-like, shape (n_points, n_dims)
        The deformed (transformed) point set, which should have the same shape as `x_canonical`.
    n_neighbors : int, optional
        The number of nearest neighbors to use for computing pairwise distances.
        Default is 5.

    Returns
    -------
    loss : float
        The AIAP loss between `x_canonical` and `x_deformed`, computed as the L1 norm
        of the difference between their pairwise distances. The loss is a scalar value.
    Raises
    ------
    ValueError
        If `x_canonical` and `x_deformed` have different shapes.
    """

    if x_canonical.shape != x_deformed.shape:
        raise ValueError("Input point sets must have the same shape.")

    _, nn_ix, _ = knn_points(x_canonical.unsqueeze(0),
                             x_canonical.unsqueeze(0),
                             K=n_neighbors,
                             return_sorted=True)

    dists_canonical = torch.cdist(x_canonical[nn_ix], x_canonical[nn_ix])
    dists_deformed = torch.cdist(x_deformed[nn_ix], x_deformed[nn_ix])

    loss = l1_loss(dists_canonical, dists_deformed)

    return loss