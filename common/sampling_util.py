import torch
from dgl.geometry import farthest_point_sampler


def farthest_point_sampling(faces, vertices, num_points=1000):
    random_sampling = sample_surface(faces, vertices, num_points=30000)
    point_cloud_indices = farthest_point_sampler(random_sampling.unsqueeze(0), npoints=num_points)
    point_cloud = random_sampling[point_cloud_indices[0]]
    return point_cloud


def face_areas_normals(faces, vs):
    face_normals = torch.cross(vs[:, faces[:, 1], :] - vs[:, faces[:, 0], :],
                               vs[:, faces[:, 2], :] - vs[:, faces[:, 1], :], dim=2)
    face_areas = torch.norm(face_normals, dim=2)
    face_normals = face_normals / face_areas[:, :, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals


def sample_surface(faces, vertices, num_points=1000):
    """
    sample mesh surface
    Args
    ---------
    vertices: vertices
    faces: triangle faces (torch.long)
    num_points: number of samples in the final point cloud
    Return
    ---------
    samples: (count, 3) points in space on the surface of mesh
    normals: (count, 3) corresponding face normals for points
    """
    bsize, nvs, _ = vertices.shape
    weights, normal = face_areas_normals(faces, vertices)
    weights_sum = torch.sum(weights, dim=1)
    dist = torch.distributions.categorical.Categorical(probs=weights / weights_sum[:, None])
    face_index = dist.sample((num_points,))

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vertices[:, faces[:, 0], :]
    tri_vectors = vertices[:, faces[:, 1:], :].clone()
    tri_vectors -= tri_origins.repeat(1, 1, 2).reshape((bsize, len(faces), 2, 3))

    # pull the vectors for the faces we are going to sample from
    face_index = face_index.transpose(0, 1)
    face_index = face_index[:, :, None].expand((bsize, num_points, 3))
    tri_origins = torch.gather(tri_origins, dim=1, index=face_index)
    face_index2 = face_index[:, :, None, :].expand((bsize, num_points, 2, 3))
    tri_vectors = torch.gather(tri_vectors, dim=1, index=face_index2)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = torch.rand(num_points, 2, 1, device=vertices.device, dtype=tri_vectors.dtype)

    # points will be distributed on a quadrilateral if we use 2x [0-1] samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = torch.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths[None, :]).sum(dim=2)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    # normals = torch.gather(normal, dim=1, index=face_index)

    # return samples, normals
    return samples[0]
