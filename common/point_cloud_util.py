import torch


def normalize_point_cloud(point_cloud, use_center_of_bounding_box=True):
    min_x, max_x = torch.min(point_cloud[:, 0]), torch.max(point_cloud[:, 0])
    min_y, max_y = torch.min(point_cloud[:, 1]), torch.max(point_cloud[:, 1])
    min_z, max_z = torch.min(point_cloud[:, 2]), torch.max(point_cloud[:, 2])
    # center the point cloud
    if use_center_of_bounding_box:
        center = torch.tensor([(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2])
    else:
        center = torch.mean(point_cloud, dim=0)
    point_cloud = point_cloud - center
    dist = torch.max(torch.sqrt(torch.sum((point_cloud ** 2), dim=1)))
    point_cloud = point_cloud / dist  # scale the point cloud
    return point_cloud
