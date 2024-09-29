import torch

def chamfer_distance(point_set_a, point_set_b):
    point_set_a = torch.tensor(point_set_a)
    point_set_b = torch.tensor(point_set_b)

    # Verify batch dimensions
    assert point_set_a.shape[-3:] == point_set_b.shape[-3:], "Batch dimensions do not match"

    # Verify that the last axis of the tensors has the same dimension
    dimension = point_set_a.shape[-1]
    assert point_set_b.shape[-1] == dimension, "Last axis dimension does not match"

    # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of dimension D)
    difference = point_set_a.unsqueeze(-2) - point_set_b.unsqueeze(-3)

    # Calculate the square distances between each two points: |ai - bj|^2
    square_distances = (difference * difference).sum(dim=-1)

    minimum_square_distance_a_to_b = square_distances.min(dim=-1)[0]
    minimum_square_distance_b_to_a = square_distances.min(dim=-2)[0]

    return (minimum_square_distance_a_to_b.mean() + minimum_square_distance_b_to_a.mean())
