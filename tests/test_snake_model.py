import torch

from paco.snake.model import CircularConv1d, SnakeRefiner, refine_vertices


def test_circular_conv_preserves_vertex_count():
    conv = CircularConv1d(4, 8, kernel_size=3)
    x = torch.randn(2, 4, 17)
    y = conv(x)
    assert y.shape == (2, 8, 17)


def test_snake_forward_shape():
    model = SnakeRefiner(feature_dim=6, hidden_dim=16, num_layers=2, max_offset=4.0)
    vertices = torch.randn(3, 12, 2)
    features = torch.randn(3, 12, 6)

    offsets = model(vertices, features)
    assert offsets.shape == (3, 12, 2)
    assert torch.isfinite(offsets).all()
    assert offsets.abs().max() <= 4.0


def test_iterative_refinement_is_shape_stable():
    model = SnakeRefiner(feature_dim=5, hidden_dim=16, num_layers=1)
    vertices = torch.tensor(
        [[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]], dtype=torch.float32
    )

    def sampler(points):
        assert points.shape == (4, 2)
        return torch.zeros((4, 5), dtype=torch.float32)

    refined = refine_vertices(model, vertices, (10, 10), sampler, steps=2)
    assert refined.shape == vertices.shape
    assert refined[:, 0].min() >= 0
    assert refined[:, 0].max() <= 9
    assert refined[:, 1].min() >= 0
    assert refined[:, 1].max() <= 9
