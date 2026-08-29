import torch


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def ellipsoid_surface(mean, cov, n_std=2.0, resolution=20):
    """
    Returns (X, Y, Z) grids (each resolution x resolution) tracing the
    n_std-standard-deviation ellipsoid surface of a 3D Gaussian, suitable for
    ax.plot_surface / ax.plot_wireframe.
    """
    mean = torch.as_tensor(mean, dtype=torch.float32)
    cov = torch.as_tensor(cov, dtype=torch.float32)

    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = torch.clamp(eigvals, min=0.0)
    radii = n_std * torch.sqrt(eigvals)

    u = torch.linspace(0, 2 * torch.pi, resolution, device="cuda")
    v = torch.linspace(0, torch.pi, resolution, device="cuda")
    x = torch.outer(torch.cos(u), torch.sin(v))
    y = torch.outer(torch.sin(u), torch.sin(v))
    z = torch.outer(torch.ones_like(u), torch.cos(v))
    unit_sphere = torch.stack([x, y, z], dim=-1)  # (res, res, 3)

    ellipsoid = unit_sphere * radii.cuda()  # scale along principal axes
    ellipsoid = ellipsoid @ eigvecs.transpose(-2, -1)  # rotate into world frame
    ellipsoid = ellipsoid + mean

    return ellipsoid[..., 0].cpu().numpy(), ellipsoid[..., 1].cpu().numpy(), ellipsoid[..., 2].cpu().numpy()


def ellipse_points(mean, cov2d, n_std=2.0, num_points=100):
    """
    Returns (x, y) arrays (num_points,) tracing the n_std ellipse boundary of
    a 2D Gaussian, suitable for ax.plot.
    """
    mean = torch.as_tensor(mean, dtype=torch.float32)
    cov2d = torch.as_tensor(cov2d, dtype=torch.float32)

    eigvals, eigvecs = torch.linalg.eigh(cov2d)
    eigvals = torch.clamp(eigvals, min=0.0)
    radii = n_std * torch.sqrt(eigvals)

    theta = torch.linspace(0, 2 * torch.pi, num_points, device="cuda")
    circle = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # (num_points, 2)

    ellipse = circle * radii
    ellipse = ellipse @ eigvecs.transpose(-2, -1)
    ellipse = ellipse + mean

    return ellipse[:, 0].cpu().numpy(), ellipse[:, 1].cpu().numpy()
