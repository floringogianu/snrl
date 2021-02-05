import torch


def conv2mat(kernel, image_shape):
    result_dims = torch.tensor(image_shape[1:]) - torch.tensor(kernel.shape[2:]) + 1

    d = len(kernel.shape[2:])

    x = torch.nn.functional.pad(
        kernel,
        tuple(
            torch.stack(
                (torch.zeros_like(result_dims), result_dims.flip(0)), 1
            ).flatten()
        ),
    )
    grid = torch.stack(
        torch.meshgrid(
            *map(torch.arange, (kernel.shape[0], *result_dims, *image_shape))
        ),
        -1,
    )
    y = (grid[..., d + 2 :] - grid[..., 1 : d + 1]) % torch.tensor(image_shape[1:])
    m = x[
        (
            grid[..., 0],
            grid[..., d + 1],
            *y.permute(*torch.arange(len(y.shape)).roll(1, 0)),
        )
    ]
    return m.flatten(end_dim=len(kernel.shape[2:])).flatten(start_dim=1)
