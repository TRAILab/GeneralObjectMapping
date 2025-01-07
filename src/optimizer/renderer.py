"""
IO Renderer to deal with differentiable rendering.
"""

# path to shap-e official library
path_shape_lib = "./shap-e"
import sys

sys.path.insert(0, path_shape_lib)

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from shap_e.util.collections import AttrDict


def append_tensor(val_list: Optional[List[torch.Tensor]], output: Optional[torch.Tensor]):
    if val_list is None:
        return [output]
    return val_list + [output]


def render_views_from_rays(
    rays,
    render_rays: Callable[[AttrDict, AttrDict, AttrDict], AttrDict],
    ray_batch_size: int = 1024,
    params: Optional[Dict] = None,
    options: Optional[Dict] = None,
    cam_z: Optional[torch.Tensor] = None,  # if you need depth map
) -> AttrDict:
    """
    A function refered to shap-e/shap_e/models/renderer.py

    The original function renders for a whole image;
    The new function renders for given rays.

    @ rays: [ray_num, 2, 3]
    """
    batch_size = 1
    inner_shape = [1]

    # length of ray
    ray_num = rays.shape[0]  # TODO: Check ray num

    camera = AttrDict()
    camera.height = 1
    camera.width = ray_num  # automatically infer from rays  (In fact, it's the number of rays)

    inner_batch_size = int(np.prod(inner_shape))

    # mip-NeRF radii calculation from: https://github.com/google/mipnerf/blob/84c969e0a623edd183b75693aed72a7e7c22902d/internal/datasets.py#L193-L200
    directions = rays.view(batch_size, inner_batch_size, camera.height, camera.width, 2, 3)[
        ..., 1, :
    ]
    neighbor_dists = torch.linalg.norm(directions[:, :, :, 1:] - directions[:, :, :, :-1], dim=-1)
    neighbor_dists = torch.cat([neighbor_dists, neighbor_dists[:, :, :, -2:-1]], dim=3)
    radii = (neighbor_dists * 2 / np.sqrt(12)).view(batch_size, -1, 1)

    rays = rays.view(batch_size, inner_batch_size * camera.height * camera.width, 2, 3)

    if cam_z is not None:
        z_directions = (
            (cam_z / torch.linalg.norm(cam_z, dim=-1, keepdim=True))
            .reshape([batch_size, inner_batch_size, 1, 3])
            .repeat(1, 1, camera.width * camera.height, 1)
            .reshape(1, inner_batch_size * camera.height * camera.width, 3)
        )

    # ray_batch_size = batch.get("ray_batch_size", batch.get("inner_batch_size", 4096))
    n_batches = rays.shape[1] // ray_batch_size
    remainder = rays.shape[1] % ray_batch_size
    if remainder != 0:
        n_batches += 1

    output_list = AttrDict(aux_losses=dict())

    for idx in range(n_batches):
        start_idx = idx * ray_batch_size
        end_idx = (
            (idx + 1) * ray_batch_size
            if (idx != n_batches - 1 or (n_batches == 1 and idx == 0))
            else start_idx + remainder
        )
        rays_batch = AttrDict(
            rays=rays[:, start_idx:end_idx],
            radii=radii[:, start_idx:end_idx],
        )
        output = render_rays(rays_batch, params=params, options=options)

        if cam_z is not None:
            z_batch = z_directions[:, idx * ray_batch_size : (idx + 1) * ray_batch_size]
            ray_directions = rays_batch.rays[:, :, 1]
            z_dots = (ray_directions * z_batch).sum(-1, keepdim=True)
            output.depth = output.distances * z_dots

        output_list = output_list.combine(output, append_tensor)

    def _resize(val_list: List[torch.Tensor]):
        val = torch.cat(val_list, dim=1)
        assert val.shape[1] == inner_batch_size * camera.height * camera.width
        return val.view(batch_size, *inner_shape, camera.height, camera.width, -1)

    def _avg(_key: str, loss_list: List[torch.Tensor]):
        return sum(loss_list) / n_batches

    output = AttrDict(
        {name: _resize(val_list) for name, val_list in output_list.items() if name != "aux_losses"}
    )
    output.aux_losses = output_list.aux_losses.map(_avg)

    return output
