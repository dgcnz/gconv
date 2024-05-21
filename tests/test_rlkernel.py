import torch
from gconv.geometry.groups import so3
import gconv.gnn.functional as gF
import torch.nn.functional as F
import pytest


@pytest.mark.parametrize("num_filter_banks", [1, 2])
@pytest.mark.parametrize("in_channels", [3, 6])
@pytest.mark.parametrize("out_channels", [6, 9])
@pytest.mark.parametrize("groups", [1, 3])
@pytest.mark.parametrize("kernel_size", [5])
def test_vectorized_relaxed_lifting_kernel(
    num_filter_banks, in_channels, out_channels, groups, kernel_size
):
    """Test vectorized relaxed lifting kernel"""
    H = so3.quat_to_matrix(so3.octahedral())
    sample_Rn_kwargs = {"mode": "bilinear", "padding_mode": "border"}
    dim = 3
    grid_Rn = gF.create_grid_R3(kernel_size)
    H_product = so3.left_apply_to_R3(so3.matrix_inverse(H), grid_Rn)

    product_dims = (1,) * (H_product.ndim - 1)
    kernel_size_expanded = (kernel_size,) * dim
    sample_Rn_kwargs = {"mode": "bilinear", "padding_mode": "border"}
    weight = torch.rand(
        num_filter_banks, out_channels, in_channels // groups, *(kernel_size_expanded)
    )

    weight_vec = F.grid_sample(
        weight.flatten(0, 1).repeat_interleave(H.shape[0], dim=0),
        H_product.repeat(num_filter_banks * out_channels, *product_dims),
        **sample_Rn_kwargs,
    ).view(
        num_filter_banks,
        out_channels,
        H.shape[0],
        in_channels // groups,
        *kernel_size_expanded,
    )
    weight_vec = F.grid_sample(
        weight.flatten(0, 1).repeat_interleave(H.shape[0], dim=0),
        H_product.repeat(num_filter_banks * out_channels, *product_dims),
        **sample_Rn_kwargs,
    ).view(
        num_filter_banks,
        out_channels,
        H.shape[0],
        in_channels // groups,
        *kernel_size_expanded,
    )

    weights_sec = []
    for i in range(num_filter_banks):
        weight_sec_ = F.grid_sample(
            weight[i].repeat_interleave(H.shape[0], dim=0),
            H_product.repeat(out_channels, *product_dims),
            **sample_Rn_kwargs,
        ).view(
            out_channels,
            H.shape[0],
            in_channels // groups,
            *kernel_size_expanded,
        )
        weights_sec.append(weight_sec_)
    weight_sec = torch.stack(weights_sec, dim=0)
    assert torch.allclose(weight_vec, weight_sec)
