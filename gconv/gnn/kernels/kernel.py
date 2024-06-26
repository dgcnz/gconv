"""
kernel.py

Contains base classes for group kernel, separable group kernel, and
lifting kernels.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.init as init
from gconv.utils import unsqueeze_like

from torch import Tensor

import math


class GroupKernel(nn.Module):
    __constants__ = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "group_kernel_size",
        "groups",
        "mask",
        "grid_H",
        "grid_Rn",
        "det_H",
        "inverse_H",
        "left_apply_to_H",
        "left_apply_to_Rn",
        "sample_H",
        "sample_Rn",
        "_group_kernel_dim",
        "sample_H_kwargs",
        "sample_Rn_kwargs",
    ]

    def reset_parameters(self) -> None: ...

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: tuple,
        groups: int = 1,
        grid_H: Optional[Tensor] = None,
        grid_Rn: Optional[Tensor] = None,
        mask: Tensor | None = None,
        det_H: Callable | None = None,
        inverse_H: Callable | None = None,
        left_apply_to_H: Callable | None = None,
        left_apply_to_Rn: Callable | None = None,
        sample_H: Callable | None = None,
        sample_Rn: Callable | None = None,
        sample_H_kwargs: dict = {},
        sample_Rn_kwargs: dict = {},
    ) -> None:
        """
        The group kernel manages the group and sampling of weights.

        Arguments:
            - in_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - group_kernel_size: tuple where each element denotes the size of the
                                 corresponding subgroup weight.
            - groups: int denoting the number of groups for depth-wise separability.
            - grid_H: tensor containing the reference grid of group elements.
            - grid_Rn: tensor containing the reference spatial grid.
            - mask: tensor containing a mask to be applied to spatial kernels if provided.
            - det_H: callable that returns the determinant of the given group elements.
            - inverse_H: callable that returns the inverse of the given group elements.
            - left_apply_to_H: callable that implements the pairwise group action of H
                               given two grids of group elements.
            - left_apply_to_Rn: callable that implements the group product of on Rn
                                given a grid of group elements and a grid of Rn vectors.
            - sample_H: callable that samples the weights for a given grid of
                        group elements, the weights, and the reference grid_H.
            - sample_Rn: callable that samples the weights for a given grids
                         of Rn reference grids transformed by H.
            - sample_H_kwargs: dict containing keyword arguments to be passed to
                               sample_H.
            - sample_Rn_kwargs: dict containing keyword arguments to be passed to
                                sample_Rn.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.group_kernel_size = group_kernel_size
        self._group_kernel_dim = sum(group_kernel_size)

        self.groups = groups

        self.register_buffer("grid_H", grid_H)
        self.register_buffer("grid_Rn", grid_Rn)

        self.register_buffer("mask", mask)

        self.det_H = det_H
        self.inverse_H = inverse_H
        self.left_apply_to_H = left_apply_to_H
        self.left_apply_to_Rn = left_apply_to_Rn
        self.sample_H = sample_H
        self.sample_Rn = sample_Rn
        self.sample_H_kwargs = sample_H_kwargs
        self.sample_Rn_kwargs = sample_Rn_kwargs

    def extra_repr(self):
        s = f"mask={self.mask is not None}"
        if self.sample_H_kwargs:
            s += f", sample_H_kwargs={self.sample_H_kwargs}"
        if self.sample_Rn_kwargs:
            s += f", sample_Rn_kwargs={self.sample_Rn_kwargs}"
        return s


class GLiftingKernel(GroupKernel):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        group_kernel_size: tuple,
        grid_H,
        grid_Rn,
        groups: int = 1,
        mask: Tensor | None = None,
        det_H: Callable | None = None,
        inverse_H: Callable | None = None,
        left_apply_to_Rn: Callable | None = None,
        sample_Rn: Callable | None = None,
        sample_Rn_kwargs: dict = {},
    ) -> None:
        """
        The lifting kernel manages the group and weights for
        lifting an Rn input to Rn x H input.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            groups=groups,
            grid_H=grid_H,
            grid_Rn=grid_Rn,
            mask=mask,
            det_H=det_H,
            inverse_H=inverse_H,
            left_apply_to_Rn=left_apply_to_Rn,
            sample_Rn=sample_Rn,
            sample_Rn_kwargs=sample_Rn_kwargs,
        )

        self.weight = torch.nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )

        # for expanding determinant to correct size
        self.weight_dims = (1,) * (self.weight.ndim - 2)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, H) -> Tensor:
        num_H = H.shape[0]

        H_product = self.left_apply_to_Rn(self.inverse_H(H), self.grid_Rn)

        product_dims = (1,) * (H_product.ndim - 1)

        weight = self.sample_Rn(
            self.weight.repeat_interleave(H.shape[0], dim=0),
            H_product.repeat(self.out_channels, *product_dims),
            **self.sample_Rn_kwargs,
        ).view(
            self.out_channels, num_H, self.in_channels // self.groups, *self.kernel_size
        )

        if self.mask is not None:
            weight = self.mask * weight

        if self.det_H is not None:
            weight = weight / self.det_H(H).view(-1, 1, *self.weight_dims)

        return weight


class RGLiftingKernel(GroupKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filter_banks: int,
        kernel_size,
        group_kernel_size: tuple,
        grid_H,
        grid_Rn,
        groups: int = 1,
        mask: Tensor | None = None,
        det_H: Callable | None = None,
        inverse_H: Callable | None = None,
        left_apply_to_Rn: Callable | None = None,
        sample_Rn: Callable | None = None,
        sample_Rn_kwargs: dict = {},
    ) -> None:
        """
        The lifting kernel manages the group and weights for
        lifting an Rn input to Rn x H input.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            groups=groups,
            grid_H=grid_H,
            grid_Rn=grid_Rn,
            mask=mask,
            det_H=det_H,
            inverse_H=inverse_H,
            left_apply_to_Rn=left_apply_to_Rn,
            sample_Rn=sample_Rn,
            sample_Rn_kwargs=sample_Rn_kwargs,
        )

        self.weight = torch.nn.Parameter(
            torch.empty(
                num_filter_banks, out_channels, in_channels // groups, *self.kernel_size
            )
        )
        if len(group_kernel_size) != 1:
            raise NotImplementedError(
                "Relaxed Group Convolutions only support group kernels of size 1"
            )

        self.num_filter_banks = num_filter_banks
        self.relaxed_weights = torch.nn.Parameter(
            torch.ones(num_filter_banks, group_kernel_size[0])
        )

        # for expanding determinant to correct size
        self.weight_dims = (1,) * (self.weight.ndim - 2)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, H) -> Tensor:
        num_H = H.shape[0]

        H_product = self.left_apply_to_Rn(self.inverse_H(H), self.grid_Rn)

        product_dims = (1,) * (H_product.ndim - 1)

        weight = self.sample_Rn(
            self.weight.flatten(0, 1).repeat_interleave(H.shape[0], dim=0),
            H_product.repeat(self.num_filter_banks * self.out_channels, *product_dims),
            **self.sample_Rn_kwargs,
        ).view(
            self.num_filter_banks,
            self.out_channels,
            num_H,
            self.in_channels // self.groups,
            *self.kernel_size,
        )
        weight = torch.sum(
            self.relaxed_weights.view(
                self.num_filter_banks,
                1,
                self.relaxed_weights.shape[1],
                *product_dims,
            )
            * weight,
            dim=0,
        )

        if self.mask is not None:
            weight = self.mask * weight

        if self.det_H is not None:
            weight = weight / self.det_H(H).view(-1, 1, *self.weight_dims)

        return weight


class GSeparableKernel(GroupKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: tuple,
        grid_H: Tensor,
        grid_Rn: Tensor,
        groups: int = 1,
        mask: Tensor | None = None,
        det_H: Callable | None = None,
        inverse_H: Callable | None = None,
        left_apply_to_H: Callable | None = None,
        left_apply_to_Rn: Callable | None = None,
        sample_H: Callable | None = None,
        sample_Rn: Callable | None = None,
        sample_H_kwargs: dict = {},
        sample_Rn_kwargs: dict = {},
    ) -> None:
        """
        The separable kernel manages the group and weights for
        separable group convolutions, returning weights for
        subgroup H and Rn separately.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size=group_kernel_size,
            grid_H=grid_H,
            grid_Rn=grid_Rn,
            groups=groups,
            mask=mask,
            det_H=det_H,
            inverse_H=inverse_H,
            left_apply_to_H=left_apply_to_H,
            left_apply_to_Rn=left_apply_to_Rn,
            sample_H=sample_H,
            sample_Rn=sample_Rn,
            sample_H_kwargs=sample_H_kwargs,
            sample_Rn_kwargs=sample_Rn_kwargs,
        )

        self.weight_H = nn.Parameter(
            torch.empty(self._group_kernel_dim, out_channels, in_channels // groups)
        )

        self.weight = nn.Parameter(torch.empty(out_channels, 1, *kernel_size))

        # for expanding determinant to correct size
        self.weight_dims = (1,) * len(kernel_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_H, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, in_H: Tensor, out_H: Tensor) -> tuple[Tensor, Tensor]:
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]
        H_dims = in_H.shape[1:]

        out_H_inverse = self.inverse_H(out_H)

        H_product_H = self.left_apply_to_H(out_H_inverse, in_H)
        H_product_Rn = self.left_apply_to_Rn(out_H_inverse, self.grid_Rn)

        product_dims = (1,) * (H_product_Rn.ndim - 1)

        # sample SO3
        weight_H = (
            self.sample_H(
                H_product_H.view(-1, *H_dims),
                self.weight_H.reshape(self._group_kernel_dim, -1),
                self.grid_H,
                **self.sample_H_kwargs,
            )
            .view(
                num_in_H,
                num_out_H,
                self.in_channels // self.groups,
                self.out_channels,
                *self.weight_dims,
            )
            .transpose(0, 3)
            .transpose(1, 3)
        )

        # sample R3
        weight = self.sample_Rn(
            self.weight.repeat_interleave(num_out_H, dim=0),
            H_product_Rn.repeat(self.out_channels, *product_dims),
            **self.sample_Rn_kwargs,
        ).view(
            self.out_channels,
            num_out_H,
            1,
            *self.kernel_size,
        )

        if self.mask is not None:
            weight = self.mask * weight

        if self.det_H is not None:
            weight = weight / self.det_H(out_H).view(-1, 1, *self.weight_dims)

        return weight_H, weight


class RGSeparableKernel(GroupKernel):  # TODO: Implement this
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filter_banks: int,
        kernel_size: tuple,
        group_kernel_size: tuple,
        grid_H: Tensor,
        grid_Rn: Tensor,
        groups: int = 1,
        mask: Tensor | None = None,
        det_H: Callable | None = None,
        inverse_H: Callable | None = None,
        left_apply_to_H: Callable | None = None,
        left_apply_to_Rn: Callable | None = None,
        sample_H: Callable | None = None,
        sample_Rn: Callable | None = None,
        sample_H_kwargs: dict = {},
        sample_Rn_kwargs: dict = {},
    ) -> None:
        """
        The separable kernel manages the group and weights for
        separable group convolutions, returning weights for
        subgroup H and Rn separately.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size=group_kernel_size,
            grid_H=grid_H,
            grid_Rn=grid_Rn,
            groups=groups,
            mask=mask,
            det_H=det_H,
            inverse_H=inverse_H,
            left_apply_to_H=left_apply_to_H,
            left_apply_to_Rn=left_apply_to_Rn,
            sample_H=sample_H,
            sample_Rn=sample_Rn,
            sample_H_kwargs=sample_H_kwargs,
            sample_Rn_kwargs=sample_Rn_kwargs,
        )

        if len(group_kernel_size) != 1:
            raise NotImplementedError(
                "Relaxed Group Convolutions only support group kernels of size 1"
            )
        self.weight_H = nn.Parameter(
            torch.empty(
                num_filter_banks,
                self._group_kernel_dim,
                out_channels,
                in_channels // groups,
            )
        )
        self.num_filter_banks = num_filter_banks
        self.rweight_H = nn.Parameter(
            torch.ones(num_filter_banks, self._group_kernel_dim)
        )

        self.weight = nn.Parameter(torch.empty(out_channels, 1, *kernel_size))
        self.weight_dims = (1,) * len(kernel_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_H, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, in_H: Tensor, out_H: Tensor) -> tuple[Tensor, Tensor]:
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]
        out_H_inverse = self.inverse_H(out_H)

        H_product_H = self.left_apply_to_H(out_H_inverse, in_H)
        H_product_Rn = self.left_apply_to_Rn(out_H_inverse, self.grid_Rn)

        product_dims = (1,) * (H_product_Rn.ndim - 1)
        # TODO: vectorize this for loop
        weight_H = torch.stack(
            [
                self.sample_H(
                    H_product_H.flatten(0, 1),
                    self.weight_H[i].flatten(1, -1),
                    self.grid_H,
                    **self.sample_H_kwargs,
                )
                .view(
                    num_in_H,
                    num_out_H,
                    self.in_channels // self.groups,
                    self.out_channels,
                    *self.weight_dims,
                )
                .transpose(0, 3)
                .transpose(1, 3)
                for i in range(self.num_filter_banks)
            ],
            dim=0,
        )

        # linear combination of filters
        weight_H = torch.sum(
            unsqueeze_like(self.rweight_H[:, None, :], weight_H) * weight_H,
            dim=0,
        )

        # sample R3
        weight = self.sample_Rn(
            self.weight.repeat_interleave(num_out_H, dim=0),
            H_product_Rn.repeat(self.out_channels, *product_dims),
            **self.sample_Rn_kwargs,
        ).view(
            self.out_channels,
            num_out_H,
            1,
            *self.kernel_size,
        )

        if self.mask is not None:
            weight = self.mask * weight

        if self.det_H is not None:
            weight = weight / self.det_H(out_H).view(-1, 1, *self.weight_dims)

        return weight_H, weight


class GSubgroupKernel(GroupKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: tuple,
        grid_H: Tensor,
        groups: int = 1,
        det_H: Callable | None = None,
        inverse_H: Callable | None = None,
        left_apply_to_H: Callable | None = None,
        sample_H: Callable | None = None,
        sample_H_kwargs: dict = {},
    ) -> None:
        """
        The subgroup kernel manages the group and weights for
        convolutions applied only to the subgroup H.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size=group_kernel_size,
            grid_H=grid_H,
            groups=groups,
            det_H=det_H,
            inverse_H=inverse_H,
            left_apply_to_H=left_apply_to_H,
            sample_H=sample_H,
            sample_H_kwargs=sample_H_kwargs,
        )

        self.weight = nn.Parameter(
            torch.empty(self._group_kernel_dim, out_channels, in_channels // groups)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, in_H: Tensor, out_H: Tensor) -> Tensor:
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]
        H_dims = in_H.shape[1:]

        out_H_inverse = self.inverse_H(out_H)

        H_product_H = self.left_apply_to_H(out_H_inverse, in_H)

        # sample SO3
        weight = (
            self.sample_H(
                H_product_H.view(-1, *H_dims),
                self.weight.reshape(self._group_kernel_dim, -1),
                self.grid_H,
                **self.sample_H_kwargs,
            )
            .view(
                num_in_H,
                num_out_H,
                self.in_channels // self.groups,
                self.out_channels,
                *self.kernel_size,
            )
            .transpose(0, 3)
            .transpose(1, 3)
        )

        return weight


class GKernel(GroupKernel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: tuple,
        grid_H: Tensor,
        grid_Rn: Tensor,
        groups: int = 1,
        mask: Tensor | None = None,
        det_H: Callable | None = None,
        inverse_H: Callable | None = None,
        left_apply_to_H: Callable | None = None,
        left_apply_to_Rn: Callable | None = None,
        sample_H: Callable | None = None,
        sample_Rn: Callable | None = None,
        sample_H_kwargs: dict = {},
        sample_Rn_kwargs: dict = {},
    ) -> None:
        """
        Manages the group and weights for the full group convolution.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size=group_kernel_size,
            grid_H=grid_H,
            grid_Rn=grid_Rn,
            groups=groups,
            mask=mask,
            det_H=det_H,
            inverse_H=inverse_H,
            left_apply_to_H=left_apply_to_H,
            left_apply_to_Rn=left_apply_to_Rn,
            sample_H=sample_H,
            sample_Rn=sample_Rn,
            sample_H_kwargs=sample_H_kwargs,
            sample_Rn_kwargs=sample_Rn_kwargs,
        )

        self.weight = nn.Parameter(
            torch.empty(
                self._group_kernel_dim,
                out_channels,
                in_channels // groups,
                *kernel_size,
            )
        )

        # for expanding determinant to correct size
        self.weight_dims = (1,) * (self.weight.ndim - 2)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, in_H: Tensor, out_H: Tensor) -> Tensor:
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]
        H_dims = in_H.shape[1:]

        out_H_inverse = self.inverse_H(out_H)

        H_product_H = self.left_apply_to_H(out_H_inverse, in_H)
        H_product_Rn = self.left_apply_to_Rn(out_H_inverse, self.grid_Rn)

        # sample SO3
        weight = self.sample_H(
            H_product_H.view(-1, *H_dims),
            self.weight.reshape(self._group_kernel_dim, -1),
            self.grid_H,
            **self.sample_H_kwargs,
        ).view(
            num_out_H * num_in_H,
            (self.in_channels // self.groups) * self.out_channels,
            *self.kernel_size,
        )

        # sample R3
        weight = (
            self.sample_Rn(
                weight,
                H_product_Rn.repeat_interleave(num_in_H, dim=0),
                **self.sample_Rn_kwargs,
            )
            .view(
                num_out_H,
                num_in_H,
                self.in_channels // self.groups,
                self.out_channels,
                *self.kernel_size,
            )
            .transpose(0, 3)
            .transpose(1, 3)
        )

        if self.mask is not None:
            weight = self.mask * weight

        if self.det_H is not None:
            weight = weight / self.det_H(out_H).view(-1, 1, *self.weight_dims)

        return weight
