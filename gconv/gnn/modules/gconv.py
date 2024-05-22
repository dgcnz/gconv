"""
gconv.py

Implements group convolution base modules.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple, _triple

from gconv.gnn.kernels import (
    GKernel,
    GLiftingKernel,
    GroupKernel,
    GSeparableKernel,
    GSubgroupKernel,
)

__all__ = [
    "GConvLifting2d",
    "GConvLifting3d",
    "GConvSeparable2d",
    "GConvSeparable3d",
    "GConv2d",
    "GConv3d",
]


class GroupConvNd(nn.Module):
    __constants__ = ["stride", "padding", "dilation", "padding_mode"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: int | tuple,
        kernel: GroupKernel,
        groups: int = 1,
        stride: int = 1,
        padding: tuple | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",  # NOTE: I like it this way
        conv_mode: Literal["2d", "3d", "3d_transposed"] = "3d",
        bias: bool = False,
        output_padding: int = 0 , 
    ) -> None:
        """
        :param in_channels: int denoting the number of input channels.
        :param out_channels: int denoting the number of output channels.
        :param kernel_size: tuple denoting the spatial kernel size.
        :param group_kernel_size: int or tuple denoting the group kernel size.
        :param kernel: GroupKernel that manages the group and samples weights.
        :param groups: int denoting the number of groups for depth-wise separability.
        :param stride: int denoting the stride.
        :param padding: int or denoting padding.
        :param dilation: int denoting dilation.
        :param padding_mode: str denoting the padding mode.
        :param bias: bool that if true will initialzie bias parameters.
        :param output_padding: output padding for the transposed convolution.
        """
        super().__init__()
        if isinstance(group_kernel_size, tuple) and (
            any(i < 0 for i in group_kernel_size) or sum(group_kernel_size) <= 0
        ):
            raise ValueError("group_kernel_size must contain positive integers")
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings
                    )
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(
                    valid_padding_modes, padding_mode
                )
            )

        self.kernel = kernel

        # just for easy access
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_kernel_size = group_kernel_size
        self.groups = groups

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding

        self.padding_mode = padding_mode

        if conv_mode == "2d":
            self._conv_forward = self._conv2d_forward
            bias_shape = (1, 1, 1)
            self.padding = _pair(padding) if isinstance(self.padding, int) else padding
        elif conv_mode == "3d":
            self._conv_forward = self._conv3d_forward
            bias_shape = (1, 1, 1, 1)
            self.padding = (
                _triple(padding) if isinstance(self.padding, int) else padding
            )
        elif conv_mode == "3d_transposed":
            if self.padding_mode != "zeros":
                raise ValueError(
                    "padding_mode must be zeros for transposed convolution"
                )
            self._conv_forward = self._conv3d_transposed_forward
            bias_shape = (1, 1, 1, 1)
            self.padding = (
                _triple(padding) if isinstance(self.padding, int) else padding
            )

        else:
            raise ValueError(
                f"Unspported conv mode: got {conv_mode=}, expected `2d`, `3d` or 3d_transposed."
            )

        # init padding settings
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel.kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation,
                    kernel.kernel_size,
                    range(len(kernel.kernel_size) - 1, -1, -1),
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, *bias_shape))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.kernel.reset_parameters()

        if self.bias is None:
            return

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.kernel.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _conv2d_forward(
        self, input: Tensor, weight: Tensor, groups: int, padding: int | None = None
    ):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                None,
                self.stride,
                _pair(0),
                self.dilation,
                groups,
            )
        if padding is None:
            padding = self.padding
        return F.conv2d(
            input, weight, None, self.stride, padding, self.dilation, groups
        )

    def _conv3d_forward(
        self, input: Tensor, weight: Tensor, groups: int, padding: int | None = None
    ):
        print("NORMAL")
        print(groups)
        print(self.groups)
        print(input.shape)
        print(weight.shape)
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                None,
                self.stride,
                _triple(0),
                self.dilation,
                groups,
            )
        if padding is None:
            padding = self.padding

        
        return F.conv3d(
            input, weight, None, self.stride, padding, self.dilation, groups
        )

    def _conv3d_transposed_forward(self, input: Tensor, weight: Tensor, groups: int, padding: int | None = None):
        """ Performs a transposed conv3d, commonly used to upsample. """
        #print("TRANSPOSED")
        #print(groups)
        #print(self.groups)
        #print(input.shape)
        #print(weight.shape)
        if self.padding_mode != "zeros":
            raise ValueError("padding_mode must be zero for transposed conv")
        if padding is None:
            padding = self.padding
        out_channels, in_channels, D, H , W = weight.shape
        in_channels *= groups   #Switching these around
        out_channels //= groups
        weight_new = weight.reshape(in_channels,out_channels,D,H,W) #Transposed needs a different shape
        #print(weight_new.shape)
        
        return F.conv_transpose3d(
            input, weight_new, None, self.stride, padding,self.output_padding, groups, self.dilation,
        )
        

    def extra_repr(self):
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, group_kernel_size={self.group_kernel_size}, stride={self.stride}"
        if self.padding != (0,) * len(self.padding):
            s += f", padding={self.padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode}"
        return s


class GLiftingConvNd(GroupConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: int,
        kernel: GroupKernel,
        groups: int = 1,
        stride: int = 1,
        padding: tuple | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        conv_mode: str = "3d",
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            kernel,
            groups,
            stride,
            padding,
            dilation,
            padding_mode,
            conv_mode,
            bias,
        )

    def forward(self, input: Tensor, H: Tensor) -> tuple[Tensor, Tensor]:
        N = input.shape[0]
        num_out_H = H.shape[0]

        weight = self.kernel(H)

        input = self._conv_forward(
            input,
            weight.reshape(-1, self.in_channels // self.groups, *self.kernel_size),
            self.groups,
        )

        input = input.view(N, self.out_channels, num_out_H, *input.shape[2:])

        if self.bias is not None:
            input = input + self.bias

        return input, H


class GSeparableConvNd(GroupConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: int,
        kernel: GroupKernel,
        groups: int = 1,
        stride: int = 1,
        padding: tuple | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        conv_mode: str = "3d",        
        bias: bool = False,
        output_padding: int = 0,
    ) -> None:
        """
        Implementation the Nd separable group convolution.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            kernel,
            groups,
            stride,
            padding,
            dilation,
            padding_mode,
            conv_mode,
            bias,
            output_padding=output_padding
        )

    def forward(
        self, input: Tensor, in_H: Tensor, out_H: Tensor
    ) -> tuple[Tensor, Tensor]:
        N, _, _, *input_dims = input.shape
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]

        weight_H, weight = self.kernel(in_H, out_H)
        assert weight_H.shape[4:] == len(self.kernel_size) * (
            1,
        ), f"Pointwise kernel must have size 1. vs {weight_H.shape[4:]}"
        # subgroup conv
        input = self._conv_forward(
            input.reshape(N, self.in_channels * num_in_H, *input_dims),
            weight_H.reshape(
                self.out_channels * num_out_H,
                (self.in_channels // self.groups) * num_in_H,
                *weight_H.shape[4:],
            ),
            self.groups,
            padding=0,  # no padding for pointwise conv
        )

        # spatial conv
        input = self._conv_forward(
            input,
            weight.reshape(
                self.out_channels * num_out_H,
                1,
                *self.kernel_size,
            ),
            self.out_channels * num_out_H,
        )

        input = input.view(N, self.out_channels, num_out_H, *input.shape[2:])

        if self.bias is not None:
            input = input + self.bias

        return input, out_H


class GConvNd(GroupConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        group_kernel_size: int,
        kernel: GroupKernel,
        groups: int = 1,
        stride: int = 1,
        padding: tuple | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        conv_mode: str = "3d",
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            kernel,
            groups,
            stride,
            padding,
            dilation,
            padding_mode,
            conv_mode,
            bias,
        )

    def forward(
        self, input: Tensor, in_H: Tensor, out_H: Tensor
    ) -> tuple[Tensor, Tensor]:
        N, _, _, *input_dims = input.shape
        num_in_H, num_out_H = in_H.shape[0], out_H.shape[0]

        weight = self.kernel(in_H, out_H)

        input = self._conv_forward(
            input.reshape(N, self.in_channels * num_in_H, *input_dims),
            weight.reshape(
                self.out_channels * num_out_H,
                (self.in_channels // self.groups) * num_in_H,
                *self.kernel_size,
            ),
            self.groups,
        )

        input = input.view(N, self.out_channels, num_out_H, *input.shape[2:])

        if self.bias is not None:
            input = input + self.bias

        return input, out_H


class GLiftingConv2d(GLiftingConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int | tuple,
        kernel: GLiftingKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        """
        Implements 2d lifting convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - group_kernel_size: int or tuple denoting the group kernel size.
                                    In the case of a tuple, each element denotes
                                    a separate kernel size for each subgroup. For
                                    example, (4, 2) could denote a O3 kernel with
                                    rotation and reflection kernels of size 4 and
                                    2, respectively.
            - kernel: GroupKernel that manages the group and samples weights.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - padding_mode: str denoting the padding mode.
            - bias: bool that if true will initialzie bias parameters.
        """
        super().__init__(
            in_channels,
            out_channels,
            _pair(kernel_size),
            group_kernel_size,
            kernel,
            groups,
            _pair(stride),
            padding,
            _pair(dilation),
            padding_mode,
            conv_mode="2d",
            bias=bias,
        )


class GSeparableConv2d(GSeparableConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int | tuple,
        kernel: GSubgroupKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        """
        Implements 2d separable group convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - group_kernel_size: int or tuple denoting the group kernel size.
                                    In the case of a tuple, each element denotes
                                    a separate kernel size for each subgroup. For
                                    example, (4, 2) could denote a O3 kernel with
                                    rotation and reflection kernels of size 4 and
                                    2, respectively.
            - kernel: GroupKernel that manages the group and samples weights.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - padding_mode: str denoting the padding mode.
            - bias: bool that if true will initialzie bias parameters.
        """
        super().__init__(
            in_channels,
            out_channels,
            _pair(kernel_size),
            group_kernel_size,
            kernel,
            groups,
            _pair(stride),
            padding,
            _pair(dilation),
            padding_mode,
            conv_mode="2d",
            bias=bias,
        )


class GConv2d(GConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int | tuple,
        kernel: GKernel | GSubgroupKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        """
        Implements 2d group convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - group_kernel_size: int or tuple denoting the group kernel size.
                                    In the case of a tuple, each element denotes
                                    a separate kernel size for each subgroup. For
                                    example, (4, 2) could denote a O3 kernel with
                                    rotation and reflection kernels of size 4 and
                                    2, respectively.
            - kernel: GroupKernel that manages the group and samples weights.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - padding_mode: str denoting the padding mode.
            - bias: bool that if true will initialzie bias parameters.
        """
        super().__init__(
            in_channels,
            out_channels,
            _pair(kernel_size),
            group_kernel_size,
            kernel,
            groups,
            _pair(stride),
            padding,
            _pair(dilation),
            padding_mode,
            conv_mode="2d",
            bias=bias,
        )


class GLiftingConv3d(GLiftingConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int,
        kernel: GLiftingKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        """
        Implements 3d lifting convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - group_kernel_size: int or tuple denoting the group kernel size.
                                    In the case of a tuple, each element denotes
                                    a separate kernel size for each subgroup. For
                                    example, (4, 2) could denote a O3 kernel with
                                    rotation and reflection kernels of size 4 and
                                    2, respectively.
            - kernel: GroupKernel that manages the group and samples weights.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - padding_mode: str denoting the padding mode.
            - bias: bool that if true will initialzie bias parameters.
        """
        super().__init__(
            in_channels,
            out_channels,
            _triple(kernel_size),
            group_kernel_size,
            kernel,
            groups,
            _triple(stride),
            (padding),
            _triple(dilation),
            padding_mode,
            conv_mode="3d",
            bias=bias,
        )


class GSeparableConv3d(GSeparableConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int,
        kernel: GSeparableKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
        conv_mode: Literal["2d", "3d", "3d_transposed"] = "3d",
        output_padding: int=0,
    ) -> None:
        """
        Implements 3d separable group convolution.

        :param int_channels: int denoting the number of input channels.
        :param out_channels: int denoting the number of output channels.
        :param kernel_size: tuple denoting the spatial kernel size.
        :param group_kernel_size: int or tuple denoting the group kernel size.
                              In the case of a tuple, each element denotes
                              a separate kernel size for each subgroup. For
                              example, (4, 2) could denote a O3 kernel with
                              rotation and reflection kernels of size 4 and
                              2, respectively.
        :param kernel: GroupKernel that manages the group and samples weights.
        :param groups: int denoting the number of groups for depth-wise separability.
        :param stride: int denoting the stride.
        :param padding: int or denoting padding.
        :param dilation: int denoting dilation.
        :param padding_mode: str denoting the padding mode.
        :param bias: bool that if true will initialzie bias parameters.
        :param conv_mode: str denoting the convolution mode. Supports 2d, 3d and 3d_transposed.
        :param output_padding: output padding for the transposed convolution.
        """
        super().__init__(
            in_channels,
            out_channels,
            _triple(kernel_size),
            group_kernel_size,
            kernel,
            groups,
            _triple(stride),
            padding,
            _triple(dilation),
            padding_mode,
            bias=bias,
            conv_mode=conv_mode,
            output_padding=output_padding,
        )


class GConv3d(GConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int,
        kernel: GKernel | GSubgroupKernel,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = False,
    ) -> None:
        """
        Implements 3d group convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - group_kernel_size: int or tuple denoting the group kernel size.
                                    In the case of a tuple, each element denotes
                                    a separate kernel size for each subgroup. For
                                    example, (4, 2) could denote a O3 kernel with
                                    rotation and reflection kernels of size 4 and
                                    2, respectively.
            - kernel: GroupKernel that manages the group and samples weights.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - padding_mode: str denoting the padding mode.
            - bias: bool that if true will initialzie bias parameters.
        """
        super().__init__(
            in_channels,
            out_channels,
            _triple(kernel_size),
            group_kernel_size,
            kernel,
            groups,
            _triple(stride),
            padding,
            _triple(dilation),
            padding_mode,
            conv_mode="3d",
            bias=bias,
        )
