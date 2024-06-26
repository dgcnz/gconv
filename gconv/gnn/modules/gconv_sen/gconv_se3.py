from typing import Optional, Literal
from torch import Tensor

from gconv.gnn.modules.gconv import GLiftingConv3d, GSeparableConv3d, GConv3d
from gconv.gnn.kernels import (
    GLiftingKernelSE3,
    GSeparableKernelSE3,
    GKernelSE3,
    RGLiftingKernelSE3,
    RGSeparableKernelSE3,
)

from gconv.geometry import so3


class GLiftingConvSE3(GLiftingConv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        group_kernel_size: int = 4,
        grid_H: Optional[Tensor] = None,
        padding_mode: str = "zeros",
        permute_output_grid: bool = True,
        sampling_mode="bilinear",
        sampling_padding_mode="border",
        bias: bool = False,
        mask: bool = True,
    ) -> None:
        """
        Implements SE3 lifting convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - group_kernel_size: int denoting the group kernel size (default 4).
            - grid_H: tensor of shape (N, 3, 3) of SO3 elements (rotation matrices). If
                      not provided, a uniform grid will be initalizd of size group_kernel_size.
                      If provided, group_kernel_size will be set to N.
            - padding_mode: str denoting the padding mode.
            - permute_output_grid: bool that if true will randomly permute output group grid
                                   for estimating continuous groups.
            - sampling_mode: mode used for sampling weights. Supports bilinear (default) or nearest.
            - sampling_padding_mode: padding mode for weight sampling, border (default) is recommended.
            - bias: bool that if true, will initialzie bias parameters.
            - mask: bool that if true, will initialize spherical mask applied to spatial weights.
        """
        kernel = GLiftingKernelSE3(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size=group_kernel_size,
            groups=groups,
            sampling_mode=sampling_mode,
            sampling_padding_mode=sampling_padding_mode,
            mask=mask,
            grid_H=grid_H,
        )

        self.permute_output_grid = permute_output_grid

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
            bias,
        )

    def forward(
        self, input: Tensor, H: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        if H is None:
            H = self.kernel.grid_H

        if self.permute_output_grid:
            H = so3.left_apply_matrix(so3.random_matrix(1, device=input.device), H)

        return super().forward(input, H)


class RGLiftingConvSE3(GLiftingConv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filter_banks: int,
        kernel_size: int,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        group_kernel_size: int = 4,
        grid_H: Optional[Tensor] = None,
        padding_mode: str = "zeros",
        permute_output_grid: bool = True,
        sampling_mode="bilinear",
        sampling_padding_mode="border",
        bias: bool = False,
        mask: bool = True,
    ) -> None:
        """
        Implements SE3 lifting convolution.

        :param int_channels: int denoting the number of input channels.
        :param out_channels: int denoting the number of output channels.
        :param num_filter_banks: int denoting the number of filter banks.
        :param kernel_size: tuple denoting the spatial kernel size.
        :param groups: int denoting the number of groups for depth-wise separability.
        :param stride: int denoting the stride.
        :param padding: int or denoting padding.
        :param dilation: int denoting dilation.
        :param group_kernel_size: int denoting the group kernel size (default 4).
        :param grid_H: tensor of shape (N, 3, 3) of SO3 elements (rotation matrices). If
                not provided, a uniform grid will be initalizd of size group_kernel_size.
                If provided, group_kernel_size will be set to N.
        :param padding_mode: str denoting the padding mode.
        :param permute_output_grid: bool that if true will randomly permute output group grid
                             for estimating continuous groups.
        :param sampling_mode: mode used for sampling weights. Supports bilinear (default) or nearest.
        :param sampling_padding_mode: padding mode for weight sampling, border (default) is recommended.
        :param bias: bool that if true, will initialzie bias parameters.
        :param mask: bool that if true, will initialize spherical mask applied to spatial weights.
        """
        kernel = RGLiftingKernelSE3(
            in_channels,
            out_channels,
            num_filter_banks,
            kernel_size,
            group_kernel_size=group_kernel_size,
            groups=groups,
            sampling_mode=sampling_mode,
            sampling_padding_mode=sampling_padding_mode,
            mask=mask,
            grid_H=grid_H,
        )

        self.permute_output_grid = permute_output_grid

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
            bias,
        )

    def forward(
        self, input: Tensor, H: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        if H is None:
            H = self.kernel.grid_H

        if self.permute_output_grid:
            H = so3.left_apply_matrix(so3.random_matrix(1, device=input.device), H)

        return super().forward(input, H)


class GSeparableConvSE3(GSeparableConv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int = 4,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        permute_output_grid: bool = True,
        group_sampling_mode: str = "rbf",
        group_sampling_width: float = 0.0,
        spatial_sampling_mode: str = "bilinear",
        spatial_sampling_padding_mode: str = "border",
        mask: bool = True,
        bias: bool = False,
        grid_H: Optional[Tensor] = None,
        conv_mode: Literal["2d", "3d", "3d_transposed"] = "3d",
        output_padding: int = 0,
    ) -> None:
        """
        Implements SE3 separable group convolution.

        :param in_channels: int denoting the number of input channels.
        :param out_channels: int denoting the number of output channels.
        :param kernel_size: tuple denoting the spatial kernel size.
        :param groups: int denoting the number of groups for depth-wise separability.
        :param stride: int denoting the stride.
        :param padding: int or denoting padding.
        :param dilation: int denoting dilation.
        :param group_kernel_size: int denoting the group kernel size (default 4).
        :param grid_H: tensor of shape (N, 3, 3) of SO3 elements (rotation matrices). If
                not provided, a uniform grid will be initalizd of size group_kernel_size.
                If provided, group_kernel_size will be set to N.
        :param padding_mode: str denoting the padding mode.
        :param permute_output_grid: bool that if true will randomly permute output group grid
                             for estimating continuous groups.
        :param group_sampling_mode: str denoting mode used for sampling group weights. Supports
                             rbf (default) or nearest.
        :param group_sampling_width: float denoting width of Gaussian rbf kernel when using rbf sampling.
                              If set to 0.0 (default, recommended), width will be initialized on
                              the density of grid_H.
        :param spatial_sampling_mode: str denoting mode used for sampling spatial weights. Supports
                               bilinear (default) or nearest.
        :param spatial_sampling_padding_mode: str denoting padding mode for spatial weight sampling,
                                       border (default) is recommended.
        :param bias: bool that if true, will initialzie bias parameters.
        :param mask: bool that if true, will initialize spherical mask applied to spatial weights.
        :param conv_mode: str denoting the convolution mode. Supports 2d, 3d, or 3d_transposed.
        :param output_padding: int denoting the output padding for 3d_transposed convolution.
        """
        kernel = GSeparableKernelSE3(
            in_channels,
            out_channels,
            kernel_size,
            group_kernel_size,
            groups=groups,
            group_sampling_mode=group_sampling_mode,
            group_sampling_width=group_sampling_width,
            spatial_sampling_mode=spatial_sampling_mode,
            spatial_sampling_padding_mode=spatial_sampling_padding_mode,
            mask=mask,
            grid_H=grid_H,
        )

        self.permute_output_grid = permute_output_grid

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
            bias,
            conv_mode=conv_mode,
            output_padding=output_padding,
        )

    def forward(
        self, input: Tensor, in_H: Tensor, out_H: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        if out_H is None:
            out_H = in_H

        if self.permute_output_grid:
            out_H = so3.left_apply_matrix(
                so3.random_matrix(1, device=input.device), out_H
            )

        return super().forward(input, in_H, out_H)


class RGSeparableConvSE3(GSeparableConv3d):
    """Implements SE3 separable group convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filter_banks: int,
        kernel_size: int,
        group_kernel_size: int = 4,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        permute_output_grid: bool = True,
        group_sampling_mode: str = "rbf",
        group_sampling_width: float = 0.0,
        spatial_sampling_mode: str = "bilinear",
        spatial_sampling_padding_mode: str = "border",
        mask: bool = True,
        bias: bool = False,
        grid_H: Optional[Tensor] = None,
        conv_mode: Literal["2d", "3d", "3d_transposed"] = "3d",
        output_padding: int = 0,
    ) -> None:
        """
        :param int_channels: int denoting the number of input channels.
        :param out_channels: int denoting the number of output channels.
        :param num_filter_banks: int denoting the number of filter banks.
        :param kernel_size: tuple denoting the spatial kernel size.
        :param groups: int denoting the number of groups for depth-wise separability.
        :param stride: int denoting the stride.
        :param padding: int or denoting padding.
        :param dilation: int denoting dilation.
        :param group_kernel_size: int denoting the group kernel size (default 4).
        :param grid_H: tensor of shape (N, 3, 3) of SO3 elements (rotation matrices). If
                not provided, a uniform grid will be initalizd of size group_kernel_size.
                If provided, group_kernel_size will be set to N.
        :param padding_mode: str denoting the padding mode.
        :param permute_output_grid: bool that if true will randomly permute output group grid
                             for estimating continuous groups.
        :param group_sampling_mode: str denoting mode used for sampling group weights. Supports
                             rbf (default) or nearest.
        :param group_sampling_width: float denoting width of Gaussian rbf kernel when using rbf sampling.
                              If set to 0.0 (default, recommended), width will be initialized on
                              the density of grid_H.
        :param spatial_sampling_mode: str denoting mode used for sampling spatial weights. Supports
                               bilinear (default) or nearest.
        :param spatial_sampling_padding_mode: str denoting padding mode for spatial weight sampling,
                                       border (default) is recommended.
        :param bias: bool that if true, will initialzie bias parameters.
        :param mask: bool that if true, will initialize spherical mask applied to spatial weights.
        :param conv_mode: str denoting the convolution mode. Supports 2d, 3d, or 3d_transposed.
        :param output_padding: int denoting the output for 3d_transposed convolution.
        """
        kernel = RGSeparableKernelSE3(
            in_channels,
            out_channels,
            num_filter_banks,
            kernel_size,
            group_kernel_size,
            groups=groups,
            group_sampling_mode=group_sampling_mode,
            group_sampling_width=group_sampling_width,
            spatial_sampling_mode=spatial_sampling_mode,
            spatial_sampling_padding_mode=spatial_sampling_padding_mode,
            mask=mask,
            grid_H=grid_H,
        )

        self.permute_output_grid = permute_output_grid

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
            bias,
            conv_mode=conv_mode,
            output_padding=output_padding,
        )

    def forward(
        self, input: Tensor, in_H: Tensor, out_H: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        if out_H is None:
            out_H = in_H

        if self.permute_output_grid:
            out_H = so3.left_apply_matrix(
                so3.random_matrix(1, device=input.device), out_H
            )

        return super().forward(input, in_H, out_H)


class GConvSE3(GConv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        group_kernel_size: int | tuple = 4,
        groups: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        padding_mode: str = "zeros",
        permute_output_grid: bool = True,
        group_sampling_mode: str = "rbf",
        group_sampling_width: float = 0.0,
        spatial_sampling_mode: str = "bilinear",
        spatial_sampling_padding_mode: str = "border",
        mask: bool = True,
        bias: bool = False,
        grid_H: Optional[Tensor] = None,
    ) -> None:
        """
        Implements SE3 separable group convolution.

        Arguments:
            - int_channels: int denoting the number of input channels.
            - out_channels: int denoting the number of output channels.
            - kernel_size: tuple denoting the spatial kernel size.
            - groups: int denoting the number of groups for depth-wise separability.
            - stride: int denoting the stride.
            - padding: int or denoting padding.
            - dilation: int denoting dilation.
            - group_kernel_size: int denoting the group kernel size (default 4).
            - grid_H: tensor of shape (N, 3, 3) of SO3 elements (rotation matrices). If
                      not provided, a uniform grid will be initalizd of size group_kernel_size.
                      If provided, group_kernel_size will be set to N.
            - padding_mode: str denoting the padding mode.
            - permute_output_grid: bool that if true will randomly permute output group grid
                                   for estimating continuous groups.
            - group_sampling_mode: str denoting mode used for sampling group weights. Supports
                                   rbf (default) or nearest.
            - group_sampling_width: float denoting width of Gaussian rbf kernel when using rbf sampling.
                                    If set to 0.0 (default, recommended), width will be initialized on
                                    the density of grid_H.
            - spatial_sampling_mode: str denoting mode used for sampling spatial weights. Supports
                                     bilinear (default) or nearest.
            - spatial_sampling_padding_mode: str denoting padding mode for spatial weight sampling,
                                             border (default) is recommended.
            - bias: bool that if true, will initialzie bias parameters.
            - mask: bool that if true, will initialize spherical mask applied to spatial weights.
        """
        kernel = GKernelSE3(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            group_kernel_size=group_kernel_size,
            groups=groups,
            group_sampling_mode=group_sampling_mode,
            group_sampling_width=group_sampling_width,
            spatial_sampling_mode=spatial_sampling_mode,
            spatial_sampling_padding_mode=spatial_sampling_padding_mode,
            mask=mask,
            grid_H=grid_H,
        )

        self.permute_output_grid = permute_output_grid

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
            bias,
        )

    def forward(
        self, input: Tensor, in_H: Tensor, out_H: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        if out_H is None:
            out_H = in_H

        if self.permute_output_grid:
            out_H = so3.left_apply_matrix(
                so3.random_matrix(1, device=input.device), out_H
            )

        return super().forward(input, in_H, out_H)
