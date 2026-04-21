from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as wn

from KlassikAR.pixelcnnpp_utils import concat_elu, down_shift, right_shift


class NIN(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.lin = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin(x.contiguous().view(shp[0] * shp[1] * shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


class DownShiftedConv2d(nn.Module):
    def __init__(
        self,
        num_filters_in: int,
        num_filters_out: int,
        filter_size: tuple[int, int] = (2, 3),
        stride: tuple[int, int] = (1, 1),
        shift_output_down: bool = False,
    ) -> None:
        super().__init__()
        self.conv = wn(nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride))
        self.shift_output_down = shift_output_down
        self.pad = nn.ZeroPad2d(
            (
                int((filter_size[1] - 1) / 2),
                int((filter_size[1] - 1) / 2),
                filter_size[0] - 1,
                0,
            )
        )
        if shift_output_down:
            self.down_shift = lambda x: down_shift(x, pad=nn.ZeroPad2d((0, 0, 1, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        return self.down_shift(x) if self.shift_output_down else x


class DownShiftedDeconv2d(nn.Module):
    def __init__(
        self,
        num_filters_in: int,
        num_filters_out: int,
        filter_size: tuple[int, int] = (2, 3),
        stride: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride, output_padding=1))
        self.filter_size = filter_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[:, :, : (xs[2] - self.filter_size[0] + 1), int((self.filter_size[1] - 1) / 2) : (xs[3] - int((self.filter_size[1] - 1) / 2))]


class DownRightShiftedConv2d(nn.Module):
    def __init__(
        self,
        num_filters_in: int,
        num_filters_out: int,
        filter_size: tuple[int, int] = (2, 2),
        stride: tuple[int, int] = (1, 1),
        shift_output_right: bool = False,
    ) -> None:
        super().__init__()
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        self.conv = wn(nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride))
        self.shift_output_right = shift_output_right
        if shift_output_right:
            self.right_shift = lambda x: right_shift(x, pad=nn.ZeroPad2d((1, 0, 0, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        return self.right_shift(x) if self.shift_output_right else x


class DownRightShiftedDeconv2d(nn.Module):
    def __init__(
        self,
        num_filters_in: int,
        num_filters_out: int,
        filter_size: tuple[int, int] = (2, 2),
        stride: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride, output_padding=1))
        self.filter_size = filter_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[:, :, : (xs[2] - self.filter_size[0] + 1), : (xs[3] - self.filter_size[1] + 1)]


class GatedResNet(nn.Module):
    def __init__(self, num_filters: int, conv_op: type[nn.Module], skip_connection: int = 0) -> None:
        super().__init__()
        self.skip_connection = skip_connection
        self.conv_input = conv_op(2 * num_filters, num_filters)
        if skip_connection != 0:
            self.nin_skip = NIN(2 * skip_connection * num_filters, num_filters)
        self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)

    def forward(self, og_x: torch.Tensor, a: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv_input(concat_elu(og_x))
        if a is not None:
            x = x + self.nin_skip(concat_elu(a))
        x = concat_elu(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        a_chunk, b_chunk = torch.chunk(x, 2, dim=1)
        c3 = a_chunk * torch.sigmoid(b_chunk)
        return og_x + c3
