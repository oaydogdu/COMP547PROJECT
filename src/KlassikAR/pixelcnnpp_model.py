from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from KlassikAR.pixelcnnpp_layers import (
    DownRightShiftedConv2d,
    DownRightShiftedDeconv2d,
    DownShiftedConv2d,
    DownShiftedDeconv2d,
    GatedResNet,
    NIN,
)


class PixelCNNLayerUp(nn.Module):
    def __init__(self, nr_resnet: int, nr_filters: int) -> None:
        super().__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([GatedResNet(nr_filters, DownShiftedConv2d, skip_connection=0) for _ in range(nr_resnet)])
        self.ul_stream = nn.ModuleList([GatedResNet(nr_filters, DownRightShiftedConv2d, skip_connection=1) for _ in range(nr_resnet)])

    def forward(self, u: torch.Tensor, ul: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        u_list, ul_list = [], []
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list.append(u)
            ul_list.append(ul)
        return u_list, ul_list


class PixelCNNLayerDown(nn.Module):
    def __init__(self, nr_resnet: int, nr_filters: int) -> None:
        super().__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([GatedResNet(nr_filters, DownShiftedConv2d, skip_connection=1) for _ in range(nr_resnet)])
        self.ul_stream = nn.ModuleList([GatedResNet(nr_filters, DownRightShiftedConv2d, skip_connection=2) for _ in range(nr_resnet)])

    def forward(
        self, u: torch.Tensor, ul: torch.Tensor, u_list: list[torch.Tensor], ul_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))
        return u, ul


class PixelCNNPP(nn.Module):
    def __init__(self, nr_resnet: int = 5, nr_filters: int = 160, nr_logistic_mix: int = 10, input_channels: int = 3) -> None:
        super().__init__()
        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayerDown(down_nr_resnet[i], nr_filters) for i in range(3)])
        self.up_layers = nn.ModuleList([PixelCNNLayerUp(nr_resnet, nr_filters) for _ in range(3)])

        self.downsize_u_stream = nn.ModuleList([DownShiftedConv2d(nr_filters, nr_filters, stride=(2, 2)) for _ in range(2)])
        self.downsize_ul_stream = nn.ModuleList(
            [DownRightShiftedConv2d(nr_filters, nr_filters, stride=(2, 2)) for _ in range(2)]
        )
        self.upsize_u_stream = nn.ModuleList([DownShiftedDeconv2d(nr_filters, nr_filters, stride=(2, 2)) for _ in range(2)])
        self.upsize_ul_stream = nn.ModuleList(
            [DownRightShiftedDeconv2d(nr_filters, nr_filters, stride=(2, 2)) for _ in range(2)]
        )

        self.u_init = DownShiftedConv2d(input_channels + 1, nr_filters, filter_size=(2, 3), shift_output_down=True)
        self.ul_init = nn.ModuleList(
            [
                DownShiftedConv2d(input_channels + 1, nr_filters, filter_size=(1, 3), shift_output_down=True),
                DownRightShiftedConv2d(input_channels + 1, nr_filters, filter_size=(2, 1), shift_output_right=True),
            ]
        )

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = NIN(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        if self.init_padding is None and not sample:
            xs = [int(y) for y in x.size()]
            self.init_padding = torch.ones(xs[0], 1, xs[2], xs[3], device=x.device)

        if sample:
            xs = [int(y) for y in x.size()]
            padding = torch.ones(xs[0], 1, xs[2], xs[3], device=x.device)
            x = torch.cat((x, padding), 1)
        else:
            x = torch.cat((x, self.init_padding), 1)

        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list += u_out
            ul_list += ul_out
            if i != 2:
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        u = u_list.pop()
        ul = ul_list.pop()
        for i in range(3):
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))
        return x_out
