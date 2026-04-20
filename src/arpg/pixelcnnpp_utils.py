from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def concat_elu(x: torch.Tensor) -> torch.Tensor:
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))


def log_sum_exp(x: torch.Tensor) -> torch.Tensor:
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x: torch.Tensor) -> torch.Tensor:
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss(x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix : 2 * nr_mix], min=-7.0)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix : 3 * nr_mix])

    x = x.contiguous().unsqueeze(-1) + torch.zeros(xs + [nr_mix], device=x.device)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
    m3 = (
        means[:, :, :, 2, :]
        + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :]
        + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]
    ).view(xs[0], xs[1], xs[2], 1, nr_mix)
    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = torch.sigmoid(min_in)

    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1.0 - inner_inner_cond) * (
        log_pdf_mid - math.log(127.5)
    )
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    return -torch.sum(log_sum_exp(log_probs))


def discretized_mix_logistic_loss_1d(x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix : 2 * nr_mix], min=-7.0)

    x = x.contiguous().unsqueeze(-1) + torch.zeros(xs + [nr_mix], device=x.device)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1.0 - inner_inner_cond) * (
        log_pdf_mid - math.log(127.5)
    )
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    return -torch.sum(log_sum_exp(log_probs))


def to_one_hot(tensor: torch.Tensor, n: int, fill_with: float = 1.0) -> torch.Tensor:
    one_hot = torch.zeros(tensor.size() + (n,), device=tensor.device, dtype=torch.float32)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def sample_from_discretized_mix_logistic_1d(l: torch.Tensor, nr_mix: int) -> torch.Tensor:
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1]

    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])

    temp = torch.rand_like(logit_probs).clamp_(1e-5, 1.0 - 1e-5)
    temp = logit_probs - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(l[:, :, :, :, nr_mix : 2 * nr_mix] * sel, dim=4), min=-7.0)
    u = torch.rand_like(means).clamp_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))
    x0 = torch.clamp(x[:, :, :, 0], min=-1.0, max=1.0)
    return x0.unsqueeze(1)


def sample_from_discretized_mix_logistic(l: torch.Tensor, nr_mix: int) -> torch.Tensor:
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])

    temp = torch.rand_like(logit_probs).clamp_(1e-5, 1.0 - 1e-5)
    temp = logit_probs - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])

    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(l[:, :, :, :, nr_mix : 2 * nr_mix] * sel, dim=4), min=-7.0)
    coeffs = torch.sum(torch.tanh(l[:, :, :, :, 2 * nr_mix : 3 * nr_mix]) * sel, dim=4)

    u = torch.rand_like(means).clamp_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))
    x0 = torch.clamp(x[:, :, :, 0], min=-1.0, max=1.0)
    x1 = torch.clamp(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.0, max=1.0)
    x2 = torch.clamp(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.0, max=1.0)
    out = torch.cat([x0.unsqueeze(-1), x1.unsqueeze(-1), x2.unsqueeze(-1)], dim=3)
    return out.permute(0, 3, 1, 2)


def down_shift(x: torch.Tensor, pad: nn.Module | None = None) -> torch.Tensor:
    xs = [int(y) for y in x.size()]
    x = x[:, :, : xs[2] - 1, :]
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


def right_shift(x: torch.Tensor, pad: nn.Module | None = None) -> torch.Tensor:
    xs = [int(y) for y in x.size()]
    x = x[:, :, :, : xs[3] - 1]
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)
