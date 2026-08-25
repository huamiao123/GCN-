import torch
from .reference import torch_v2


class TFSConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, a_off):
        p, s, hs = torch_v2(x, weight, bias, a_off)
        ctx.save_for_backward(weight, a_off, s, hs)
        return p

    @staticmethod
    def backward(ctx, grad_output):
        weight, a_off, s, hs = ctx.saved_tensors
        eye = torch.eye(a_off.shape[0], dtype=a_off.dtype, device=a_off.device)
        grad_scaled = s[:, None] * grad_output
        ybar = (a_off.transpose(0, 1) + eye) @ grad_scaled
        grad_weight = hs.transpose(0, 1) @ ybar
        grad_input = s[:, None] * (ybar @ weight.transpose(0, 1))
        grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias, None
