import importlib
import torch


def backend():
    return importlib.import_module("tfs_train_v2_c0_ext")


class TFSConvCSRFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale):
        out, hs = backend().c2_forward(x, weight, bias, rowptr, colidx, scale)
        ctx.save_for_backward(hs, weight, rowptr, colidx, scale)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        hs, weight, rowptr, colidx, scale = ctx.saved_tensors
        compute_dx = bool(ctx.needs_input_grad[0])
        dx, dw, db, _ = backend().c2_backward_selective(
            grad_output.contiguous(), hs, weight, rowptr, colidx, scale,
            compute_dx)
        if not compute_dx:
            dx = None
        return dx, dw, db, None, None, None


class TFSConvCSRParallelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule,
                threads, private_dw, threads_per_numa):
        out, hs, _ = backend().c3_forward(
            x, weight, bias, rowptr, colidx, scale, schedule, int(threads))
        ctx.save_for_backward(hs, weight, rowptr, colidx, scale, schedule)
        ctx.threads = int(threads)
        ctx.private_dw = bool(private_dw)
        ctx.threads_per_numa = int(threads_per_numa)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        hs, weight, rowptr, colidx, scale, schedule = ctx.saved_tensors
        compute_dx = bool(ctx.needs_input_grad[0])
        dx, dw, db, _, _, _ = backend().c3_backward_selective(
            grad_output.contiguous(), hs, weight, rowptr, colidx, scale,
            schedule, ctx.threads, ctx.private_dw, ctx.threads_per_numa,
            compute_dx)
        if not compute_dx:
            dx = None
        return dx, dw, db, None, None, None, None, None, None, None
