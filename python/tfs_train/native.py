import importlib
import torch


def backend():
    return importlib.import_module("tfs_train_v2_c0_ext")


# ``tfs_train_v2_c0_ext`` registers only the AMX C3 entry points (see
# csrc/experiments/products_saved_t_20260812/bindings_aggregate.cpp).  The
# historical non-AMX C3 fallbacks -- ``c3_forward``, ``c3_forward_transform``
# and ``c3_backward_selective`` -- have no definition anywhere in ``csrc/`` or
# ``include/``, so every one of those call sites used to die with a bare
# ``AttributeError`` raised from inside autograd.  Fail at the decision point
# with the actual contract instead.  Reachability: every shipped launcher and
# the release profile set HYBRID_AMX_FORWARD=1 / HYBRID_AMX_BACKWARD=1 on the
# TFS arm (``=0`` appears only on the dgl_stock arm, which never enters this
# module), so this is a latent path, not an active regression.
MISSING_NON_AMX_C3_ENTRY_POINTS = (
    "c3_forward", "c3_forward_transform", "c3_backward_selective")


def require_non_amx_c3(entry_point):
    """Always raises: the named non-AMX C3 kernel is absent from this release."""
    raise RuntimeError(
        f"{entry_point} is not implemented in this release: "
        "tfs_train_v2_c0_ext ships only the AMX C3 kernels.  Set "
        "HYBRID_AMX_FORWARD=1 and HYBRID_AMX_BACKWARD=1 (scripts/"
        "tfs_standard_env.sh and every authority launcher already do), or use "
        'the C2 runtime via TFSConvCSR(runtime="reference").')


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
    """Unavailable in this release.

    Both halves of this primitive called ``c3_forward`` / ``c3_backward_
    selective``, which the extension never exported, so it could only ever
    raise ``AttributeError``.  It is kept so ``tfs_train.__init__`` and
    ``TFSConvCSR(runtime="parallel")`` keep their import surface, but it now
    states the contract instead.  The AMX C3 path lives in
    ``tfs_train.authority_autograd``; the CPU reference path is
    ``TFSConvCSRFunction`` above.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule,
                threads, private_dw, threads_per_numa):
        require_non_amx_c3("c3_forward")

    @staticmethod
    def backward(ctx, grad_output):
        require_non_amx_c3("c3_backward_selective")
