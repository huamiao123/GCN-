#!/usr/bin/env python3
import csv
import contextlib
import gc
import json
import os
import time
from pathlib import Path

# Earliest in-script monotonic marker; external launcher wall additionally
# includes interpreter startup.
process_start_ns = time.perf_counter_ns()

import numpy as np

# Freeze the Slurm/numactl CPU set before importing torch: importing the CPU
# runtime can initialize OpenMP and narrow the calling thread's affinity.
if "TFS_WORKER_CPUS" not in os.environ:
    cpus = sorted(os.sched_getaffinity(0))
    ranges = []
    start = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu != previous + 1:
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = cpu
        previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    os.environ["TFS_WORKER_CPUS"] = ",".join(ranges)

import torch
import torch.nn.functional as F
try:
    from torch_geometric.nn import GCNConv
except ImportError:  # The AMX/TFS path does not require PyG.
    GCNConv = None

from tfs_train.datasets import NodePropertyDataset
from tfs_train.graph import CSRGraph, preprocess_undirected_fast
from tfs_train.modules import TFSConvCSR
from tfs_train.native import backend
from tfs_train.persistent_hs_cache import (
    PersistentAggregateCache,
    PersistentHsCache,
    get_or_build_if_within_budget,
)
from tfs_train.execution_plan import emit_plan_logs, plan_layers
from tfs_train.aggregate_saved import AggregateSavedFunction
from tfs_train.dimension_dispatch import (
    AggregateWideAMX as SharedAggregateWideAMX,
    WideOutputAMX as SharedWideOutputAMX,
)
from tfs_train.standard_runtime import configure_dgl, validate_authority_variant
from tfs_train.standard_dgl import (DGLGCN,
                                    build_stock_dgl_graph_from_pull_csr)
from tfs_train.authority_autograd import (
    AggregateCachedT0 as _AuthorityAggregateCachedT0,
    AggregateFirst as _AuthorityAggregateFirst,
    TransformFirst as _AuthorityTransformFirst,
)


def load_igb_hom_small(root):
    """Load the locally staged IGB-HOM-small 19-class representation.

    The official archive contains a 1M x 1024 FP32 feature matrix, a 19-class
    node-label vector, and a directed edge list.  TFS's active contract is an
    undirected CSR with an implicit self loop, so both TFS and DGL receive the
    same deterministic `preprocess_undirected_fast` graph below.  IGB-HOM-small
    does not ship a training split in this staging layout; the fixed 60/20/20
    split is generated once per process from the recorded seed, outside timing.
    """
    root = Path(root) / "small" / "processed"
    paper = root / "paper"
    x_np = np.load(paper / "node_feat.npy", mmap_mode="r")
    # This runner's default contract is IGB-HOM-small with 19 classes.  The
    # 2K labels are a different task and require an explicit output dimension.
    label_file = os.environ.get("IGB_LABEL_FILE", "node_label_19.npy")
    labels_np = np.load(paper / label_file, mmap_mode="r")
    edge_np = np.load(root / "paper__cites__paper" / "edge_index.npy",
                      mmap_mode="r")
    if x_np.ndim != 2 or x_np.shape[1] != 1024 or x_np.dtype != np.float32:
        raise ValueError(f"unexpected IGB feature matrix: {x_np.shape} {x_np.dtype}")
    if labels_np.ndim != 1 or labels_np.shape[0] != x_np.shape[0]:
        raise ValueError(f"unexpected IGB labels: {labels_np.shape}")
    if edge_np.ndim != 2 or edge_np.shape[1] != 2:
        raise ValueError(f"unexpected IGB edges: {edge_np.shape}")
    x = torch.from_numpy(x_np)
    labels = torch.from_numpy(labels_np).to(torch.long)
    edge_index = torch.from_numpy(edge_np.T.copy()).to(torch.long)
    graph = preprocess_undirected_fast(edge_index, int(x.shape[0]),
                                       torch.float32)
    split_seed = int(os.environ.get("IGB_SPLIT_SEED", "20260813"))
    generator = torch.Generator(device="cpu").manual_seed(split_seed)
    permutation = torch.randperm(x.shape[0], generator=generator)
    n_train = int(x.shape[0] * 0.6)
    n_valid = int(x.shape[0] * 0.2)
    train_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    valid_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    train_mask[permutation[:n_train]] = True
    valid_mask[permutation[n_train:n_train + n_valid]] = True
    test_mask[permutation[n_train + n_valid:]] = True
    return NodePropertyDataset(x, labels, graph, train_mask, valid_mask,
                               test_mask)


def make_tail_correctness_fixture():
    # Retained as a small debugging fixture; the formal IGB run sets
    # HYBRID_CHECK_FULL=1 and gates the complete graph before timing.
    n = 4096
    src = torch.arange(n, dtype=torch.long)
    edge_index = torch.stack((src, (src * 17 + 23) % n))
    graph = preprocess_undirected_fast(edge_index, n, torch.float32)
    input_dim = int(os.environ.get("HYBRID_INPUT_DIM", "1024"))
    out_dim = int(os.environ.get("HYBRID_OUT_DIM", "19"))
    x = torch.randn(n, input_dim, dtype=torch.float32)
    labels = torch.remainder(torch.arange(n), out_dim).long()
    train = torch.ones(n, dtype=torch.bool)
    empty = torch.zeros(n, dtype=torch.bool)
    return NodePropertyDataset(x, labels, graph, train, empty, empty)


class _C3Base(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        hs, weight, rowptr, colidx, scale, schedule = ctx.saved_tensors
        compute_dx = bool(ctx.needs_input_grad[0])
        if getattr(ctx, "wide_amx", False):
            # The AMX primitive has a D<=128 contract.  Keep the complete
            # 2983-class layer on AMX by applying that primitive to output
            # tiles and reducing dX across tiles.  No generic/C2 fallback is
            # used on this path.
            dx = None
            dw_parts = []
            db_parts = []
            for start in range(0, weight.shape[1], ctx.amx_tile):
                end = min(start + ctx.amx_tile, weight.shape[1])
                dx_part, dw_part, db_part, _ = backend().c3_backward_amx_v2(
                    grad_output[:, start:end].contiguous(), hs,
                    weight[:, start:end].contiguous(), rowptr, colidx, scale,
                    ctx.threads, compute_dx)
                if compute_dx:
                    if dx is None:
                        dx = dx_part
                    else:
                        dx.add_(dx_part)
                dw_parts.append(dw_part)
                db_parts.append(db_part)
            dw = torch.cat(dw_parts, dim=1)
            db = torch.cat(db_parts, dim=0)
        elif ctx.amx:
            dx, dw, db, _ = backend().c3_backward_amx_v2(
                grad_output.contiguous(), hs, weight, rowptr, colidx, scale,
                ctx.threads, compute_dx)
        elif ctx.c2:
            raise RuntimeError("IGB AMX run attempted a C2 backward fallback")
        else:
            raise RuntimeError("IGB AMX run reached a non-AMX backward path")
        if not compute_dx:
            dx = None
        # forward() has ten inputs; cached_hs and backward_hs are
        # non-differentiable cache storage tensors.
        return dx, dw, db, None, None, None, None, None, None, None


def cached_c3_forward(x, cached_hs, weight, bias, rowptr, colidx, scale,
                      threads, transform_first):
    """Dispatch the canonical V1 or optional padded-Hs V2 consumer."""
    if cached_hs.shape[1] != x.shape[1]:
        return backend().c3_forward_cached_hs_padded_amx_v2(
            x, cached_hs, weight, bias, rowptr, colidx, scale,
            int(threads), transform_first)
    return backend().c3_forward_cached_hs_amx_v1(
        x, cached_hs, weight, bias, rowptr, colidx, scale,
        int(threads), transform_first)


class LegacyWideOutputAMX(torch.autograd.Function):
    """Historical-only wide-D implementation kept for provenance.

    The extension keeps the established transform-first BF16 rounding order,
    but performs Hs construction, weight conversion, schedule setup, output
    tiling, and backward concatenation inside native code.  No C2 or generic
    fallback is reachable from this path.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule, threads):
        if os.environ.get("HYBRID_AMX_FORWARD") != "1":
            raise RuntimeError("wide output path requires AMX forward")
        out, hs = backend().c3_forward_wide_amx_v3(
            x, weight, bias, rowptr, colidx, scale, int(threads))
        ctx.save_for_backward(hs, weight, rowptr, colidx, scale)
        ctx.threads = int(threads)
        ctx.amx = os.environ.get("HYBRID_AMX_BACKWARD") == "1"
        return out

    @staticmethod
    def backward(ctx, grad_output):
        hs, weight, rowptr, colidx, scale = ctx.saved_tensors
        compute_dx = bool(ctx.needs_input_grad[0])
        if not ctx.amx:
            raise RuntimeError("wide output path requires AMX backward")
        dx, dw, db, _ = backend().c3_backward_wide_amx_v3(
            grad_output.contiguous(), hs, weight, rowptr, colidx, scale,
            ctx.threads, compute_dx)
        if not compute_dx:
            dx = None
        return dx, dw, db, None, None, None, None, None


class LegacyAggregateWideAMX(torch.autograd.Function):
    """Historical-only aggregate-wide implementation kept for provenance.

    The sparse operator is symmetric after TFS preprocessing, so the
    aggregate-first backward can reuse the validated AMX pull primitive with
    an identity weight for A^T(dP).  The wide dW/dP products stay in native
    PyTorch GEMM and avoid 24 output-tile sparse traversals.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule, threads):
        if os.environ.get("HYBRID_AMX_FORWARD") != "1":
            raise RuntimeError("wide aggregate path requires AMX forward")
        out, hs, pulled = backend().c3_forward_aggregate_wide_amx_v3(
            x, weight, bias, rowptr, colidx, scale, int(threads))
        ctx.save_for_backward(pulled, weight, rowptr, colidx, scale)
        ctx.threads = int(threads)
        ctx.amx = os.environ.get("HYBRID_AMX_BACKWARD") == "1"
        return out

    @staticmethod
    def backward(ctx, grad_output):
        pulled, weight, rowptr, colidx, scale = ctx.saved_tensors
        if not ctx.amx:
            raise RuntimeError("wide aggregate path requires AMX backward")
        grad = grad_output.contiguous()
        scale_f = scale.contiguous()
        # Keep the wide backward GEMMs on the same BF16/AMX path as forward.
        # The returned parameter gradient remains FP32 for Adam; PyTorch's
        # BF16 matmul accumulates in FP32 internally before the BF16 store.
        db_native = os.environ.get("TFS_SCALE_GRAD_DB_NATIVE") == "1"
        if db_native:
            # Fuse row scaling, FP32->BF16 conversion, and db while each
            # gradient row is resident; the native routine returns a
            # deterministic per-worker reduction for db.
            grad_scaled_bf16, db = backend().c3_scale_grad_bf16_db_v2(
                grad, scale_f, ctx.threads)
        elif os.environ.get("TFS_SCALE_GRAD_BF16_NATIVE") == "1":
            # Fuse row scaling and BF16 conversion to avoid a full FP32
            # temporary for the wide output gradient.
            grad_scaled_bf16 = backend().c3_scale_grad_bf16_v1(
                grad, scale_f, ctx.threads)
            db = grad.sum(0)
        else:
            grad_scaled_bf16 = (grad * scale_f.unsqueeze(1)).to(torch.bfloat16)
            db = grad.sum(0)
        weight_bf16 = weight.to(torch.bfloat16)
        # pulled is the pre-output-scale BF16 aggregate saved by the forward.
        dw = torch.matmul(pulled.transpose(0, 1), grad_scaled_bf16).float()
        compute_dx = bool(ctx.needs_input_grad[0])
        if compute_dx:
            # dP = d(A(H)W); A is symmetric for the undirected normalized CSR.
            d_pulled = torch.matmul(
                grad_scaled_bf16, weight_bf16.transpose(0, 1)).float()
            # The identity-weight workaround is only legal for K<=128.  The
            # native pull-only loop is stride-parametric, so select it
            # automatically for high-K aggregate layers instead of
            # constructing an oversized identity matrix.
            if (os.environ.get("TFS_PULL_ONLY_BACKWARD") == "1" or
                    int(weight.shape[0]) > 128):
                # Preserve the old BF16 rounding/CSR order while avoiding the
                # identity weight, identity GEMM and general forward epilogue.
                d_h = backend().c3_pull_only_amx_v1(
                    d_pulled, rowptr, colidx, ctx.threads)
            else:
                k = int(weight.shape[0])
                eye = torch.eye(k, dtype=weight.dtype, device=weight.device)
                zero = torch.zeros(k, dtype=weight.dtype, device=weight.device)
                ones = torch.ones_like(scale_f)
                d_h, _ = backend().c3_forward_amx_v2(
                    d_pulled, eye, zero, rowptr, colidx, ones,
                    ctx.threads, False)
            dx = d_h * scale_f.unsqueeze(1)
        else:
            dx = None
        return dx, dw, db, None, None, None, None, None


def _amx_shape_supported(x, weight):
    """Shape gate for the standard C3/wide-K AMX input contract.

    The native C3 implementation is output-tiled at D<=128 and is
    stride-parametric in K.  The old gate treated K>128 as an experimental
    opt-in, which made IGB's 1024->128 first layer fall through even though
    the native wide-K loops and padded-Hs producer were already present.
    Dimension selection now belongs to the execution planner; this helper
    only checks the native contract and therefore accepts arbitrary logical K.
    """
    return (x.dim() == 2 and weight.dim() == 2 and x.shape[1] >= 1 and
            weight.shape[0] == x.shape[1] and weight.shape[1] <= 128)


class AggregateFirst(_C3Base):
    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule, threads,
                cached_hs=None, backward_hs=None):
        use_amx = (os.environ.get("HYBRID_AMX_FORWARD") == "1" and
                   _amx_shape_supported(x, weight))
        if use_amx:
            if cached_hs is None:
                out, hs = backend().c3_forward_amx_v2(
                    x, weight, bias, rowptr, colidx, scale, int(threads), False)
            else:
                out, hs = cached_c3_forward(
                    x, cached_hs, weight, bias, rowptr, colidx, scale,
                    threads, False)
            use_c2 = False
        elif x.shape[1] > 128:
            raise RuntimeError("wide-K IGB run requires AMX aggregate-first path")
        else:
            raise RuntimeError("IGB AMX run reached non-AMX aggregate path")
        hs_saved = hs if backward_hs is None else backward_hs
        ctx.save_for_backward(hs_saved, weight, rowptr, colidx, scale, schedule)
        ctx.threads = int(threads)
        ctx.c2 = use_c2
        ctx.wide_amx = False
        ctx.amx_tile = 128
        ctx.amx = (os.environ.get("HYBRID_AMX_BACKWARD") == "1" and
                   _amx_shape_supported(hs, weight))
        return out


class TransformFirst(_C3Base):
    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule, threads,
                cached_hs=None, backward_hs=None):
        use_amx = (os.environ.get("HYBRID_AMX_FORWARD") == "1" and
                   _amx_shape_supported(x, weight))
        if use_amx:
            if cached_hs is None:
                out, hs = backend().c3_forward_amx_v2(
                    x, weight, bias, rowptr, colidx, scale, int(threads), True)
            else:
                out, hs = cached_c3_forward(
                    x, cached_hs, weight, bias, rowptr, colidx, scale,
                    threads, True)
            use_c2 = False
            use_wide_amx = False
        elif weight.shape[1] > 128:
            raise RuntimeError(
                "wide output must use the fused WideOutputAMX path")
        else:
            raise RuntimeError("IGB AMX run reached non-AMX transform path")
        hs_saved = hs if backward_hs is None else backward_hs
        ctx.save_for_backward(hs_saved, weight, rowptr, colidx, scale, schedule)
        ctx.threads = int(threads)
        ctx.c2 = False if use_wide_amx else use_c2
        ctx.wide_amx = use_wide_amx
        ctx.amx_tile = 128
        ctx.amx = (os.environ.get("HYBRID_AMX_BACKWARD") == "1" and
                   (use_wide_amx or _amx_shape_supported(hs, weight)))
        return out


class AggregateCachedT0(torch.autograd.Function):
    """Opt-in V3 aggregate cache for eligible non-wide layers."""

    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule,
                threads, cached_t0):
        if os.environ.get("HYBRID_AMX_FORWARD") != "1":
            raise RuntimeError("static aggregate cache requires AMX forward")
        out, t0 = backend().c3_forward_cached_aggregate_amx_v3(
            x, cached_t0, weight, bias, rowptr, colidx, scale,
            int(threads))
        ctx.save_for_backward(t0, weight, rowptr, colidx, scale, schedule)
        ctx.threads = int(threads)
        ctx.amx = os.environ.get("HYBRID_AMX_BACKWARD") == "1"
        return out

    @staticmethod
    def backward(ctx, grad_output):
        t0, weight, rowptr, colidx, scale, schedule = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            raise RuntimeError(
                "static aggregate cache cannot provide dX; disable it when x requires grad")
        if not ctx.amx:
            raise RuntimeError("static aggregate cache requires AMX backward")
        dw, db, _ = backend().c3_backward_cached_aggregate_amx_v3(
            grad_output.contiguous(), t0, rowptr, colidx, scale, ctx.threads)
        return None, dw, db, None, None, None, None, None, None


# The generic C3 primitives are shared by Products, Arxiv, and IGB.  IGB's
# high-D variants above remain dataset-specific and are dispatched before
# these aliases are reached.
AggregateFirst = _AuthorityAggregateFirst
TransformFirst = _AuthorityTransformFirst
AggregateCachedT0 = _AuthorityAggregateCachedT0


class LegacyHybridConv(torch.nn.Module):
    def __init__(self, k, d, order, threads, cache_static_hs=False, plan=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(k, d))
        self.bias = torch.nn.Parameter(torch.zeros(d))
        self.plan = plan
        self.order = plan.order if plan is not None else order
        self.threads = int(threads)
        if plan is not None:
            cache_static_hs = bool(plan.static_hs)
            if plan.execution_variant == "aggregate_saved_v4":
                cache_static_hs = False
        self.cache_static_hs = bool(cache_static_hs)
        pad_to = int(os.environ.get("HYBRID_HS_PAD_TO", "0")) or None
        use_t0 = (self.cache_static_hs and self.order == "aggregate" and
                  (plan.static_pulled if plan is not None else
                   os.environ.get("HYBRID_STATIC_AGG_CACHE") == "1"))
        self._aggregate_cache = (PersistentAggregateCache()
                                 if use_t0 else None)
        self._hs_cache = (PersistentHsCache(pad_to=pad_to)
                          if self.cache_static_hs and not use_t0 else None)
        # Keep dimension dispatch in the same planner used for the log line.
        # In particular, a 1024 -> 128 first layer is ``wide_k`` and must not
        # require the old HYBRID_WIDE_K_AMX opt-in; D=2983 remains a separate
        # wide-output/aggregate family.
        self.dimension_path = (plan.dimension_path if plan is not None else
                               ("wide_aggregate" if self.order == "aggregate" and d > 128
                                else "wide_output" if self.order == "transform" and d > 128
                                else "wide_k" if k > 128 else "c3"))
        self.wide_output_amx = self.dimension_path == "wide_output"
        self.wide_aggregate_amx = self.dimension_path == "wide_aggregate"
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, graph):
        if (torch.is_grad_enabled() and
                bool(x.requires_grad) != bool(self.plan.compute_dx)):
            raise RuntimeError(
                "TFS plan/autograd mismatch: "
                f"plan_id={self.plan.plan_id} planned_compute_dx="
                f"{int(self.plan.compute_dx)} actual_compute_dx="
                f"{int(bool(x.requires_grad))}")
        scale = graph.scale.to(x.dtype)
        variant = self.plan.execution_variant
        if variant.startswith("aggregate_highd_"):
            return SharedAggregateWideAMX.apply(
                x, self.weight, self.bias, graph.rowptr, graph.colidx,
                scale, graph.schedule, self.threads, self.plan)
        if variant.startswith("transform_highd_"):
            return SharedWideOutputAMX.apply(
                x, self.weight, self.bias, graph.rowptr, graph.colidx,
                scale, graph.schedule, self.threads, self.plan)
        if (variant == "aggregate_saved_v4" and
                os.environ.get("HYBRID_AMX_FORWARD") == "1" and
                os.environ.get("HYBRID_AMX_BACKWARD") == "1"):
            return AggregateSavedFunction.apply(
                x, self.weight, self.bias, graph.rowptr, graph.colidx,
                scale, graph.schedule, self.threads, self.plan)
        if (variant == "aggregate_static_v3" and
                self._aggregate_cache is not None and
                os.environ.get("HYBRID_PERSISTENT_HS_CACHE") == "1" and
                os.environ.get("HYBRID_AMX_FORWARD") == "1" and
                x.dtype == torch.float32 and scale.dtype == torch.float32 and
                not x.requires_grad):
            aggregate_cache = get_or_build_if_within_budget(
                self._aggregate_cache, x, graph, self.threads, backend())
            if aggregate_cache is not None:
                cached_t0, _ = aggregate_cache
                self.plan.assert_dispatch("aggregate_static_v3")
                return AggregateCachedT0.apply(
                    x, self.weight, self.bias, graph.rowptr, graph.colidx,
                    scale, graph.schedule, self.threads, cached_t0)
        if variant not in {"native_c3", "native_wide_k"}:
            raise RuntimeError(
                f"unhandled authority execution variant: {variant}")
        self.plan.assert_dispatch(
            "native_wide_k" if self.plan.k > 128 else "native_c3")
        fn = AggregateFirst if self.order == "aggregate" else TransformFirst
        cached_hs = None
        backward_hs = None
        if (self._hs_cache is not None and
                os.environ.get("HYBRID_PERSISTENT_HS_CACHE") == "1" and
                os.environ.get("HYBRID_AMX_FORWARD") == "1" and
                x.dtype == torch.float32 and scale.dtype == torch.float32):
            hs_cache = get_or_build_if_within_budget(
                self._hs_cache, x, graph, self.threads, backend())
            if hs_cache is not None:
                cached_storage, _ = hs_cache
                # The compact view preserves the fast sparse-pull stride.  The
                # aligned physical view is supplied separately to backward so
                # its dW packer can consume it without a copy.
                cached_hs = self._hs_cache.forward_tensor
                if cached_hs is None:
                    cached_hs = cached_storage
                if cached_hs.shape[1] != x.shape[1]:
                    raise RuntimeError("persistent Hs forward view must be logical-width")
                if cached_storage.shape[1] != cached_hs.shape[1]:
                    backward_hs = cached_storage
        return fn.apply(x, self.weight, self.bias, graph.rowptr, graph.colidx,
                        scale, graph.schedule, self.threads, cached_hs,
                        backward_hs)


class LegacyHybridGCN(torch.nn.Module):
    def __init__(self, threads, layers=2, dropout=0.5, in_dim=1024,
                 hidden_dim=128, out_dim=19, num_nodes=None):
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be at least 2")
        dims = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
        self._dims = dims
        if num_nodes is None:
            raise ValueError("HybridGCN requires num_nodes for the authority template")
        self._plan_node_count = int(num_nodes)
        self._plans = plan_layers(self._plan_node_count, dims,
                                  feature_static_first=True,
                                  threads=int(threads))
        self._plans_logged = False
        self.convs = torch.nn.ModuleList([
            LegacyHybridConv(dims[i], dims[i + 1],
                       self._plans[i].order, threads,
                       cache_static_hs=(i == 0), plan=self._plans[i])
            for i in range(layers)
        ])
        self.dropout = dropout

    def forward(self, x, graph):
        if not self._plans_logged:
            if int(x.shape[0]) != self._plan_node_count:
                raise RuntimeError(
                    "authority graph node count changed after planning: "
                    f"planned={self._plan_node_count} actual={int(x.shape[0])}")
            emit_plan_logs(self._plans)
            self._plans_logged = True
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, graph))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, graph)


from tfs_train.authority_model import HybridConv as _CanonicalHybridConv
from tfs_train.authority_model import HybridGCN as _CanonicalHybridGCN

# Dataset-local copies above are retained only as the one-cycle comparison
# fixture.  All authority runs bind to the canonical implementation.
HybridConv = _CanonicalHybridConv
HybridGCN = _CanonicalHybridGCN


class ReferenceGCN(torch.nn.Module):
    def __init__(self, layers=2, dropout=0.5, in_dim=1024,
                 hidden_dim=128, out_dim=19):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
        self.convs = torch.nn.ModuleList([
            TFSConvCSR(dims[i], dims[i + 1], runtime="reference")
            for i in range(layers)
        ])
        self.dropout = dropout

    def forward(self, x, graph):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, graph))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, graph)


class PyGGCN(torch.nn.Module):
    def __init__(self, layers=2, in_dim=1024, hidden_dim=128, out_dim=19):
        super().__init__()
        if GCNConv is None:
            raise RuntimeError("PyG path requires torch_geometric")
        dims = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
        self.convs = torch.nn.ModuleList([
            GCNConv(dims[i], dims[i + 1], add_self_loops=True,
                    normalize=True, cached=True)
            for i in range(layers)
        ])

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, edge_index)


class DGLCachedConv(torch.nn.Module):
    """DGL GSpMM with the static symmetric normalization already cached.

    This is the framework-native strong path used by the paired timing
    harness.  The graph has an explicit self-loop and `scale` is the
    canonical node degree^-1/2 vector from the TFS CSR preprocessing.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        from dgl.nn.pytorch import GraphConv
        self.base = GraphConv(in_dim, out_dim, norm="none", weight=True,
                              bias=False, allow_zero_in_degree=True)
        self.bias = torch.nn.Parameter(torch.zeros(out_dim))

    def forward(self, graph, x, scale):
        out = self.base(graph, x * scale.unsqueeze(1))
        return out * scale.unsqueeze(1) + self.bias


class DGLCachedGCN(torch.nn.Module):
    def __init__(self, layers=2, in_dim=1024, hidden_dim=128, out_dim=19):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
        self.convs = torch.nn.ModuleList([
            DGLCachedConv(dims[i], dims[i + 1]) for i in range(layers)
        ])

    def forward(self, x, arg):
        graph, scale = arg
        for conv in self.convs[:-1]:
            x = F.relu(conv(graph, x, scale))
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](graph, x, scale)


def metrics(a, b):
    d = (a.double() - b.double()).reshape(-1)
    denom = torch.linalg.vector_norm(a.double().reshape(-1)).clamp_min(1e-30)
    av = a.double().reshape(-1)
    bv = b.double().reshape(-1)
    cosine = torch.dot(av, bv) / (torch.linalg.vector_norm(av) *
                                  torch.linalg.vector_norm(bv)).clamp_min(1e-30)
    return {"relative_l2": float(torch.linalg.vector_norm(d) / denom),
            "max_abs": float(d.abs().max()), "cosine": float(cosine)}


threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
OUT_DIM = int(os.environ.get("HYBRID_OUT_DIM", "19"))
HIDDEN_DIM = int(os.environ.get("HYBRID_HIDDEN_DIM", "128"))
LAYERS = int(os.environ.get("HYBRID_LAYERS", "2"))
if LAYERS < 2:
    raise ValueError("HYBRID_LAYERS must be at least 2")
torch.set_num_interop_threads(1)
torch.set_num_threads(threads)
if torch.get_num_interop_threads() != 1:
    raise RuntimeError("PyTorch inter-op thread count must be exactly 1")
torch.manual_seed(int(os.environ.get("HYBRID_SEED", "101")))
ds = (make_tail_correctness_fixture()
      if os.environ.get("HYBRID_CHECK") == "1" and
      os.environ.get("HYBRID_CHECK_FULL") != "1"
      else load_igb_hom_small(os.environ["IGB_ROOT"]))
# The NumPy mmap is already row-major.  Keep one read-only FP32 tensor shared
# by the timed path; all correctness gates clone it explicitly when gradients
# are required.
x, labels, graph = ds.x.contiguous(), ds.labels, ds.graph
data_ready_ns = time.perf_counter_ns()
# TFS already owns the canonical CSR at this point.  DGL updates this marker
# below only after its direct-CSC DGLGraph construction is complete.
framework_graph_ready_ns = data_ready_ns

if os.environ.get("HYBRID_CHECK") == "1":
    if os.environ.get("HYBRID_CHECK_DGL") == "1":
        # Independent framework gate for each valid strong-DGL variant.
        # Both variants use the same explicit-self-loop graph as the timed
        # path; only the mathematically equivalent normalization placement
        # differs (stock DGL node normalization vs cached node scaling).
        import dgl
        configure_dgl(dgl, threads)
        dg, _dgl_graph_metadata = build_stock_dgl_graph_from_pull_csr(
            graph.rowptr, graph.colidx, x.shape[0])
        variant = os.environ.get("HYBRID_DGL_VARIANT", "cached")
        ref = ReferenceGCN(LAYERS, dropout=0.0, in_dim=x.shape[1],
                          hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM)
        dgl_model = (DGLCachedGCN(LAYERS, in_dim=x.shape[1],
                                   hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM)
                     if variant == "cached" else
                     DGLGCN(LAYERS, in_dim=x.shape[1],
                            hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
                            norm="both", allow_zero_in_degree=True))
        if variant not in ("cached", "stock"):
            raise ValueError(f"unknown HYBRID_DGL_VARIANT={variant}")
        if variant == "cached":
            dgl_arg = (dg, graph.scale)
            for ref_conv, dgl_conv in zip(ref.convs, dgl_model.convs):
                dgl_conv.base.weight.data.copy_(ref_conv.weight.data)
                dgl_conv.bias.data.copy_(ref_conv.bias.data)
        else:
            dgl_arg = (dg, None)
            for ref_conv, dgl_conv in zip(ref.convs, dgl_model.convs):
                dgl_conv.weight.data.copy_(ref_conv.weight.data)
                dgl_conv.bias.data.copy_(ref_conv.bias.data)
        # The numerical gate is dropout-free; do not leave the DGL model in
        # its default training mode while the reference uses dropout=0.0.
        ref.eval()
        dgl_model.eval()
        xr = x.clone().requires_grad_(True)
        xd = x.clone().requires_grad_(True)
        yr = ref(xr, graph)
        yd = dgl_model(xd, dgl_arg)
        lr = F.cross_entropy(yr[ds.train_mask], labels[ds.train_mask])
        ld = F.cross_entropy(yd[ds.train_mask], labels[ds.train_mask])
        lr.backward(); ld.backward()
        result = {
            "framework": f"dgl_{variant}",
            "reference": "tfs_reference_csr",
            "loss_abs": abs(float(lr) - float(ld)),
            "logits": metrics(yr, yd),
            "input_grad": metrics(xr.grad, xd.grad),
            "parameter_grads": {},
        }
        for layer, (ref_conv, dgl_conv) in enumerate(
                zip(ref.convs, dgl_model.convs)):
            if variant == "cached":
                dgl_weight_grad = dgl_conv.base.weight.grad
                dgl_bias_grad = dgl_conv.bias.grad
            else:
                dgl_weight_grad = dgl_conv.weight.grad
                dgl_bias_grad = dgl_conv.bias.grad
            result["parameter_grads"][f"convs.{layer}.weight"] = metrics(
                ref_conv.weight.grad, dgl_weight_grad)
            result["parameter_grads"][f"convs.{layer}.bias"] = metrics(
                ref_conv.bias.grad, dgl_bias_grad)
        tol = float(os.environ.get("DGL_GRAD_TOL", "5e-4"))
        forward_tol = float(os.environ.get("DGL_FORWARD_TOL", "2e-5"))
        ok = result["logits"]["relative_l2"] < forward_tol
        ok &= result["input_grad"]["relative_l2"] < tol
        ok &= all(v["relative_l2"] < tol
                  for v in result["parameter_grads"].values())
        result["tolerance"] = {"forward_relative_l2": forward_tol,
                                "gradient_relative_l2": tol}
        result["status"] = "pass" if ok else "fail"
        Path(os.environ["HYBRID_OUTPUT"]).write_text(
            json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        raise SystemExit(0 if ok else 3)

    # A full IGB graph cannot safely retain both models' backward graphs at
    # once.  The optional sequential gate records only the reference outputs
    # and parameter gradients, releases that graph, then executes TFS.  This
    # validates the identical training contract without turning the checker
    # itself into the largest memory consumer in the process.
    sequential_check = os.environ.get("HYBRID_CHECK_SEQUENTIAL") == "1"
    if sequential_check:
        logits_bytes = int(x.shape[0]) * OUT_DIM * torch.empty((), dtype=x.dtype).element_size()
        max_logits_bytes = int(os.environ.get(
            "HYBRID_SEQUENTIAL_LOGITS_MAX_BYTES", str(1 << 30)))
        if logits_bytes > max_logits_bytes:
            raise ValueError(
                "sequential correctness gate would retain an oversized logits "
                f"snapshot ({logits_bytes} bytes > {max_logits_bytes}); use a "
                "smaller class-count dataset or implement an on-disk comparator")
        ref = ReferenceGCN(LAYERS, dropout=0.0, in_dim=x.shape[1],
                           hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM)
        state = {name: value.detach().clone()
                 for name, value in ref.state_dict().items()}
        yr = ref(x, graph)
        lr = F.cross_entropy(yr[ds.train_mask], labels[ds.train_mask])
        lr.backward()
        reference_logits = yr.detach().clone()
        reference_loss = float(lr)
        reference_grads = {name: parameter.grad.detach().clone()
                           for name, parameter in ref.named_parameters()}
        del yr, lr, ref
        gc.collect()

        hyb = HybridGCN(threads, LAYERS, dropout=0.0, in_dim=x.shape[1],
                        hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
                        num_nodes=x.shape[0])
        hyb.load_state_dict(state)
        yh = hyb(x, graph)
        lh = F.cross_entropy(yh[ds.train_mask], labels[ds.train_mask])
        lh.backward()
        result = {"loss_abs": abs(reference_loss - float(lh)),
                  "logits": metrics(reference_logits, yh),
                  "input_grad": {"checked": False,
                                  "reason": "immutable input; layer0 compute_dx=false"},
                  "parameter_grads": {}}
        for name, parameter in hyb.named_parameters():
            result["parameter_grads"][name] = metrics(
                reference_grads[name], parameter.grad)
    else:
        ref = ReferenceGCN(LAYERS, dropout=0.0, in_dim=x.shape[1],
                           hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM)
        hyb = HybridGCN(threads, LAYERS, dropout=0.0, in_dim=x.shape[1],
                        hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
                        num_nodes=x.shape[0])
        hyb.load_state_dict(ref.state_dict())
        # Input features are immutable in the authority workload.  Requiring dX
        # would bypass the persistent first-layer contract and validate a path
        # that the 200-epoch training process never executes.
        yr = ref(x, graph); yh = hyb(x, graph)
        lr = F.cross_entropy(yr[ds.train_mask], labels[ds.train_mask])
        lh = F.cross_entropy(yh[ds.train_mask], labels[ds.train_mask])
        lr.backward(); lh.backward()
        result = {"loss_abs": abs(float(lr)-float(lh)), "logits": metrics(yr,yh),
                  "input_grad": {"checked": False,
                                  "reason": "immutable input; layer0 compute_dx=false"},
                  "parameter_grads": {}}
        for (nr,pr),(nh,ph) in zip(ref.named_parameters(),hyb.named_parameters()):
            assert nr == nh
            result["parameter_grads"][nr] = metrics(pr.grad,ph.grad)
    grad_tol = float(os.environ.get(
        "HYBRID_GRAD_TOL",
        "3e-2" if os.environ.get("HYBRID_AMX_BACKWARD") == "1" else "2e-5"))
    forward_tol = float(os.environ.get(
        "HYBRID_FORWARD_TOL",
        "3e-2" if os.environ.get("HYBRID_AMX_FORWARD") == "1" else "2e-5"))
    ok = result["logits"]["relative_l2"] < forward_tol
    ok &= all(v["relative_l2"] < grad_tol for v in result["parameter_grads"].values())
    result["tolerance"] = {"forward_relative_l2": forward_tol,
                           "gradient_relative_l2": grad_tol}
    result["status"] = "pass" if ok else "fail"
    Path(os.environ["HYBRID_OUTPUT"]).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
    raise SystemExit(0 if ok else 3)

path = os.environ["HYBRID_PATH"]
validate_authority_variant(path)
if path == "hybrid":
    model, graph_arg = HybridGCN(threads, LAYERS, in_dim=x.shape[1],
                                 hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
                                 num_nodes=x.shape[0]), graph
elif path == "tfs_reference":
    model, graph_arg = ReferenceGCN(LAYERS, in_dim=x.shape[1],
                                    hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM), graph
elif path == "pyg":
    rows = torch.repeat_interleave(torch.arange(x.shape[0]), graph.degree)
    edge_index = torch.stack((graph.colidx, rows))
    framework_graph_ready_ns = time.perf_counter_ns()
    model, graph_arg = PyGGCN(LAYERS, x.shape[1], HIDDEN_DIM, OUT_DIM), edge_index
elif path in ("dgl", "dgl_stock", "dgl_cached"):
    import dgl
    configure_dgl(dgl, threads)
    dg, dgl_graph_metadata = build_stock_dgl_graph_from_pull_csr(
        graph.rowptr, graph.colidx, x.shape[0])
    framework_graph_ready_ns = time.perf_counter_ns()
    if path == "dgl":
        # Retain the historical edge-weight path for provenance only; the
        # strong comparison below uses dgl_stock and dgl_cached.
        from dgl.nn.pytorch import EdgeWeightNorm
        ew = EdgeWeightNorm(norm="both")(
            dg, torch.ones(dg.num_edges(), dtype=x.dtype))
        model = DGLGCN(LAYERS, x.shape[1], HIDDEN_DIM, OUT_DIM,
                       norm="none", allow_zero_in_degree=True)
        graph_arg = (dg, ew)
    elif path == "dgl_stock":
        model = DGLGCN(LAYERS, x.shape[1], HIDDEN_DIM, OUT_DIM,
                       norm="both", allow_zero_in_degree=True)
        graph_arg = (dg, None)
    else:
        model = DGLCachedGCN(LAYERS, x.shape[1], HIDDEN_DIM, OUT_DIM)
        graph_arg = (dg, graph.scale)
else:
    raise ValueError(path)

requested_dtype = os.environ.get("HYBRID_DTYPE", "fp32").lower()
if requested_dtype == "bf16":
    requested_dtype = "bf16_native"
if requested_dtype not in ("fp32", "bf16_native", "bf16_mixed"):
    raise ValueError("HYBRID_DTYPE must be fp32, bf16_native, or bf16_mixed")
if requested_dtype != "fp32" and path not in ("dgl", "dgl_stock", "dgl_cached", "pyg"):
    raise ValueError("framework BF16 gates are only defined for DGL/PyG")
if requested_dtype == "bf16_native":
    x = x.to(torch.bfloat16)
    model = model.to(torch.bfloat16)
if requested_dtype != "fp32" and path == "dgl":
    # DGL gspmm requires node messages and edge weights to have identical
    # dtype. Autocast converts the node path but does not cast edge features.
    graph_arg = (graph_arg[0], graph_arg[1].to(torch.bfloat16))
if requested_dtype == "bf16_mixed" and path in ("dgl", "dgl_stock"):
    # Strict official CPU autocast: model, Adam state and input stay FP32.
    # Eligible operations choose BF16 inside the autocast scope below.
    model.force_bf16_activations = False

reported_dtype = ("dgl_official_autocast_bf16_fp32_master"
                  if path in ("dgl", "dgl_stock") and
                  requested_dtype == "bf16_mixed" else requested_dtype)
execution_plan_json = "[]"
if (path == "hybrid" and os.environ.get("HYBRID_AMX_FORWARD") == "1" and
        os.environ.get("HYBRID_AMX_BACKWARD") == "1"):
    reported_dtype = "bf16_inputs_fp32_accum_fp32_master"
    # Dtype is a numerical contract only.  Execution families belong in the
    # separate immutable planner metadata so a wide-aggregate layer cannot be
    # mislabeled as ``wide_output`` in downstream CSV aggregation.
    execution_plan_json = json.dumps(
        [plan.to_dict() for plan in getattr(model, "_plans", ())],
        sort_keys=True, separators=(",", ":"))

def persistent_hs_stats(module):
    """Expose cache hits/misses without changing the timed model path."""
    caches = [getattr(conv, "_hs_cache", None)
              for conv in getattr(module, "convs", ())]
    caches += [getattr(conv, "_aggregate_cache", None)
               for conv in getattr(module, "convs", ())]
    caches = [cache for cache in caches if cache is not None]
    return {
        "contract": (sorted({cache.contract_version for cache in caches})[0]
                     if len({cache.contract_version for cache in caches}) == 1
                     else sorted({cache.contract_version for cache in caches})),
        "hits": sum(int(cache.hits) for cache in caches),
        "misses": sum(int(cache.misses) for cache in caches),
        "entries": sum(int(cache.tensor is not None) for cache in caches),
        "build_calls": sum(int(cache.build_calls) for cache in caches),
        "build_ms": sum(float(cache.build_ms) for cache in caches),
        "lookup_ms": sum(float(cache.lookup_ms) for cache in caches),
        "bytes": sum(int(cache.bytes) for cache in caches),
    }


def compute_context():
    if requested_dtype == "bf16_mixed":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()

# Layer hooks are enabled only in separate profile runs. They are deliberately
# absent from formal total-time runs because Python hook overhead would change
# the quantity being measured.
layer_profile = os.environ.get("HYBRID_LAYER_PROFILE") == "1"
layer_backward_ms = [0.0] * LAYERS
layer_backward_begin = [0] * LAYERS
layer_backward_end = [0] * LAYERS
hook_handles = []
if layer_profile:
    for layer_id, conv in enumerate(model.convs):
        # Module full-backward hooks omit most of layer 0 when its data input
        # does not require a gradient. Parameter hooks cover that legitimate
        # boundary without forcing an otherwise-unneeded dX computation.
        for parameter in conv.parameters():
            def parameter_done(gradient, idx=layer_id):
                layer_backward_end[idx] = max(
                    layer_backward_end[idx], time.perf_counter_ns())
                return gradient
            hook_handles.append(parameter.register_hook(parameter_done))

opt = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
model_ready_ns = time.perf_counter_ns()
training_runtime_start_ns = model_ready_ns
data_ready_ms = (data_ready_ns - process_start_ns) / 1e6
framework_graph_ready_ms = (
    framework_graph_ready_ns - process_start_ns) / 1e6
# Backward-compatible column name retained for frozen result readers.
graph_ready_ms = framework_graph_ready_ms
model_ready_ms = (model_ready_ns - process_start_ns) / 1e6
framework_graph_build_ms = (framework_graph_ready_ns - data_ready_ns) / 1e6
model_setup_ms = (model_ready_ns - framework_graph_ready_ns) / 1e6
training_ready_ms = model_ready_ms
warmups=int(os.environ.get("HYBRID_WARMUPS","2")); repeats=int(os.environ.get("HYBRID_REPEATS","7")); seed=int(os.environ.get("HYBRID_SEED","101"))

quality_epochs = int(os.environ.get("HYBRID_TRAIN_EPOCHS", "0"))
if quality_epochs > 0:
    quality_rows = []
    for epoch in range(1, quality_epochs + 1):
        torch.manual_seed(seed * 100000 + epoch)
        train_begin = time.perf_counter_ns()
        model.train(); opt.zero_grad(set_to_none=True)
        with compute_context():
            logits = model(x, graph_arg)
        loss = F.cross_entropy(logits.float()[ds.train_mask], labels[ds.train_mask])
        loss.backward(); opt.step()
        train_end = time.perf_counter_ns()
        eval_begin = time.perf_counter_ns()
        model.eval()
        with torch.no_grad():
            with compute_context():
                logits = model(x, graph_arg)
            logits = logits.float()
            val_loss = F.cross_entropy(logits[ds.valid_mask], labels[ds.valid_mask])
            test_loss = F.cross_entropy(logits[ds.test_mask], labels[ds.test_mask])
            val_acc = (logits[ds.valid_mask].argmax(-1) == labels[ds.valid_mask]).float().mean()
            test_acc = (logits[ds.test_mask].argmax(-1) == labels[ds.test_mask]).float().mean()
        eval_end = time.perf_counter_ns()
        quality_rows.append({
            "path": path, "layers": LAYERS, "dtype": reported_dtype,
            "dtype_contract": reported_dtype,
            "execution_plans": execution_plan_json,
            "threads": threads, "seed": seed, "epoch": epoch,
            "train_loss": float(loss.detach()), "val_loss": float(val_loss),
            "val_accuracy": float(val_acc), "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "train_step_ms": (train_end - train_begin) / 1e6,
            "evaluation_ms": (eval_end - eval_begin) / 1e6,
            "graph_ready_ms": graph_ready_ms,
            "data_ready_ms": data_ready_ms,
            "framework_graph_build_ms": framework_graph_build_ms,
            "framework_graph_ready_ms": framework_graph_ready_ms,
            "model_setup_ms": model_setup_ms,
            "model_ready_ms": model_ready_ms,
            "training_ready_ms": training_ready_ms,
            "persistent_hs_cache": persistent_hs_stats(model),
        })
    quality_out = Path(os.environ["HYBRID_OUTPUT"])
    with quality_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(quality_rows[0]))
        writer.writeheader(); writer.writerows(quality_rows)
    training_runtime_end_ns = time.perf_counter_ns()
    timing_payload = {
        "clock": "time.perf_counter_ns",
        "process_marker_ns": process_start_ns,
        "data_ready_ns": data_ready_ns,
        "framework_graph_ready_ns": framework_graph_ready_ns,
        "model_ready_ns": model_ready_ns,
        "training_runtime_start_ns": training_runtime_start_ns,
        "training_runtime_end_ns": training_runtime_end_ns,
        "data_ready_ms_from_process_marker": data_ready_ms,
        "data_ready_ms": data_ready_ms,
        "framework_graph_build_ms": framework_graph_build_ms,
        "framework_graph_ready_ms_from_process_marker": framework_graph_ready_ms,
        "framework_graph_ready_ms": framework_graph_ready_ms,
        "model_setup_ms": model_setup_ms,
        "model_ready_ms_from_process_marker": model_ready_ms,
        "model_ready_ms": model_ready_ms,
        "training_ready_ms": training_ready_ms,
        "training_runtime_wall_ms":
            (training_runtime_end_ns - training_runtime_start_ns) / 1e6,
    }
    timing_out = os.environ.get("HYBRID_TIMING_METADATA", "").strip()
    if timing_out:
        Path(timing_out).write_text(json.dumps(timing_payload, indent=2) + "\n")
    print(json.dumps({"status": "pass", "mode": "training_quality",
                      "path": path, "layers": LAYERS,
                      "dtype": reported_dtype, "threads": threads,
                      "seed": seed, "epochs": quality_epochs,
                      "timing": timing_payload,
                      "final": quality_rows[-1], "output": str(quality_out),
                      "graph_ready_ms": graph_ready_ms,
                      "training_runtime_wall_ms":
                          (time.perf_counter_ns() - training_runtime_start_ns) / 1e6}))
    raise SystemExit(0)

raw=[]
for step in range(1,warmups+repeats+1):
    torch.manual_seed(seed*100000+step)
    layer_backward_ms[:] = [0.0] * LAYERS
    layer_backward_begin[:] = [0] * LAYERS
    layer_backward_end[:] = [0] * LAYERS
    begin=time.perf_counter_ns(); model.train(); opt.zero_grad(set_to_none=True); f0=time.perf_counter_ns()
    layer_forward_ms=[]; activation_ms=[]
    if layer_profile:
        value=x
        with compute_context():
            for layer_id,conv in enumerate(model.convs):
                layer_input=value
                q0=time.perf_counter_ns()
                if path == "dgl":
                    dg,ew=graph_arg
                    if requested_dtype == "bf16_mixed":
                        value = value.to(torch.bfloat16)
                    value=conv(dg,value,edge_weight=ew)
                else:
                    value=conv(value,graph_arg)
                q1=time.perf_counter_ns(); layer_forward_ms.append((q1-q0)/1e6)
                def layer_enter(gradient, idx=layer_id):
                    layer_backward_begin[idx] = time.perf_counter_ns()
                    return gradient
                value.register_hook(layer_enter)
                if layer_input.requires_grad:
                    def input_done(gradient, idx=layer_id):
                        layer_backward_end[idx] = max(
                            layer_backward_end[idx], time.perf_counter_ns())
                        return gradient
                    layer_input.register_hook(input_done)
                if layer_id + 1 < LAYERS:
                    a0=time.perf_counter_ns(); value=F.relu(value); value=F.dropout(value,p=0.5,training=True)
                    a1=time.perf_counter_ns(); activation_ms.append((a1-a0)/1e6)
        logits=value
    else:
        with compute_context():
            logits=model(x,graph_arg)
    f1=time.perf_counter_ns()
    loss=F.cross_entropy(logits.float()[ds.train_mask],labels[ds.train_mask]); f2=time.perf_counter_ns()
    loss.backward(); f3=time.perf_counter_ns(); opt.step(); end=time.perf_counter_ns()
    if layer_profile:
        for layer_id in range(LAYERS):
            if layer_backward_begin[layer_id] and layer_backward_end[layer_id]:
                layer_backward_ms[layer_id] = (
                    layer_backward_end[layer_id] -
                    layer_backward_begin[layer_id]) / 1e6
    record={"path":path,"layers":LAYERS,"dtype":reported_dtype,
                "dtype_contract":reported_dtype,
                "execution_plans":execution_plan_json,
                "threads":threads,"seed":seed,"step":step,"is_warmup":step<=warmups,
                "elapsed_ms":(end-begin)/1e6,"forward_ms":(f1-f0)/1e6,"loss_ms":(f2-f1)/1e6,
                "backward_ms":(f3-f2)/1e6,"optimizer_ms":(end-f3)/1e6,"loss":float(loss.detach()),
                "persistent_hs_cache":persistent_hs_stats(model)}
    if layer_profile:
        for layer_id in range(LAYERS):
            record[f"layer{layer_id}_forward_ms"] = layer_forward_ms[layer_id]
            record[f"layer{layer_id}_backward_ms"] = layer_backward_ms[layer_id]
        for activation_id in range(LAYERS - 1):
            record[f"activation{activation_id}_ms"] = activation_ms[activation_id]
    raw.append(record)
out=Path(os.environ["HYBRID_OUTPUT"])
with out.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
print(json.dumps({"status":"pass","path":path,"threads":threads,"seed":seed,
                  "output":str(out),"graph_ready_ms":graph_ready_ms,
                  "training_runtime_wall_ms":
                      (time.perf_counter_ns() - training_runtime_start_ns) / 1e6}))
