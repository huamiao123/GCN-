#!/usr/bin/env python3
import csv
import contextlib
import json
import os
import time
from pathlib import Path

# Earliest in-script monotonic marker.  The launcher records the true cold
# process wall (including interpreter startup); this marker includes imports,
# dataset preparation, framework graph construction, and model setup.
process_start_ns = time.perf_counter_ns()

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
except ImportError:
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
from tfs_train.dimension_dispatch import AggregateWideAMX, WideOutputAMX
from tfs_train.authority_autograd import (
    AggregateCachedT0 as _AuthorityAggregateCachedT0,
    AggregateFirst as _AuthorityAggregateFirst,
    TransformFirst as _AuthorityTransformFirst,
)
from tfs_train.standard_runtime import configure_dgl, validate_authority_variant
from tfs_train.standard_dgl import (DGLGCN,
                                    build_stock_dgl_graph_from_pull_csr)


def load_products_cache(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    graph = CSRGraph(payload["rowptr"], payload["colidx"],
                     payload["scale"], payload["degree"],
                     payload["schedule"])
    return NodePropertyDataset(payload["x"], payload["labels"], graph,
                               payload["train_mask"], payload["valid_mask"],
                               payload["test_mask"])


def make_tail_correctness_fixture():
    # This gate exercises the products-specific K=100 and D=47 tails before
    # any full-graph timing. The performance matrix itself always uses the
    # complete official graph loaded from PRODUCTS_CACHE.
    n = 4096
    src = torch.arange(n, dtype=torch.long)
    edge_index = torch.stack((src, (src * 17 + 23) % n))
    graph = preprocess_undirected_fast(edge_index, n, torch.float32)
    input_dim = int(os.environ.get("HYBRID_INPUT_DIM", "100"))
    out_dim = int(os.environ.get("HYBRID_OUT_DIM", "47"))
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
        if ctx.amx:
            dx, dw, db, _ = backend().c3_backward_amx_v2(
                grad_output.contiguous(), hs, weight, rowptr, colidx, scale,
                ctx.threads, compute_dx)
        else:
            dx, dw, db, _, _, _ = backend().c3_backward_selective(
                grad_output.contiguous(), hs, weight, rowptr, colidx, scale,
                schedule, ctx.threads, False, 8, compute_dx)
        if not compute_dx:
            dx = None
        # forward() has eleven inputs.  cached_hs, backward_hs and the
        # optional replica tensor are cache
        # storage tensors and are not differentiable; keep arity exact on
        # both cache hit and cache miss paths.
        return (dx, dw, db, None, None, None, None, None, None, None, None)


def cached_c3_forward(x, cached_hs, weight, bias, rowptr, colidx, scale,
                      threads, transform_first, cached_hs_replicas=None):
    """Dispatch the canonical V1 or optional padded-Hs V2 consumer."""
    if cached_hs.shape[1] != x.shape[1]:
        return backend().c3_forward_cached_hs_padded_amx_v2(
            x, cached_hs_replicas if cached_hs_replicas is not None else cached_hs,
            weight, bias, rowptr, colidx, scale,
            int(threads), transform_first)
    return backend().c3_forward_cached_hs_amx_v1(
        x, cached_hs_replicas if cached_hs_replicas is not None else cached_hs,
        weight, bias, rowptr, colidx, scale,
        int(threads), transform_first)


class AggregateFirst(_C3Base):
    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule, threads,
                cached_hs=None, backward_hs=None, cached_hs_replicas=None):
        if os.environ.get("HYBRID_AMX_FORWARD") == "1":
            if cached_hs is None:
                out, hs = backend().c3_forward_amx_v2(
                    x, weight, bias, rowptr, colidx, scale, int(threads), False)
            else:
                out, hs = cached_c3_forward(
                    x, cached_hs, weight, bias, rowptr, colidx, scale,
                    threads, False, cached_hs_replicas)
        else:
            out, hs, _ = backend().c3_forward(
                x, weight, bias, rowptr, colidx, scale, schedule, int(threads))
        hs_saved = hs if backward_hs is None else backward_hs
        ctx.save_for_backward(hs_saved, weight, rowptr, colidx, scale, schedule)
        ctx.threads = int(threads)
        ctx.amx = os.environ.get("HYBRID_AMX_BACKWARD") == "1"
        return out


class TransformFirst(_C3Base):
    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule, threads,
                cached_hs=None, backward_hs=None, cached_hs_replicas=None):
        if os.environ.get("HYBRID_AMX_FORWARD") == "1":
            if cached_hs is None:
                out, hs = backend().c3_forward_amx_v2(
                    x, weight, bias, rowptr, colidx, scale, int(threads), True)
            else:
                out, hs = cached_c3_forward(
                    x, cached_hs, weight, bias, rowptr, colidx, scale,
                    threads, True, cached_hs_replicas)
        else:
            out, hs, _ = backend().c3_forward_transform(
                x, weight, bias, rowptr, colidx, scale, schedule, int(threads))
        hs_saved = hs if backward_hs is None else backward_hs
        ctx.save_for_backward(hs_saved, weight, rowptr, colidx, scale, schedule)
        ctx.threads = int(threads)
        ctx.amx = os.environ.get("HYBRID_AMX_BACKWARD") == "1"
        return out


class AggregateCachedT0(torch.autograd.Function):
    """Opt-in V3 layer-0 aggregate cache (T0 = B*Hs0)."""

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


# Migration bridge: runtime uses the shared primitives while the old local
# definitions remain available only for this cycle's numerical comparison.
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
        pad_to = int(os.environ.get("HYBRID_HS_PAD_TO", "0")) or None
        use_t0 = (cache_static_hs and self.order == "aggregate" and
                  (plan.static_pulled if plan is not None else
                   os.environ.get("HYBRID_STATIC_AGG_CACHE") == "1"))
        self._aggregate_cache = (PersistentAggregateCache()
                                 if use_t0 else None)
        self._hs_cache = (PersistentHsCache(pad_to=pad_to)
                          if cache_static_hs and not use_t0 else None)
        self.dimension_path = (plan.dimension_path if plan is not None else
                               ("wide_aggregate" if order == "aggregate" and d > 128
                                else "wide_output" if order == "transform" and d > 128
                                else "wide_k" if k > 128 else "c3"))
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
        if (variant.startswith("aggregate_highd_") and
                os.environ.get("HYBRID_AMX_FORWARD") == "1" and
                os.environ.get("HYBRID_AMX_BACKWARD") == "1"):
            return AggregateWideAMX.apply(
                x, self.weight, self.bias, graph.rowptr, graph.colidx,
                scale, graph.schedule, self.threads, self.plan)
        if (variant.startswith("transform_highd_") and
                os.environ.get("HYBRID_AMX_FORWARD") == "1" and
                os.environ.get("HYBRID_AMX_BACKWARD") == "1"):
            return WideOutputAMX.apply(
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
        cached_hs_replicas = None
        if (self._hs_cache is not None and
                os.environ.get("HYBRID_PERSISTENT_HS_CACHE") == "1" and
                os.environ.get("HYBRID_AMX_FORWARD") == "1" and
                x.dtype == torch.float32 and scale.dtype == torch.float32):
            hs_cache = get_or_build_if_within_budget(
                self._hs_cache, x, graph, self.threads, backend())
            if hs_cache is not None:
                cached_storage, _ = hs_cache
                # V2 returns the aligned storage for backward, while the native
                # consumer receives the compact logical-width view for forward.
                cached_hs = self._hs_cache.forward_tensor
                if cached_hs is None:
                    cached_hs = cached_storage
                if cached_hs.shape[1] != x.shape[1]:
                    raise RuntimeError("persistent Hs forward view must be logical-width")
                if cached_storage.shape[1] != cached_hs.shape[1]:
                    backward_hs = cached_storage
                cached_hs_replicas = self._hs_cache.replica_tensor
        return fn.apply(x, self.weight, self.bias, graph.rowptr, graph.colidx,
                        scale, graph.schedule, self.threads, cached_hs,
                        backward_hs, cached_hs_replicas)


class LegacyHybridGCN(torch.nn.Module):
    def __init__(self, threads, layers=2, dropout=0.5, in_dim=100,
                 hidden_dim=128, out_dim=47, num_nodes=None):
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be at least 2")
        dims = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
        self._dims = dims
        # Plan against the actual graph size before constructing the modules.
        # This keeps CSR-width decisions and cache contracts identical to the
        # plan that is logged; a construction-time n=1 placeholder could hide
        # an int32/int64 boundary in a future large graph.
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
    def __init__(self, layers=2, dropout=0.5, in_dim=100,
                 hidden_dim=128, out_dim=47):
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
    def __init__(self, layers=2, in_dim=100, hidden_dim=128, out_dim=47):
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


def metrics(a, b):
    d = (a.double() - b.double()).reshape(-1)
    denom = torch.linalg.vector_norm(a.double().reshape(-1)).clamp_min(1e-30)
    av = a.double().reshape(-1)
    bv = b.double().reshape(-1)
    cosine = torch.dot(av, bv) / (torch.linalg.vector_norm(av) *
                                  torch.linalg.vector_norm(bv)).clamp_min(1e-30)
    return {"relative_l2": float(torch.linalg.vector_norm(d) / denom),
            "max_abs": float(d.abs().max()), "cosine": float(cosine)}


def persistent_hs_stats(module):
    caches = [getattr(conv, "_hs_cache", None)
              for conv in getattr(module, "convs", ())]
    caches += [getattr(conv, "_aggregate_cache", None)
              for conv in getattr(module, "convs", ())]
    caches = [cache for cache in caches if cache is not None]
    contracts = sorted({cache.contract_version for cache in caches})
    return {"contract": contracts[0] if len(contracts) == 1 else contracts,
            "hits": sum(int(cache.hits) for cache in caches),
            "misses": sum(int(cache.misses) for cache in caches),
            "entries": sum(int(cache.tensor is not None) for cache in caches),
            "build_calls": sum(int(cache.build_calls) for cache in caches),
            "build_ms": sum(float(cache.build_ms) for cache in caches),
            "lookup_ms": sum(float(cache.lookup_ms) for cache in caches),
            "bytes": sum(int(cache.bytes) for cache in caches)}


threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
OUT_DIM = int(os.environ.get("HYBRID_OUT_DIM", "47"))
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
      else load_products_cache(os.environ["PRODUCTS_CACHE"]))
# pandas exposes the official CSV feature matrix with column-major strides.
# Canonicalize it once before any timed method so every framework receives the
# same row-major input and no layer repays a full N x K copy each epoch.
x, labels, graph = ds.x.contiguous(), ds.labels, ds.graph
if os.environ.get("PRODUCTS_PAD_INPUT_128") == "1" and x.shape[1] < 128:
    x = torch.cat((x, torch.zeros(x.shape[0], 128 - x.shape[1],
                                  dtype=x.dtype)), dim=1).contiguous()
data_ready_ns = time.perf_counter_ns()
# TFS consumes this canonical pull CSR directly.  Frameworks that construct
# another graph object replace this marker below after that object is ready.
framework_graph_ready_ns = data_ready_ns

if os.environ.get("HYBRID_CHECK") == "1":
    if os.environ.get("HYBRID_CHECK_DGL") == "1":
        import dgl
        configure_dgl(dgl, threads)
        dg, _dgl_graph_metadata = build_stock_dgl_graph_from_pull_csr(
            graph.rowptr, graph.colidx, x.shape[0])
        ref = ReferenceGCN(LAYERS, dropout=0.0, in_dim=x.shape[1],
                          hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM)
        dgl_model = DGLGCN(LAYERS, x.shape[1], HIDDEN_DIM, OUT_DIM,
                           norm="both")
        for ref_conv, dgl_conv in zip(ref.convs, dgl_model.convs):
            dgl_conv.weight.data.copy_(ref_conv.weight.data)
            dgl_conv.bias.data.copy_(ref_conv.bias.data)
        ref.eval(); dgl_model.eval()
        xr = x.clone().requires_grad_(True)
        xd = x.clone().requires_grad_(True)
        yr = ref(xr, graph)
        yd = dgl_model(xd, (dg, None))
        lr = F.cross_entropy(yr[ds.train_mask], labels[ds.train_mask])
        ld = F.cross_entropy(yd[ds.train_mask], labels[ds.train_mask])
        lr.backward(); ld.backward()
        result = {"framework": "dgl_stock", "reference": "tfs_reference_csr",
                  "loss_abs": abs(float(lr) - float(ld)),
                  "logits": metrics(yr, yd), "input_grad": metrics(xr.grad, xd.grad),
                  "parameter_grads": {}}
        for i, (ref_conv, dgl_conv) in enumerate(zip(ref.convs, dgl_model.convs)):
            result["parameter_grads"][f"convs.{i}.weight"] = metrics(
                ref_conv.weight.grad, dgl_conv.weight.grad)
            result["parameter_grads"][f"convs.{i}.bias"] = metrics(
                ref_conv.bias.grad, dgl_conv.bias.grad)
        grad_tol = float(os.environ.get("DGL_GRAD_TOL", "5e-4"))
        forward_tol = float(os.environ.get("DGL_FORWARD_TOL", "2e-5"))
        ok = result["logits"]["relative_l2"] < forward_tol
        ok &= result["input_grad"]["relative_l2"] < grad_tol
        ok &= all(v["relative_l2"] < grad_tol
                  for v in result["parameter_grads"].values())
        result["tolerance"] = {"forward_relative_l2": forward_tol,
                                "gradient_relative_l2": grad_tol}
        result["status"] = "pass" if ok else "fail"
        Path(os.environ["HYBRID_OUTPUT"]).write_text(
            json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        raise SystemExit(0 if ok else 3)
    ref = ReferenceGCN(LAYERS, dropout=0.0, in_dim=x.shape[1],
                       hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM)
    hyb = HybridGCN(threads, LAYERS, dropout=0.0, in_dim=x.shape[1],
                    hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
                    num_nodes=x.shape[0])
    hyb.load_state_dict(ref.state_dict())
    # Authority training treats input features as immutable data.  Requiring
    # dX here would deliberately disable the static layer-0 V3 path and test
    # a different execution plan from the one used by the 200-epoch run.
    xr = x.clone(); xh = x.clone()
    yr = ref(xr, graph); yh = hyb(xh, graph)
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
elif path in ("dgl", "dgl_stock"):
    import dgl
    configure_dgl(dgl, threads)
    dg, dgl_graph_metadata = build_stock_dgl_graph_from_pull_csr(
        graph.rowptr, graph.colidx, x.shape[0])
    framework_graph_ready_ns = time.perf_counter_ns()
    if path == "dgl":
        from dgl.nn.pytorch import EdgeWeightNorm
        ew = EdgeWeightNorm(norm="both")(dg,torch.ones(dg.num_edges(),dtype=x.dtype))
        model = DGLGCN(LAYERS, x.shape[1], HIDDEN_DIM, OUT_DIM, norm="none")
        graph_arg = (dg, ew)
    else:
        model = DGLGCN(LAYERS, x.shape[1], HIDDEN_DIM, OUT_DIM, norm="both")
        graph_arg = (dg, None)
else:
    raise ValueError(path)

requested_dtype = os.environ.get("HYBRID_DTYPE", "fp32").lower()
if requested_dtype == "bf16":
    requested_dtype = "bf16_native"
if requested_dtype not in ("fp32", "bf16_native", "bf16_mixed"):
    raise ValueError("HYBRID_DTYPE must be fp32, bf16_native, or bf16_mixed")
if requested_dtype != "fp32" and path not in ("dgl", "dgl_stock", "pyg"):
    raise ValueError("framework BF16 gates are only defined for DGL/PyG")
if requested_dtype == "bf16_native":
    x = x.to(torch.bfloat16)
    model = model.to(torch.bfloat16)
if requested_dtype != "fp32" and path in ("dgl", "dgl_stock"):
    # DGL gspmm requires node messages and edge weights to have identical
    # dtype. Autocast converts the node path but does not cast edge features.
    if graph_arg[1] is not None:
        graph_arg = (graph_arg[0], graph_arg[1].to(torch.bfloat16))
    if requested_dtype == "bf16_mixed":
        # DGL may execute aggregation before its dense transform, so autocast
        # alone leaves the first sparse input FP32. Keep activations/metadata
        # BF16 explicitly while learnable parameters and Adam remain FP32.
        x = x.to(torch.bfloat16)
        model.force_bf16_activations = True

reported_dtype = (
    "bf16_inputs_fp32_accum_fp32_master"
    if path == "hybrid" and os.environ.get("HYBRID_AMX_FORWARD") == "1"
    and os.environ.get("HYBRID_AMX_BACKWARD") == "1"
    else requested_dtype
)


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
model_ready_ms = (model_ready_ns - process_start_ns) / 1e6
framework_graph_build_ms = (framework_graph_ready_ns - data_ready_ns) / 1e6
model_setup_ms = (model_ready_ns - framework_graph_ready_ns) / 1e6
training_ready_ms = model_ready_ms
warmups=int(os.environ.get("HYBRID_WARMUPS","2")); repeats=int(os.environ.get("HYBRID_REPEATS","7")); seed=int(os.environ.get("HYBRID_SEED","101"))

quality_epochs = int(os.environ.get("HYBRID_TRAIN_EPOCHS", "0"))
if quality_epochs > 0:
    detailed = os.environ.get("HYBRID_DETAILED_PROFILE") == "1"
    profile_phase = {"name": "idle"}
    layer_train_forward_ms = [0.0] * LAYERS
    layer_eval_forward_ms = [0.0] * LAYERS
    layer_backward_ms_detail = [0.0] * LAYERS
    layer_forward_start = [0] * LAYERS
    layer_backward_start = [0] * LAYERS
    op_profile_rows = []
    detail_handles = []

    def detail_forward_pre(module, inputs, idx):
        if detailed:
            layer_forward_start[idx] = time.perf_counter_ns()

    def detail_forward_post(module, inputs, output, idx):
        if detailed and layer_forward_start[idx]:
            elapsed = (time.perf_counter_ns() - layer_forward_start[idx]) / 1e6
            if profile_phase["name"] == "train":
                layer_train_forward_ms[idx] += elapsed
            elif profile_phase["name"] == "eval":
                layer_eval_forward_ms[idx] += elapsed
            layer_forward_start[idx] = 0

    def detail_backward_pre(module, grad_inputs, idx):
        if detailed:
            layer_backward_start[idx] = time.perf_counter_ns()

    def detail_backward_post(module, grad_inputs, grad_outputs, idx):
        if detailed and layer_backward_start[idx]:
            layer_backward_ms_detail[idx] += (time.perf_counter_ns() - layer_backward_start[idx]) / 1e6
            layer_backward_start[idx] = 0

    if detailed:
        for idx, conv in enumerate(model.convs):
            detail_handles.append(conv.register_forward_pre_hook(
                lambda module, inputs, idx=idx: detail_forward_pre(module, inputs, idx)))
            detail_handles.append(conv.register_forward_hook(
                lambda module, inputs, output, idx=idx: detail_forward_post(module, inputs, output, idx)))
            if hasattr(conv, "register_full_backward_pre_hook"):
                detail_handles.append(conv.register_full_backward_pre_hook(
                    lambda module, grad_inputs, idx=idx: detail_backward_pre(module, grad_inputs, idx)))
            detail_handles.append(conv.register_full_backward_hook(
                lambda module, grad_inputs, grad_outputs, idx=idx: detail_backward_post(module, grad_inputs, grad_outputs, idx)))

    def detail_model_call(call, epoch, phase):
        profile_phase["name"] = phase
        if detailed and path.startswith("dgl") and os.environ.get("HYBRID_DGL_OP_PROFILE", "1") == "1":
            with torch.autograd.profiler.profile(use_cpu=True, record_shapes=False) as prof:
                value = call()
            for event in prof.key_averages():
                op_profile_rows.append({
                    "epoch": epoch, "phase": phase, "op": event.key,
                    "calls": int(event.count),
                    "self_cpu_ms": float(event.self_cpu_time_total) / 1000.0,
                    "cpu_ms": float(event.cpu_time_total) / 1000.0,
                })
            return value
        return call()

    quality_rows = []
    for epoch in range(1, quality_epochs + 1):
        torch.manual_seed(seed * 100000 + epoch)
        layer_train_forward_ms[:] = [0.0] * LAYERS
        layer_eval_forward_ms[:] = [0.0] * LAYERS
        layer_backward_ms_detail[:] = [0.0] * LAYERS
        train_begin = time.perf_counter_ns()
        model.train(); opt.zero_grad(set_to_none=True)
        with compute_context():
            logits = detail_model_call(lambda: model(x, graph_arg), epoch, "train")
        loss = F.cross_entropy(logits.float()[ds.train_mask], labels[ds.train_mask])
        loss.backward(); opt.step()
        train_end = time.perf_counter_ns()
        eval_begin = time.perf_counter_ns()
        model.eval()
        with torch.no_grad():
            with compute_context():
                logits = detail_model_call(lambda: model(x, graph_arg), epoch, "eval")
            logits = logits.float()
            val_loss = F.cross_entropy(logits[ds.valid_mask], labels[ds.valid_mask])
            test_loss = F.cross_entropy(logits[ds.test_mask], labels[ds.test_mask])
            val_acc = (logits[ds.valid_mask].argmax(-1) == labels[ds.valid_mask]).float().mean()
            test_acc = (logits[ds.test_mask].argmax(-1) == labels[ds.test_mask]).float().mean()
        eval_end = time.perf_counter_ns()
        row = {
            "path": path, "layers": LAYERS, "dtype": reported_dtype,
            "threads": threads, "seed": seed, "epoch": epoch,
            "train_loss": float(loss.detach()), "val_loss": float(val_loss),
            "val_accuracy": float(val_acc), "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "train_step_ms": (train_end - train_begin) / 1e6,
            "evaluation_ms": (eval_end - eval_begin) / 1e6,
            "data_ready_ms": data_ready_ms,
            "framework_graph_build_ms": framework_graph_build_ms,
            "framework_graph_ready_ms": framework_graph_ready_ms,
            "model_setup_ms": model_setup_ms,
            "model_ready_ms": model_ready_ms,
            "training_ready_ms": training_ready_ms,
            "persistent_hs_cache": persistent_hs_stats(model),
        }
        if detailed:
            for idx in range(LAYERS):
                row[f"train_layer{idx}_forward_ms"] = layer_train_forward_ms[idx]
                row[f"train_layer{idx}_backward_ms"] = layer_backward_ms_detail[idx]
                row[f"eval_layer{idx}_forward_ms"] = layer_eval_forward_ms[idx]
        quality_rows.append(row)
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
    if detailed:
        op_out = Path(os.environ.get("HYBRID_OP_PROFILE", str(quality_out.with_suffix(".ops.jsonl"))))
        with op_out.open("w") as stream:
            for item in op_profile_rows:
                stream.write(json.dumps(item) + "\n")
        for handle in detail_handles:
            handle.remove()
    print(json.dumps({"status": "pass", "mode": "training_quality",
                      "path": path, "layers": LAYERS,
                      "dtype": reported_dtype, "threads": threads,
                      "seed": seed, "epochs": quality_epochs,
                      "timing": timing_payload,
                      "detailed_profile": detailed,
                      "op_profile": str(op_out) if detailed else None,
                      "final": quality_rows[-1], "output": str(quality_out)}))
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
                if path in ("dgl", "dgl_stock"):
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
    record={"path":path,"layers":LAYERS,"dtype":reported_dtype,"threads":threads,"seed":seed,"step":step,"is_warmup":step<=warmups,
                "elapsed_ms":(end-begin)/1e6,"forward_ms":(f1-f0)/1e6,"loss_ms":(f2-f1)/1e6,
                "backward_ms":(f3-f2)/1e6,"optimizer_ms":(end-f3)/1e6,"loss":float(loss.detach())}
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
print(json.dumps({"status":"pass","path":path,"threads":threads,"seed":seed,"output":str(out)}))
