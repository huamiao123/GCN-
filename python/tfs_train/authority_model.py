"""Canonical authority-model implementation shared by all real-graph runners.

Dataset runners own loading, labels and framework baselines only.  Keeping the
TFS model here makes planner, cache and native-dispatch semantics identical.
"""
import os

import torch
import torch.nn.functional as F

from .aggregate_saved import AggregateSavedFunction
from .authority_autograd import AggregateCachedT0, AggregateFirst, TransformFirst
from .dimension_dispatch import AggregateWideAMX, WideOutputAMX
from .execution_plan import emit_plan_logs, plan_layers
from .native import backend
from .persistent_hs_cache import (
    PersistentAggregateCache, PersistentHsCache, get_or_build_if_within_budget,
)


class HybridConv(torch.nn.Module):
    """One planner-governed TFS layer for the authority runners."""
    def __init__(self, k, d, order, threads, cache_static_hs=False, plan=None):
        super().__init__()
        if plan is None:
            raise ValueError("HybridConv requires an authority execution plan")
        self.weight = torch.nn.Parameter(torch.empty(k, d))
        self.bias = torch.nn.Parameter(torch.zeros(d))
        self.plan, self.order, self.threads = plan, plan.order, int(threads)
        cache_static_hs = bool(plan.static_hs) and plan.execution_variant != "aggregate_saved_v4"
        pad_to = int(os.environ.get("HYBRID_HS_PAD_TO", "0")) or None
        use_t0 = cache_static_hs and self.order == "aggregate" and bool(plan.static_pulled)
        self._aggregate_cache = PersistentAggregateCache() if use_t0 else None
        self._hs_cache = PersistentHsCache(pad_to=pad_to) if cache_static_hs and not use_t0 else None
        self.dimension_path = plan.dimension_path
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, graph):
        if torch.is_grad_enabled() and bool(x.requires_grad) != bool(self.plan.compute_dx):
            raise RuntimeError(
                "TFS plan/autograd mismatch: "
                f"plan_id={self.plan.plan_id} planned_compute_dx={int(self.plan.compute_dx)} "
                f"actual_compute_dx={int(bool(x.requires_grad))}")
        scale, variant = graph.scale.to(x.dtype), self.plan.execution_variant
        amx = os.environ.get("HYBRID_AMX_FORWARD") == "1" and os.environ.get("HYBRID_AMX_BACKWARD") == "1"
        if variant.startswith("aggregate_highd_") and amx:
            return AggregateWideAMX.apply(x, self.weight, self.bias, graph.rowptr, graph.colidx,
                                          scale, graph.schedule, self.threads, self.plan)
        if variant.startswith("transform_highd_") and amx:
            return WideOutputAMX.apply(x, self.weight, self.bias, graph.rowptr, graph.colidx,
                                       scale, graph.schedule, self.threads, self.plan)
        if variant == "aggregate_saved_v4" and amx:
            return AggregateSavedFunction.apply(x, self.weight, self.bias, graph.rowptr, graph.colidx,
                                                scale, graph.schedule, self.threads, self.plan)
        if (variant == "aggregate_static_v3" and self._aggregate_cache is not None and
                os.environ.get("HYBRID_PERSISTENT_HS_CACHE") == "1" and
                os.environ.get("HYBRID_AMX_FORWARD") == "1" and x.dtype == torch.float32 and
                scale.dtype == torch.float32 and not x.requires_grad):
            aggregate_cache = get_or_build_if_within_budget(
                self._aggregate_cache, x, graph, self.threads, backend())
            if aggregate_cache is not None:
                cached_t0, _ = aggregate_cache
                return AggregateCachedT0.apply(x, self.weight, self.bias, graph.rowptr, graph.colidx,
                                               scale, graph.schedule, self.threads, cached_t0, self.plan)
        if variant not in {"native_c3", "native_wide_k"}:
            raise RuntimeError(f"unhandled authority execution variant: {variant}")
        cached_hs = backward_hs = cached_hs_replicas = None
        if (self._hs_cache is not None and os.environ.get("HYBRID_PERSISTENT_HS_CACHE") == "1" and
                os.environ.get("HYBRID_AMX_FORWARD") == "1" and x.dtype == torch.float32 and
                scale.dtype == torch.float32):
            hs_cache = get_or_build_if_within_budget(self._hs_cache, x, graph, self.threads, backend())
            if hs_cache is not None:
                cached_storage, _ = hs_cache
                cached_hs = self._hs_cache.forward_tensor
                if cached_hs is None:
                    cached_hs = cached_storage
                if cached_hs.shape[1] != x.shape[1]:
                    raise RuntimeError("persistent Hs forward view must be logical-width")
                if cached_storage.shape[1] != cached_hs.shape[1]:
                    backward_hs = cached_storage
                cached_hs_replicas = self._hs_cache.replica_tensor
        fn = AggregateFirst if self.order == "aggregate" else TransformFirst
        return fn.apply(x, self.weight, self.bias, graph.rowptr, graph.colidx, scale,
                        graph.schedule, self.threads, cached_hs, backward_hs,
                        cached_hs_replicas, self.plan)


class HybridGCN(torch.nn.Module):
    """Canonical planner-backed GCN; dimensions remain runner configuration."""
    def __init__(self, threads, layers=2, dropout=0.5, in_dim=100,
                 hidden_dim=128, out_dim=47, num_nodes=None):
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be at least 2")
        if num_nodes is None:
            raise ValueError("HybridGCN requires num_nodes for the authority template")
        dims = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
        self._dims, self._plan_node_count = dims, int(num_nodes)
        self._plans = plan_layers(self._plan_node_count, dims, feature_static_first=True,
                                  threads=int(threads))
        self._plans_logged = False
        self.convs = torch.nn.ModuleList([
            HybridConv(dims[i], dims[i + 1], self._plans[i].order, threads,
                       cache_static_hs=(i == 0), plan=self._plans[i])
            for i in range(layers)
        ])
        self.dropout = dropout

    def forward(self, x, graph):
        if not self._plans_logged:
            if int(x.shape[0]) != self._plan_node_count:
                raise RuntimeError("authority graph node count changed after planning: "
                                   f"planned={self._plan_node_count} actual={int(x.shape[0])}")
            emit_plan_logs(self._plans)
            self._plans_logged = True
        for conv in self.convs[:-1]:
            x = F.dropout(F.relu(conv(x, graph)), p=self.dropout, training=self.training)
        return self.convs[-1](x, graph)
