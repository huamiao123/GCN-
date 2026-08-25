"""Static first-layer Hs cache for the r5 AMX C3 path.

The cache intentionally keys only on O(1) tensor metadata.  It never hashes
the feature matrix or CSR arrays and it refuses non-contiguous inputs instead
of silently creating a native-side copy.  The producer is supplied by the
extension so the BF16 conversion order stays identical to the uncached C3
forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import os
import time
import weakref
import warnings

import torch


CACHE_CONTRACT_VERSION = "r5-hs0-bf16-v1"
CACHE_PADDED_CONTRACT_VERSION = "r5-hs1-bf16-pad-v2"
CACHE_AGGREGATE_CONTRACT_VERSION = "r5-agg0-bf16-v3"


class PersistentHsCacheError(RuntimeError):
    """Raised when a caller violates the static-cache contract."""


class PersistentCacheBudgetExceeded(PersistentHsCacheError):
    """The optional cache cannot fit its explicitly configured budget."""


def get_or_build_if_within_budget(
        cache: Any, x: torch.Tensor, graph: Any, threads: int,
        backend_module: Any) -> Optional[Tuple[torch.Tensor, bool]]:
    """Build an optional static cache, or fall back without hiding errors.

    Only a declared capacity limit is recoverable.  Invalid tensors, graphs,
    or native cache output remain hard failures so the fallback cannot mask a
    contract bug.
    """

    try:
        return cache.get_or_build(x, graph, threads, backend_module)
    except PersistentCacheBudgetExceeded as exc:
        warnings.warn(f"static cache disabled for this layer: {exc}",
                      RuntimeWarning, stacklevel=2)
        return None


def _tensor_signature(tensor: torch.Tensor) -> Tuple[Any, ...]:
    """Return an O(1) identity/version signature for a tensor."""

    return (
        tensor.device.type,
        tensor.device.index,
        str(tensor.dtype),
        tuple(int(v) for v in tensor.size()),
        tuple(int(v) for v in tensor.stride()),
        int(tensor.storage_offset()),
        int(tensor.data_ptr()),
        int(getattr(tensor, "_version", 0)),
    )


def _graph_signature(graph: Any) -> Tuple[Any, ...]:
    """Capture graph/scale identity without scanning graph contents."""

    try:
        rowptr = graph.rowptr
        colidx = graph.colidx
        scale = graph.scale
    except AttributeError as exc:  # pragma: no cover - defensive contract gate
        raise PersistentHsCacheError(
            "graph must expose rowptr, colidx and scale") from exc
    generation = getattr(graph, "cache_generation", None)
    return (
        generation,
        _tensor_signature(rowptr),
        _tensor_signature(colidx),
        _tensor_signature(scale),
    )


@dataclass(frozen=True)
class PersistentHsMetadata:
    contract_version: str
    x_signature: Tuple[Any, ...]
    graph_signature: Tuple[Any, ...]
    scale_signature: Tuple[Any, ...]
    threads: int
    hs_shape: Tuple[int, int]
    hs_stride: Tuple[int, int]
    hs_dtype: str
    hs_replica_shape: Tuple[int, ...] = ()


class PersistentHsCache:
    """A one-entry static cache suitable for one fixed graph/model input.

    A cache miss calls the native producer exactly once.  For the optional
    padded V2 contract the cache retains two views of the same values: an
    AMX-aligned physical buffer for backward and a compact logical-width
    buffer for sparse forward.  This avoids paying a strided sparse-pull
    penalty while still making the backward Hs copy-free.  Changing x,
    scale, graph storage, generation, or thread count invalidates the entry
    automatically.
    """

    def __init__(self, max_bytes: Optional[int] = None,
                 pad_to: Optional[int] = None) -> None:
        self._hs: Optional[torch.Tensor] = None
        self._hs_forward: Optional[torch.Tensor] = None
        self._hs_replicas: Optional[torch.Tensor] = None
        self._metadata: Optional[PersistentHsMetadata] = None
        # Metadata alone cannot rule out allocator ABA: a new tensor can
        # eventually reuse an old data_ptr and start at version zero.  Keep a
        # non-owning identity witness so a hit additionally requires the
        # original Python tensor object to still be the caller's object.
        self._source_ref: Optional[weakref.ReferenceType[torch.Tensor]] = None
        if pad_to is not None and int(pad_to) <= 0:
            raise PersistentHsCacheError("pad_to must be positive when set")
        self.pad_to = None if pad_to is None else int(pad_to)
        self.contract_version = (CACHE_PADDED_CONTRACT_VERSION
                                 if self.pad_to is not None else
                                 CACHE_CONTRACT_VERSION)
        if max_bytes is None:
            configured = os.environ.get("TFS_HS_CACHE_MAX_BYTES")
            max_bytes = int(configured) if configured else None
        if max_bytes is not None and int(max_bytes) < 0:
            raise PersistentHsCacheError("max_bytes must be non-negative")
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self.hits = 0
        self.misses = 0
        self.build_calls = 0
        self.build_ms = 0.0
        self.lookup_ms = 0.0

    @property
    def metadata(self) -> Optional[PersistentHsMetadata]:
        return self._metadata

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        return self._hs

    @property
    def forward_tensor(self) -> Optional[torch.Tensor]:
        """Compact logical-width BF16 view used by the sparse forward."""

        return self._hs_forward

    @property
    def replica_tensor(self) -> Optional[torch.Tensor]:
        """Optional ``[numa_replicas, N, K]`` BF16 storage.

        The ordinary cache contract remains a two-dimensional Hs tensor.  A
        replica set is built only when ``TFS_HS_REPLICA=on`` (or a sufficiently
        large ``auto`` candidate) and is consumed by the native replicated
        Hs entry point; callers can therefore keep the standard path unchanged.
        """

        return self._hs_replicas

    @property
    def bytes(self) -> int:
        """Bytes retained by the cache, including distinct V2 views.

        For an unpadded cache the compact forward view is deliberately the
        exact same tensor object as the aligned backward storage.  Counting
        both references would falsely double the budget (and made a 1M x
        1024 IGB Hs cache look like 4.096 GB instead of 2.048 GB).  Padded
        V2 views and NUMA replicas are separate allocations and remain fully
        accounted for.
        """

        total = 0
        seen = set()
        for tensor in (self._hs, self._hs_forward, self._hs_replicas):
            if tensor is None or id(tensor) in seen:
                continue
            seen.add(id(tensor))
            total += int(tensor.numel() * tensor.element_size())
        return total

    def clear(self) -> None:
        self._hs = None
        self._hs_forward = None
        self._hs_replicas = None
        self._metadata = None
        self._source_ref = None

    @staticmethod
    def _replica_requested(n: int, k: int, threads: int) -> bool:
        mode = os.environ.get("TFS_HS_REPLICA", "off").strip().lower()
        if mode in ("", "off", "0", "false"):
            return False
        if mode in ("on", "1", "true"):
            return True
        if mode == "auto":
            # The native producer still collapses to one copy on a single
            # NUMA domain.  Keep auto conservative so small fixtures do not
            # pay a second allocation/copy merely to discover that fact.
            return int(threads) >= 4 and int(n) * int(k) >= (1 << 20)
        raise PersistentHsCacheError(
            "TFS_HS_REPLICA must be off|auto|on")

    @staticmethod
    def _validate_inputs(x: torch.Tensor, graph: Any, threads: int) -> torch.Tensor:
        if x.device.type != "cpu" or x.dtype != torch.float32 or x.dim() != 2:
            raise PersistentHsCacheError(
                "static Hs cache requires CPU FP32 rank-2 x")
        if x.requires_grad:
            raise PersistentHsCacheError(
                "static Hs cache requires x.requires_grad=False")
        if not x.is_contiguous():
            raise PersistentHsCacheError(
                "static Hs cache requires contiguous x; refusing implicit copy")
        if int(threads) < 1 or int(threads) > 32:
            raise PersistentHsCacheError("threads must be in [1, 32]")
        try:
            rowptr = graph.rowptr
            colidx = graph.colidx
            scale = graph.scale
        except AttributeError as exc:
            raise PersistentHsCacheError(
                "graph must expose rowptr, colidx and scale") from exc
        if (rowptr.device.type != "cpu" or rowptr.dtype != torch.long or
                rowptr.dim() != 1 or not rowptr.is_contiguous() or
                rowptr.numel() != x.shape[0] + 1):
            raise PersistentHsCacheError(
                "static Hs cache requires contiguous CPU int64 graph.rowptr")
        if (colidx.device.type != "cpu" or colidx.dtype != torch.long or
                colidx.dim() != 1 or not colidx.is_contiguous()):
            raise PersistentHsCacheError(
                "static Hs cache requires contiguous CPU int64 graph.colidx")
        if (scale.device.type != "cpu" or scale.dtype != torch.float32 or
                scale.dim() != 1 or not scale.is_contiguous() or
                scale.numel() != x.shape[0]):
            raise PersistentHsCacheError(
                "static Hs cache requires contiguous CPU FP32 graph.scale")
        return scale

    def get_or_build(self, x: torch.Tensor, graph: Any, threads: int,
                     backend_module: Any) -> Tuple[torch.Tensor, bool]:
        """Return ``(Hs, hit)`` and enforce the cache contract."""

        lookup_t0 = time.perf_counter()
        scale = self._validate_inputs(x, graph, int(threads))
        logical_k = int(x.shape[1])
        physical_k = logical_k if self.pad_to is None else max(logical_k, self.pad_to)
        if physical_k != logical_k and physical_k % 64 != 0:
            raise PersistentHsCacheError(
                "padded Hs width must be 64-aligned")
        # V2 deliberately retains both the aligned backward buffer and the
        # compact forward buffer.  Count both against the explicit budget so
        # the memory contract cannot silently under-report RSS.
        expected_bytes = int(x.shape[0]) * (physical_k + logical_k) * 2
        if physical_k == logical_k:
            expected_bytes = int(x.shape[0]) * physical_k * 2
        if self.max_bytes is not None and expected_bytes > self.max_bytes:
            raise PersistentCacheBudgetExceeded(
                f"Hs cache budget exceeded: need {expected_bytes} bytes, "
                f"limit is {self.max_bytes}")
        x_sig = _tensor_signature(x)
        graph_sig = _graph_signature(graph)
        scale_sig = _tensor_signature(scale)
        expected = (
            x_sig, graph_sig, scale_sig, int(threads), physical_k,
        )
        replica_requested = self._replica_requested(
            int(x.shape[0]), logical_k, int(threads))
        if (self._metadata is not None and self._hs is not None and
                self._hs_forward is not None and
                (not replica_requested or self._hs_replicas is not None) and
                self._source_ref is not None and self._source_ref() is x):
            current = (
                self._metadata.x_signature,
                self._metadata.graph_signature,
                self._metadata.scale_signature,
                self._metadata.threads,
                int(self._metadata.hs_shape[1]),
            )
            if current == expected:
                self.hits += 1
                self.lookup_ms += (time.perf_counter() - lookup_t0) * 1.0e3
                return self._hs, True

        build_t0 = time.perf_counter()
        with torch.no_grad():
            if physical_k == logical_k:
                hs = backend_module.c3_prepare_static_hs_v1(
                    x, scale, int(threads))
            else:
                producer = getattr(
                    backend_module, "c3_prepare_static_hs_padded_v2", None)
                if producer is None:
                    raise PersistentHsCacheError(
                        "padded Hs producer is unavailable in the extension")
                hs = producer(x, scale, physical_k, int(threads))
        if (hs.device.type != "cpu" or hs.dtype != torch.bfloat16 or
                hs.shape != (x.shape[0], physical_k) or
                not hs.is_contiguous()):
            raise PersistentHsCacheError(
                "native Hs producer returned a non-canonical tensor")
        # Keep the sparse forward on the compact logical stride.  A padded
        # row stride is useful to the dW AMX packer, but it makes the CSR
        # pull read substantially more cache-line traffic for K tails such
        # as Products K=100.  Build this view once on the cache miss rather
        # than copying it on every training iteration.
        hs_forward = (hs if physical_k == logical_k else
                      hs[:, :logical_k].contiguous())
        hs_replicas = None
        if replica_requested:
            producer = getattr(backend_module,
                               "c3_replicate_static_hs_numa_v1", None)
            if producer is None:
                raise PersistentHsCacheError(
                    "NUMA Hs replica producer is unavailable in the extension")
            with torch.no_grad():
                hs_replicas = producer(hs_forward, int(threads))
            if (hs_replicas.device.type != "cpu" or
                    hs_replicas.dtype != torch.bfloat16 or
                    hs_replicas.dim() != 3 or
                    hs_replicas.shape[1:] != hs_forward.shape or
                    not hs_replicas.is_contiguous()):
                raise PersistentHsCacheError(
                    "native Hs replica producer returned a non-canonical tensor")
        self._hs = hs
        self._hs_forward = hs_forward
        self._hs_replicas = hs_replicas
        self._source_ref = weakref.ref(x)
        if self.max_bytes is not None and self.bytes > self.max_bytes:
            actual_bytes = self.bytes
            self.clear()
            raise PersistentCacheBudgetExceeded(
                f"Hs cache budget exceeded after NUMA replicas: need {actual_bytes} bytes, "
                f"limit is {self.max_bytes}")
        self._metadata = PersistentHsMetadata(
            self.contract_version, x_sig, graph_sig, scale_sig,
            int(threads), tuple(int(v) for v in hs.shape),
            tuple(int(v) for v in hs.stride()), str(hs.dtype),
            tuple(int(v) for v in hs_replicas.shape)
            if hs_replicas is not None else ())
        self.misses += 1
        self.build_calls += 1
        self.build_ms += (time.perf_counter() - build_t0) * 1.0e3
        return hs, False


class PersistentAggregateCache:
    """Optional layer-0 aggregate cache for aggregate-first layers.

    The cached tensor is ``T0 = B * Q_BF16(S * X)``.  It is valid only when
    the layer input is static and does not require dX; the changing weight is
    still multiplied into T0 on every forward.  This class is intentionally
    separate from :class:`PersistentHsCache` so the V1/V2 Hs contract and
    its byte accounting remain unchanged when V3 is disabled.
    """

    def __init__(self, max_bytes: Optional[int] = None) -> None:
        self._t0: Optional[torch.Tensor] = None
        self._metadata: Optional[PersistentHsMetadata] = None
        self._source_ref: Optional[weakref.ReferenceType[torch.Tensor]] = None
        self.contract_version = CACHE_AGGREGATE_CONTRACT_VERSION
        if max_bytes is None:
            configured = os.environ.get("TFS_AGG_CACHE_MAX_BYTES")
            if configured is None:
                configured = os.environ.get("TFS_HS_CACHE_MAX_BYTES")
            max_bytes = int(configured) if configured else None
        if max_bytes is not None and int(max_bytes) < 0:
            raise PersistentHsCacheError("max_bytes must be non-negative")
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self.hits = 0
        self.misses = 0
        self.build_calls = 0
        self.build_ms = 0.0
        self.lookup_ms = 0.0

    @property
    def metadata(self) -> Optional[PersistentHsMetadata]:
        return self._metadata

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        return self._t0

    @property
    def forward_tensor(self) -> Optional[torch.Tensor]:
        return self._t0

    @property
    def bytes(self) -> int:
        return (0 if self._t0 is None else
                int(self._t0.numel() * self._t0.element_size()))

    def clear(self) -> None:
        self._t0 = None
        self._metadata = None
        self._source_ref = None

    def get_or_build(self, x: torch.Tensor, graph: Any, threads: int,
                     backend_module: Any) -> Tuple[torch.Tensor, bool]:
        """Return ``(T0, hit)`` with O(1) invalidation and no dX ambiguity."""

        if x.requires_grad:
            raise PersistentHsCacheError(
                "static aggregate cache requires x.requires_grad=False")
        lookup_t0 = time.perf_counter()
        scale = PersistentHsCache._validate_inputs(x, graph, int(threads))
        logical_k = int(x.shape[1])
        expected_bytes = int(x.shape[0]) * logical_k * 2
        if self.max_bytes is not None and expected_bytes > self.max_bytes:
            raise PersistentCacheBudgetExceeded(
                f"aggregate cache budget exceeded: need {expected_bytes} bytes, "
                f"limit is {self.max_bytes}")
        x_sig = _tensor_signature(x)
        graph_sig = _graph_signature(graph)
        scale_sig = _tensor_signature(scale)
        expected = (x_sig, graph_sig, scale_sig, int(threads), logical_k)
        if (self._metadata is not None and self._t0 is not None and
                self._source_ref is not None and self._source_ref() is x):
            current = (
                self._metadata.x_signature,
                self._metadata.graph_signature,
                self._metadata.scale_signature,
                self._metadata.threads,
                int(self._metadata.hs_shape[1]),
            )
            if current == expected:
                self.hits += 1
                self.lookup_ms += (time.perf_counter() - lookup_t0) * 1.0e3
                return self._t0, True

        build_t0 = time.perf_counter()
        with torch.no_grad():
            producer = getattr(backend_module,
                               "c3_prepare_static_aggregate_v3", None)
            if producer is None:
                raise PersistentHsCacheError(
                    "static aggregate producer is unavailable in the extension")
            t0 = producer(x, scale, graph.rowptr, graph.colidx, int(threads))
        if (t0.device.type != "cpu" or t0.dtype != torch.bfloat16 or
                t0.shape != (x.shape[0], logical_k) or
                not t0.is_contiguous()):
            raise PersistentHsCacheError(
                "native aggregate producer returned a non-canonical tensor")
        self._t0 = t0
        self._source_ref = weakref.ref(x)
        self._metadata = PersistentHsMetadata(
            self.contract_version, x_sig, graph_sig, scale_sig,
            int(threads), tuple(int(v) for v in t0.shape),
            tuple(int(v) for v in t0.stride()), str(t0.dtype))
        self.misses += 1
        self.build_calls += 1
        self.build_ms += (time.perf_counter() - build_t0) * 1.0e3
        return t0, False
