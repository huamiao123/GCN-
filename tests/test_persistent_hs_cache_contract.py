import os
import types
import unittest
from unittest.mock import patch

import torch

from tfs_train.persistent_hs_cache import (
    CACHE_CONTRACT_VERSION,
    CACHE_AGGREGATE_CONTRACT_VERSION,
    PersistentAggregateCache,
    PersistentHsCache,
    PersistentHsCacheError,
    get_or_build_if_within_budget,
)


class _Backend:
    def __init__(self):
        self.calls = 0
        self.replica_calls = 0

    def c3_prepare_static_hs_v1(self, x, scale, threads):
        self.calls += 1
        return (x * scale[:, None]).to(torch.bfloat16).contiguous()

    def c3_prepare_static_hs_padded_v2(self, x, scale, physical_k, threads):
        self.calls += 1
        out = torch.zeros((x.shape[0], physical_k), dtype=torch.bfloat16)
        out[:, :x.shape[1]] = (x * scale[:, None]).to(torch.bfloat16)
        return out.contiguous()

    def c3_prepare_static_aggregate_v3(self, x, scale, rowptr, colidx,
                                       threads):
        self.calls += 1
        # The fake backend only needs a stable BF16 tensor to exercise the
        # lifecycle contract; native numerical coverage is in the extension
        # smoke test.
        return (x * scale[:, None]).to(torch.bfloat16).contiguous()

    def c3_replicate_static_hs_numa_v1(self, hs, threads):
        self.replica_calls += 1
        return hs.unsqueeze(0).contiguous()


def _graph(n):
    return types.SimpleNamespace(
        rowptr=torch.arange(n + 1, dtype=torch.long),
        colidx=torch.arange(n, dtype=torch.long),
        scale=torch.ones(n, dtype=torch.float32),
    )


class PersistentHsCacheContractTest(unittest.TestCase):
    def test_hit_is_o1_and_reuses_storage(self):
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3).contiguous()
        graph = _graph(4)
        backend = _Backend()
        cache = PersistentHsCache()
        hs0, hit0 = cache.get_or_build(x, graph, 2, backend)
        hs1, hit1 = cache.get_or_build(x, graph, 2, backend)
        self.assertFalse(hit0)
        self.assertTrue(hit1)
        self.assertIs(hs0, hs1)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(cache.metadata.contract_version,
                         CACHE_CONTRACT_VERSION)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.build_calls, 1)
        self.assertGreaterEqual(cache.build_ms, 0.0)
        self.assertGreaterEqual(cache.lookup_ms, 0.0)

    def test_unpadded_forward_view_is_not_double_counted(self):
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3).contiguous()
        cache = PersistentHsCache()
        hs, _ = cache.get_or_build(x, _graph(4), 1, _Backend())
        self.assertIs(cache.forward_tensor, hs)
        self.assertEqual(cache.bytes, hs.numel() * hs.element_size())

    def test_scale_mutation_invalidates(self):
        x = torch.ones(4, 3)
        graph = _graph(4)
        backend = _Backend()
        cache = PersistentHsCache()
        cache.get_or_build(x, graph, 2, backend)
        graph.scale[0] = 2.0
        _, hit = cache.get_or_build(x, graph, 2, backend)
        self.assertFalse(hit)
        self.assertEqual(backend.calls, 2)

    def test_source_identity_witness_prevents_metadata_only_hit(self):
        x = torch.ones(4, 3)
        backend = _Backend()
        cache = PersistentHsCache()
        cache.get_or_build(x, _graph(4), 1, backend)
        other = x.clone()
        # Simulate allocator ABA: metadata can appear equal, but a cache hit
        # is legal only when the source object itself is still identical.
        cache._source_ref = lambda: other
        _, hit = cache.get_or_build(x, _graph(4), 1, backend)
        self.assertFalse(hit)
        self.assertEqual(backend.calls, 2)

    def test_noncontiguous_x_is_rejected(self):
        x = torch.ones(3, 4).t()
        self.assertFalse(x.is_contiguous())
        with self.assertRaises(PersistentHsCacheError):
            PersistentHsCache().get_or_build(x, _graph(4), 1, _Backend())

    def test_requires_grad_x_is_rejected(self):
        x = torch.ones(4, 3, requires_grad=True)
        with self.assertRaises(PersistentHsCacheError):
            PersistentHsCache().get_or_build(x, _graph(4), 1, _Backend())

    def test_explicit_byte_budget_is_checked_before_native_call(self):
        x = torch.ones(4, 3)
        backend = _Backend()
        with self.assertRaises(PersistentHsCacheError):
            PersistentHsCache(max_bytes=1).get_or_build(
                x, _graph(4), 1, backend)
        self.assertEqual(backend.calls, 0)

    def test_budget_fallback_does_not_hide_other_cache_contract_errors(self):
        backend = _Backend()
        with self.assertWarnsRegex(RuntimeWarning, "static cache disabled"):
            result = get_or_build_if_within_budget(
                PersistentHsCache(max_bytes=1), torch.ones(4, 3),
                _graph(4), 1, backend)
        self.assertIsNone(result)
        self.assertEqual(backend.calls, 0)
        with self.assertRaises(PersistentHsCacheError):
            get_or_build_if_within_budget(
                PersistentHsCache(), torch.ones(3, 4).t(), _graph(4), 1,
                backend)

    def test_noncontiguous_graph_storage_is_rejected(self):
        graph = _graph(4)
        graph.rowptr = torch.arange(10, dtype=torch.long)[::2]
        self.assertFalse(graph.rowptr.is_contiguous())
        with self.assertRaises(PersistentHsCacheError):
            PersistentHsCache().get_or_build(
                torch.ones(4, 3), graph, 1, _Backend())

    def test_padded_v2_builds_aligned_storage(self):
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3).contiguous()
        backend = _Backend()
        graph = _graph(4)
        cache = PersistentHsCache(pad_to=64)
        hs, hit = cache.get_or_build(x, graph, 1, backend)
        self.assertFalse(hit)
        self.assertEqual(tuple(hs.shape), (4, 64))
        self.assertTrue(hs.is_contiguous())
        self.assertEqual(tuple(cache.forward_tensor.shape), (4, 3))
        self.assertTrue(cache.forward_tensor.is_contiguous())
        self.assertEqual(cache.bytes,
                         hs.numel() * hs.element_size() +
                         cache.forward_tensor.numel() *
                         cache.forward_tensor.element_size())
        self.assertTrue(torch.equal(hs[:, :3],
                                    (x * graph.scale[:, None]).to(torch.bfloat16)))
        self.assertTrue(torch.equal(hs[:, 3:], torch.zeros_like(hs[:, 3:])))
        self.assertEqual(cache.metadata.hs_shape, (4, 64))
        self.assertEqual(cache.metadata.contract_version,
                         "r5-hs1-bf16-pad-v2")
        _, hit = cache.get_or_build(x, graph, 1, backend)
        self.assertTrue(hit)
        self.assertEqual(backend.calls, 1)

    def test_aggregate_v3_reuses_static_storage(self):
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3).contiguous()
        backend = _Backend()
        graph = _graph(4)
        cache = PersistentAggregateCache()
        t0, hit0 = cache.get_or_build(x, graph, 1, backend)
        t1, hit1 = cache.get_or_build(x, graph, 1, backend)
        self.assertFalse(hit0)
        self.assertTrue(hit1)
        self.assertIs(t0, t1)
        self.assertEqual(cache.metadata.contract_version,
                         CACHE_AGGREGATE_CONTRACT_VERSION)
        self.assertEqual(cache.bytes, t0.numel() * t0.element_size())
        self.assertEqual(backend.calls, 1)

    def test_aggregate_cache_requires_same_source_object(self):
        x = torch.ones(4, 3)
        backend = _Backend()
        cache = PersistentAggregateCache()
        cache.get_or_build(x, _graph(4), 1, backend)
        other = x.clone()
        cache._source_ref = lambda: other
        _, hit = cache.get_or_build(x, _graph(4), 1, backend)
        self.assertFalse(hit)
        self.assertEqual(backend.calls, 2)

    def test_opt_in_hs_replica_is_cached_and_budgeted(self):
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3).contiguous()
        backend = _Backend()
        graph = _graph(4)
        with patch.dict(os.environ, {"TFS_HS_REPLICA": "on"}, clear=False):
            cache = PersistentHsCache()
            hs, hit0 = cache.get_or_build(x, graph, 1, backend)
            replica, hit1 = cache.get_or_build(x, graph, 1, backend)
        self.assertFalse(hit0)
        self.assertTrue(hit1)
        self.assertIsNotNone(cache.replica_tensor)
        self.assertEqual(tuple(cache.replica_tensor.shape), (1, 4, 3))
        self.assertIs(replica, hs)
        self.assertEqual(backend.replica_calls, 1)
        self.assertEqual(cache.metadata.hs_replica_shape, (1, 4, 3))

    def test_replica_budget_fallback_releases_partial_cache(self):
        x = torch.ones(4, 3)
        cache = PersistentHsCache(max_bytes=30)
        with patch.dict(os.environ, {"TFS_HS_REPLICA": "on"}, clear=False):
            with self.assertWarnsRegex(RuntimeWarning, "static cache disabled"):
                result = get_or_build_if_within_budget(
                    cache, x, _graph(4), 1, _Backend())
        self.assertIsNone(result)
        self.assertEqual(cache.bytes, 0)
        self.assertIsNone(cache.metadata)


if __name__ == "__main__":
    unittest.main()
