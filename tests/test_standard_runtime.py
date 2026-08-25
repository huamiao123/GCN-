import pytest

from tfs_train.standard_runtime import validate_authority_variant


def test_final_authority_accepts_only_hybrid(monkeypatch):
    monkeypatch.setenv("HYBRID_AUTHORITY_TEMPLATE", "final-pre-numa-v1")
    monkeypatch.setenv("HYBRID_DETAILED_PROFILE", "0")
    monkeypatch.setenv("HYBRID_DGL_OP_PROFILE", "0")
    validate_authority_variant("hybrid")
    with pytest.raises(RuntimeError, match="requires HYBRID_PATH=hybrid"):
        validate_authority_variant("dgl_stock")


def test_final_authority_rejects_instrumented_timing(monkeypatch):
    monkeypatch.setenv("HYBRID_AUTHORITY_TEMPLATE", "final-pre-numa-v1")
    monkeypatch.setenv("HYBRID_DETAILED_PROFILE", "1")
    with pytest.raises(RuntimeError, match="detailed profiling off"):
        validate_authority_variant("hybrid")


def test_historical_authority_keeps_stock_dgl_only(monkeypatch):
    monkeypatch.setenv("HYBRID_AUTHORITY_TEMPLATE", "r5-standard-v1")
    monkeypatch.setenv("HYBRID_DETAILED_PROFILE", "0")
    monkeypatch.setenv("HYBRID_DGL_OP_PROFILE", "0")
    validate_authority_variant("dgl_stock")
    with pytest.raises(RuntimeError, match="HYBRID_PATH=dgl_stock"):
        validate_authority_variant("dgl_cached")
