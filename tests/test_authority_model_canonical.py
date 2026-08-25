from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = (
    ROOT / "tests" / "hybrid_fullstep_products_detailed.py",
    ROOT / "tests" / "hybrid_fullstep_arxiv_detailed.py",
    ROOT / "tests" / "hybrid_aggregatewide_bf16grad_strongdgl_v3.py",
)


def test_real_graph_authority_runners_bind_canonical_model():
    """Dataset runners may retain comparison fixtures, never production models."""
    for runner in RUNNERS:
        source = runner.read_text(encoding="utf-8")
        assert "from tfs_train.authority_model import HybridConv as _CanonicalHybridConv" in source
        assert "from tfs_train.authority_model import HybridGCN as _CanonicalHybridGCN" in source
        assert "HybridConv = _CanonicalHybridConv" in source
        assert "HybridGCN = _CanonicalHybridGCN" in source
        assert "class HybridConv(" not in source
        assert "class HybridGCN(" not in source
