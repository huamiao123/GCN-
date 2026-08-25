from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "csrc" / "experiments" / "backward_opt_20260814" / "v6_wide_aggregate_probe.cpp"


def test_small_d_gradient_production_follows_panel_owner_contract():
    source = SOURCE.read_text(encoding="utf-8")
    begin = source.index("std::vector<at::Tensor> c3_backward_amx_v2(")
    end = source.index("auto result=c3_backward_amx_v2(", begin)
    c3_backward = source[begin:end]
    # The same panel ownership drives sparse pull below, so every producer
    # branch must use it too; contiguous tid stripes create remote pages.
    assert c3_backward.count("for(int p:ws.own[tid])") >= 3
    assert "static_cast<std::int64_t>(n)*tid/threads" not in c3_backward
