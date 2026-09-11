from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "csrc" / "experiments" / "backward_opt_20260814" /
          "v6_wide_aggregate_probe.cpp")


def test_colidx_caches_use_owned_tensor_identity_and_version():
    source = SOURCE.read_text(encoding="utf-8")
    validation = source[source.index("struct ColidxValidationEntry"):
                        source.index("inline int cpu_sysfs_int")]
    workspace = source[source.index("struct Int32IndexWorkspace"):
                       source.index("std::vector<std::vector<int>> panel_schedule")]

    for section in (validation, workspace):
        assert "at::Tensor source" in section
        assert "unsafeGetTensorImpl" in section
        assert "version_counter().current_version()" in section
        assert "kColidxCacheEntryLimit" in section

    # Raw allocation address plus length was the unsafe historical identity.
    assert "const std::int64_t* identity" not in validation
    assert "const std::int64_t* identity" not in workspace


def test_all_colidx_cache_callers_pass_the_tensor_witness():
    source = SOURCE.read_text(encoding="utf-8")
    assert "formal_colidx_enabled(glue_e9,ci," not in source
    assert "int32_colidx_workspace(ci," not in source
    assert source.count("int32_colidx_workspace(ci_t,") == 6
    assert source.count("int32_colidx_workspace(colidx,") == 2
