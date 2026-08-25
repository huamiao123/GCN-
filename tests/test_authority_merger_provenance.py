import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MERGER = ROOT / "scripts" / "merge_final_authority_results.py"


def _write_cell(root: Path, method: str, partition: str) -> None:
    cell = root / "products_l2_t1_e200"
    cell.mkdir(parents=True)
    is_tfs = method == "tfs_final_pre_numa"
    (cell / "status.json").write_text(json.dumps({
        "status": "success", "epochs": 200, "wall_ms": 3000,
        "method": method if not is_tfs else "tfs_final_pre_numa",
        "path": "hybrid" if is_tfs else "dgl_stock",
        **({"requested_profile": "final_pre_numa",
            "resolved_profile": "final_pre_numa"} if is_tfs else {}),
    }))
    manifest = {
        "partition": partition, "numactl": "--cpunodebind=0-3 --localalloc",
        "method": method, "path": "hybrid" if is_tfs else "dgl_stock",
        "timing_protocol": "paper_v1",
    }
    if is_tfs:
        manifest.update({"profile_status": "authority",
                         "dtype_contract": "bf16_inputs_fp32_accum_fp32_master",
                         "numa_private": "off", "numa_reduce": "off", "seed": 101})
    else:
        manifest.update({"dgl_variant": "stock", "dgl_rerun": True})
    (cell / "manifest.json").write_text(json.dumps(manifest))
    (cell / "environment_start.txt").write_text(
        "OMP_NUM_THREADS=1\nMKL_NUM_THREADS=1\nOMP_DYNAMIC=FALSE\n"
        "MKL_DYNAMIC=FALSE\nOMP_PROC_BIND=close\nOMP_PLACES=cores\n")
    for name, value in (("hostname.txt", "host-a\n"), ("lscpu.txt", "cpu-a\n"),
                        ("affinity.txt", "affinity-a\n")):
        (cell / name).write_text(value)
    fields = ("epoch", "train_step_ms", "evaluation_ms", "path", "dtype",
              "layers", "threads", "seed", "train_loss", "val_accuracy", "test_accuracy")
    with (cell / "training_detailed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for epoch in range(1, 201):
            writer.writerow({"epoch": epoch, "train_step_ms": 1, "evaluation_ms": 1,
                             "path": "hybrid" if is_tfs else "dgl_stock",
                             "dtype": "bf16_inputs_fp32_accum_fp32_master" if is_tfs else "fp32",
                             "layers": 2, "threads": 1, "seed": 101,
                             "train_loss": 1, "val_accuracy": .5, "test_accuracy": .5})
    if is_tfs:
        (cell / "plan_validation.json").write_text(json.dumps({"status": "pass", "plans": [{}, {}]}))
        (cell / "timing_markers.json").write_text(json.dumps({
            "process_marker_ns": 1, "data_ready_ns": 2, "framework_graph_ready_ns": 3,
            "model_ready_ns": 4, "training_runtime_start_ns": 5,
            "training_runtime_end_ns": 6}))
        for name in ("source_input_sha256.tsv", "resource_usage.txt"):
            (cell / name).write_text("ok\n")


def test_merger_rejects_dgl_partition_mismatch(tmp_path):
    tfs_root, dgl_root = tmp_path / "tfs", tmp_path / "dgl"
    _write_cell(tfs_root, "tfs_final_pre_numa", "intel")
    _write_cell(dgl_root, "dgl_stock", "intel_expr")
    result = subprocess.run(
        [sys.executable, str(MERGER), str(tfs_root), str(dgl_root), str(tmp_path / "out")],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "TFS/DGL provenance mismatch for partition" in result.stderr
