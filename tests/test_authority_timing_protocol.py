import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZE = ROOT / "scripts" / "summarize_authority_cell.py"


def test_paper_steady_metric_excludes_first_epoch(tmp_path):
    csv_path, out_path = tmp_path / "epochs.csv", tmp_path / "summary.json"
    fields = ("epoch", "train_step_ms", "evaluation_ms", "train_loss",
              "val_accuracy", "test_accuracy")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerow({"epoch": 1, "train_step_ms": 100, "evaluation_ms": 20,
                         "train_loss": 1, "val_accuracy": .5, "test_accuracy": .5})
        writer.writerow({"epoch": 2, "train_step_ms": 2, "evaluation_ms": 3,
                         "train_loss": .9, "val_accuracy": .6, "test_accuracy": .6})
    subprocess.run([sys.executable, str(SUMMARIZE), str(csv_path), str(out_path),
                    "1000", "tfs_final_pre_numa", "products", "2", "1", "2"],
                   check=True)
    result = json.loads(out_path.read_text())
    assert result["cold_process_wall_ms"] == 1000
    assert result["steady_epoch_range"] == "2-2"
    assert result["steady_median_epoch_train_plus_eval_ms"] == 5.0
    assert result["first_epoch_train_plus_eval_ms"] == 120.0
