import json

import numpy as np
from scipy.sparse import csr_matrix, save_npz
import torch

from tfs_train.datasets import load_graphsaint, load_igb_homogeneous


def test_graphsaint_loader_preserves_label_modes_and_removes_diagonal(tmp_path):
    save_npz(tmp_path / "adj_full.npz", csr_matrix(([1, 1, 1], ([0, 0, 1], [0, 1, 0])), shape=(2, 2)))
    np.save(tmp_path / "feats.npy", np.array([[1., 2.], [3., 4.]]))
    (tmp_path / "class_map.json").write_text(json.dumps({"0": [1, 0], "1": [0, 1]}))
    (tmp_path / "role.json").write_text(json.dumps({"tr": [0], "va": [], "te": [1]}))
    ds = load_graphsaint(tmp_path)
    assert ds.label_mode == "multilabel"
    assert ds.x.dtype == torch.float32
    assert ds.graph.colidx.tolist() == [1, 0]
    assert ds.train_mask.tolist() == [True, False]


def test_igb_homogeneous_loader_accepts_tier_layout(tmp_path):
    root = tmp_path / "small" / "processed"
    (root / "paper").mkdir(parents=True)
    (root / "paper__cites__paper").mkdir()
    np.save(root / "paper" / "node_feat.npy", np.ones((3, 4), dtype=np.float32))
    np.save(root / "paper" / "node_label_19.npy", np.array([0, 1, 2], dtype=np.int64))
    np.save(root / "paper__cites__paper" / "edge_index.npy", np.array([[0, 1], [1, 2]]))
    ds = load_igb_homogeneous(tmp_path)
    assert ds.x.shape == (3, 4)
    assert ds.graph.colidx.numel() == 4
    assert ds.train_mask.sum() + ds.valid_mask.sum() + ds.test_mask.sum() == 3
