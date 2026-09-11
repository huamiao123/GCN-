from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from tfs_train.graph import CSRGraph
from tfs_train.supervision_scope import SupervisionScope


def _fixture():
    rowptr = torch.tensor([0, 1, 2], dtype=torch.int64)
    colidx = torch.tensor([1, 0], dtype=torch.int64)
    graph = CSRGraph(
        rowptr=rowptr, colidx=colidx, scale=torch.ones(2),
        degree=torch.ones(2, dtype=torch.int64),
        schedule=torch.arange(2, dtype=torch.int64))
    scope = SupervisionScope(
        row_ids=torch.tensor([0], dtype=torch.int64),
        transpose_rowptr=torch.tensor([0, 1, 1], dtype=torch.int64),
        transpose_colidx=torch.tensor([0], dtype=torch.int64),
        selected_schedule=torch.tensor([0, 1], dtype=torch.int64),
        transpose_schedule=torch.tensor([0, 2], dtype=torch.int64),
        node_count=2, selected_edge_count=1, thread_count=1,
        _graph_rowptr=rowptr, _graph_colidx=colidx,
        _rowptr_version=int(rowptr._version),
        _colidx_version=int(colidx._version))
    return graph, scope


def test_scope_accepts_only_its_original_graph_and_thread_contract():
    graph, scope = _fixture()
    scope.validate(graph, threads=1)

    with pytest.raises(ValueError, match="different graph"):
        scope.validate(replace(graph, colidx=graph.colidx.clone()), threads=1)
    with pytest.raises(ValueError, match="thread mismatch"):
        scope.validate(graph, threads=2)


def test_scope_rejects_in_place_csr_mutation():
    graph, scope = _fixture()
    graph.colidx[0].copy_(graph.colidx[0])
    with pytest.raises(ValueError, match="CSR changed"):
        scope.validate(graph, threads=1)
