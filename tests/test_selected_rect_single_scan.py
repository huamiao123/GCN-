import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tfs_train.native import backend


def _cuts(rows: int, threads: int) -> torch.Tensor:
    return torch.tensor(
        [rows * thread // threads for thread in range(threads + 1)],
        dtype=torch.int64)


def _graph(n: int):
    neighbors = []
    rowptr = [0]
    for row in range(n):
        values = sorted({(row * 3 + 1) % n, (row * 5 + 7) % n,
                         (row * 11 + 9) % n} - {row})
        neighbors.extend(values)
        rowptr.append(len(neighbors))
    return (torch.tensor(rowptr, dtype=torch.int64),
            torch.tensor(neighbors, dtype=torch.int64))


def _selected_reference(hs, rows, rowptr, colidx):
    result = torch.empty((rows.numel(), hs.shape[1]), dtype=torch.bfloat16)
    for pos, row_value in enumerate(rows.tolist()):
        value = hs[row_value].float().clone()
        for edge in range(int(rowptr[row_value]), int(rowptr[row_value + 1])):
            value.add_(hs[int(colidx[edge])].float())
        result[pos].copy_(value.to(torch.bfloat16))
    return result


def _rect_reference(q, rowptr, colidx):
    result = torch.empty((rowptr.numel() - 1, q.shape[1]),
                         dtype=torch.bfloat16)
    for row in range(result.shape[0]):
        value = torch.zeros(q.shape[1], dtype=torch.float32)
        for edge in range(int(rowptr[row]), int(rowptr[row + 1])):
            value.add_(q[int(colidx[edge])].float())
        result[row].copy_(value.to(torch.bfloat16))
    return result


@pytest.mark.parametrize("k", [1, 15, 16, 17, 31, 32, 47, 64, 100, 127, 128])
@pytest.mark.parametrize("threads", [1, 4])
def test_selected_and_rectangular_pulls_visit_edges_once_without_value_drift(
        k, threads):
    torch.manual_seed(9000 + k + threads)
    n = 48
    rowptr, colidx = _graph(n)
    selected = torch.tensor([1, 3, 8, 13, 21, 34, 45], dtype=torch.int64)
    hs = torch.randn(n, k).to(torch.bfloat16).contiguous()

    actual_selected = backend().c3_selected_pull_bf16_shadow_v1(
        hs, selected, rowptr, colidx, _cuts(selected.numel(), threads),
        threads)
    torch.testing.assert_close(
        actual_selected, _selected_reference(hs, selected, rowptr, colidx),
        rtol=0, atol=0)

    rect_rowptr, rect_colidx = backend().c3_build_selected_transpose_shadow_v1(
        selected, rowptr, colidx)
    q = torch.randn(selected.numel(), k).to(torch.bfloat16).contiguous()
    rect_schedule = _cuts(n, threads)
    actual_rect = backend().c3_rect_pull_bf16_shadow_v1(
        q, rect_rowptr, rect_colidx, rect_schedule, threads)
    expected_rect = _rect_reference(q, rect_rowptr, rect_colidx)
    torch.testing.assert_close(actual_rect, expected_rect, rtol=0, atol=0)

    scale = torch.linspace(0.25, 1.25, n, dtype=torch.float32)
    actual_scaled = backend().c3_rect_pull_bf16_scaled_fp32_shadow_v1(
        q, rect_rowptr, rect_colidx, scale, rect_schedule, threads)
    torch.testing.assert_close(
        actual_scaled, expected_rect.float() * scale.unsqueeze(1),
        rtol=0, atol=0)
