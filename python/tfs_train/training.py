from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F

from .modules import TFSConvCSR


def pack_bf16_vnni(matrix):
    """Pack a KxN matrix as AMX-B blocks [Kb,Nb,Kpair,Ncol,Klane]."""
    if matrix.ndim != 2:
        raise ValueError("pack_bf16_vnni expects a matrix")
    k, n = matrix.shape
    kp, np = (k + 31) // 32 * 32, (n + 15) // 16 * 16
    padded = torch.zeros((kp, np), dtype=torch.bfloat16,
                         device=matrix.device)
    padded[:k, :n] = matrix.detach().to(torch.bfloat16)
    # For each 32x16 B tile, adjacent BF16 K lanes form one 32-bit VNNI
    # element per output column, matching AMX BF16 dot-product consumption.
    return (padded.view(kp // 32, 32, np // 16, 16)
            .permute(0, 2, 1, 3)
            .reshape(kp // 32, np // 16, 16, 2, 16)
            .permute(0, 1, 2, 4, 3).contiguous())


def unpack_bf16_vnni(packed):
    if packed.ndim != 5 or packed.shape[-3:] != (16, 16, 2):
        raise ValueError("invalid AMX BF16 VNNI packed shape")
    kb, nb = packed.shape[:2]
    return (packed.permute(0, 1, 2, 4, 3)
            .reshape(kb, nb, 32, 16)
            .permute(0, 2, 1, 3)
            .reshape(kb * 32, nb * 16).contiguous())


@dataclass
class StepResult:
    loss: float
    elapsed_ms: float
    logits: torch.Tensor
    packing_ms: float


class TwoLayerGCN(torch.nn.Module):
    """Standard full-batch two-layer GCN; PyTorch owns activation/dropout."""

    def __init__(self, in_features, hidden_features, out_features,
                 dropout=0.5, input_dropout=0.0, runtime="reference",
                 threads=1, private_dw=False, safe_thread_cap=1):
        super().__init__()
        self.conv0 = TFSConvCSR(
            in_features, hidden_features, runtime=runtime, threads=threads,
            private_dw=private_dw, safe_thread_cap=safe_thread_cap)
        self.conv1 = TFSConvCSR(
            hidden_features, out_features, runtime=runtime, threads=threads,
            private_dw=private_dw, safe_thread_cap=safe_thread_cap)
        self.dropout = float(dropout)
        self.input_dropout = float(input_dropout)
        self._packed_versions = (-1, -1)
        self._weight_staging = None

    def forward(self, x, graph):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.conv0(x, graph)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv1(x, graph)

    def refresh_weight_staging(self):
        """Refresh BF16 W/W^T staging after optimizer update.

        The C4 reference runtime still consumes FP32 weights. These tensors make
        weight-version and end-to-end staging costs explicit without claiming
        that the future AMX packed layout is already active.
        """
        versions = (self.conv0.weight._version, self.conv1.weight._version)
        if versions != self._packed_versions:
            self._weight_staging = tuple(
                {
                    "w_bf16": layer.weight.detach().to(
                        torch.bfloat16).contiguous(),
                    "wt_bf16": layer.weight.detach().t().to(
                        torch.bfloat16).contiguous(),
                    "pack_w": pack_bf16_vnni(layer.weight),
                    "pack_wt": pack_bf16_vnni(layer.weight.t()),
                }
                for layer in (self.conv0, self.conv1))
            self._packed_versions = versions
        return versions

    def packing_status(self):
        return {
            "versions": list(self._packed_versions),
            "representations_per_layer": (
                0 if self._weight_staging is None else 4),
            "packed_shapes": ([] if self._weight_staging is None else [
                {"pack_w": list(q["pack_w"].shape),
                 "pack_wt": list(q["pack_wt"].shape)}
                for q in self._weight_staging]),
            "layout": "AMX BF16 B: 32x16 tiles, VNNI adjacent-K pairs; distinct mathematical W and W^T",
        }


def train_step(model, graph, x, labels, train_mask, optimizer):
    begin = time.perf_counter_ns()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x, graph)
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    pack_begin = time.perf_counter_ns()
    model.refresh_weight_staging()
    pack_end = time.perf_counter_ns()
    end = time.perf_counter_ns()
    return StepResult(float(loss.detach()), (end - begin) / 1e6,
                      logits.detach(), (pack_end - pack_begin) / 1e6)


@torch.no_grad()
def evaluate(model, graph, x, labels, mask):
    model.eval()
    logits = model(x, graph)
    pred = logits.argmax(dim=-1)
    return {
        "loss": float(F.cross_entropy(logits[mask], labels[mask])),
        "accuracy": float((pred[mask] == labels[mask]).float().mean()),
    }


def mask_sparsity(graph, active_rows):
    active_rows = torch.as_tensor(active_rows, dtype=torch.bool)
    expanded = active_rows.clone()
    rows = torch.repeat_interleave(
        torch.arange(active_rows.numel()), graph.degree)
    active_edges = active_rows[rows]
    expanded[graph.colidx[active_edges]] = True
    touched_edges = int(graph.degree[active_rows].sum())
    expanded_edges = int(graph.degree[expanded].sum())
    n = active_rows.numel()
    return {
        "active_rows": int(active_rows.sum()),
        "active_ratio": float(active_rows.float().mean()),
        "one_hop_rows": int(expanded.sum()),
        "one_hop_ratio": float(expanded.float().mean()),
        "active_touched_edges": touched_edges,
        "one_hop_touched_edges": expanded_edges,
        "total_edges": int(graph.colidx.numel()),
        "default_pruning_enabled": False,
    }


def degree_bucket_bf16_ybar_error(graph, grad):
    """Diagnostic BF16-input/FP32-accumulation ybar error by degree bucket."""
    scale = graph.scale.to(grad.dtype)
    gs = scale[:, None] * grad
    ref = gs.clone()
    approx_source = gs.to(torch.bfloat16).float()
    approx = approx_source.clone()
    rows = torch.repeat_interleave(
        torch.arange(grad.shape[0]), graph.degree)
    ref.index_add_(0, rows, gs[graph.colidx])
    approx.index_add_(0, rows, approx_source[graph.colidx])
    buckets = [(0, 4), (5, 16), (17, 64), (65, None)]
    result = []
    for low, high in buckets:
        mask = graph.degree >= low
        if high is not None:
            mask &= graph.degree <= high
        if not bool(mask.any()):
            continue
        denom = torch.linalg.vector_norm(ref[mask]).clamp_min(1e-30)
        result.append({
            "degree_low": low,
            "degree_high": high,
            "rows": int(mask.sum()),
            "relative_error": float(
                torch.linalg.vector_norm(approx[mask] - ref[mask]) / denom),
            "max_abs_error": float((approx[mask] - ref[mask]).abs().max()),
        })
    return result
