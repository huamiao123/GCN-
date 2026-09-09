import torch
import torch.nn.functional as F

from tfs_train.graph import preprocess_undirected_fast
from tfs_train.supervision_scope import SupervisionScope, terminal_logits


def _reference(hidden, weight, bias, graph):
    hs = hidden * graph.scale.unsqueeze(1)
    rows = torch.repeat_interleave(
        torch.arange(hidden.shape[0], dtype=torch.int64), graph.degree)
    out = hs.clone()
    out.index_add_(0, rows, hs[graph.colidx])
    return (out @ weight) * graph.scale.unsqueeze(1) + bias


def test_scoped_terminal_matches_fp32_algebra():
    torch.manual_seed(7)
    n, k, d = 96, 32, 11
    src = torch.arange(n, dtype=torch.int64)
    edge_index = torch.stack((src, (src * 17 + 9) % n))
    graph = preprocess_undirected_fast(edge_index, n, torch.float32)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.tensor([1, 5, 11, 27, 40, 75])] = True
    hidden = torch.randn(n, k, requires_grad=True)
    weight = torch.randn(k, d, requires_grad=True)
    bias = torch.randn(d, requires_grad=True)

    class Terminal:
        threads = 1
    terminal = Terminal()
    terminal.weight, terminal.bias = weight, bias
    scope = SupervisionScope.build(mask, graph, threads=1)
    actual = terminal_logits(hidden, terminal, graph, scope)
    reference = _reference(hidden, weight, bias, graph)[scope.row_ids]
    # The shadow path intentionally follows the authority BF16 contract.
    assert torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference) < 0.03

    labels = torch.remainder(torch.arange(scope.selected_count), d)
    loss_actual = F.cross_entropy(actual, labels)
    grads_actual = torch.autograd.grad(loss_actual, (hidden, weight, bias))
    loss_reference = F.cross_entropy(reference, labels)
    grads_reference = torch.autograd.grad(loss_reference, (hidden, weight, bias))
    for actual_grad, reference_grad in zip(grads_actual, grads_reference):
        denominator = torch.linalg.vector_norm(reference_grad).clamp_min(1e-12)
        assert torch.linalg.vector_norm(actual_grad - reference_grad) / denominator < 0.05
