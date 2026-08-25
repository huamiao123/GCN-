import torch
from .autograd import TFSConvFunction
from .native import TFSConvCSRFunction, TFSConvCSRParallelFunction


class TFSConv(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, a_off):
        return TFSConvFunction.apply(x, self.weight, self.bias, a_off)


class TFSConvCSR(torch.nn.Module):
    def __init__(self, in_features, out_features, runtime="reference",
                 threads=1, private_dw=False, threads_per_numa=8,
                 safe_thread_cap=1):
        super().__init__()
        if runtime not in ("reference", "parallel", "parallel_raw"):
            raise ValueError(
                "runtime must be reference, parallel, or parallel_raw")
        if int(threads) < 1 or int(safe_thread_cap) < 1:
            raise ValueError("threads and safe_thread_cap must be positive")
        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.runtime = runtime
        self.requested_threads = int(threads)
        self.safe_thread_cap = int(safe_thread_cap)
        self.effective_threads = (
            min(self.requested_threads, self.safe_thread_cap)
            if runtime == "parallel" else self.requested_threads)
        # Compatibility alias: this is always the count passed to the backend.
        self.threads = self.effective_threads
        self.private_dw = bool(private_dw)
        self.threads_per_numa = int(threads_per_numa)
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, graph):
        if self.runtime in ("parallel", "parallel_raw"):
            return TFSConvCSRParallelFunction.apply(
                x, self.weight, self.bias, graph.rowptr, graph.colidx,
                graph.scale.to(x.dtype), graph.schedule, self.threads,
                self.private_dw, self.threads_per_numa)
        return TFSConvCSRFunction.apply(x, self.weight, self.bias, graph.rowptr, graph.colidx, graph.scale.to(x.dtype))
