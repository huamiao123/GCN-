#!/usr/bin/env python3
import csv
import contextlib
import json
import os
import time
from pathlib import Path

# Freeze the Slurm/numactl CPU set before importing torch: importing the CPU
# runtime can initialize OpenMP and narrow the calling thread's affinity.
if "TFS_WORKER_CPUS" not in os.environ:
    cpus = sorted(os.sched_getaffinity(0))
    ranges = []
    start = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu != previous + 1:
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = cpu
        previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    os.environ["TFS_WORKER_CPUS"] = ",".join(ranges)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from tfs_train import load_ogbn_arxiv_raw
from tfs_train.modules import TFSConvCSR
from tfs_train.native import backend, require_non_amx_c3
from tfs_train.execution_plan import emit_plan_logs, plan_layers
from tfs_train.standard_runtime import configure_dgl
from tfs_train.standard_dgl import DGLGCN


class _C3Base(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        hs, weight, rowptr, colidx, scale, schedule = ctx.saved_tensors
        compute_dx = bool(ctx.needs_input_grad[0])
        if ctx.amx:
            dx, dw, db, _ = backend().c3_backward_amx_v2(
                grad_output.contiguous(), hs, weight, rowptr, colidx, scale,
                ctx.threads, compute_dx)
        else:
            # No non-AMX C3 kernel exists in this release.
            require_non_amx_c3("c3_backward_selective")
        if not compute_dx:
            dx = None
        return dx, dw, db, None, None, None, None, None


class AggregateFirst(_C3Base):
    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule, threads):
        if os.environ.get("HYBRID_AMX_FORWARD") == "1":
            out, hs = backend().c3_forward_amx_v2(
                x, weight, bias, rowptr, colidx, scale, int(threads), False)
        else:
            # No non-AMX C3 kernel exists in this release.
            require_non_amx_c3("c3_forward")
        ctx.save_for_backward(hs, weight, rowptr, colidx, scale, schedule)
        ctx.threads = int(threads)
        ctx.amx = os.environ.get("HYBRID_AMX_BACKWARD") == "1"
        return out


class TransformFirst(_C3Base):
    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule, threads):
        if os.environ.get("HYBRID_AMX_FORWARD") == "1":
            out, hs = backend().c3_forward_amx_v2(
                x, weight, bias, rowptr, colidx, scale, int(threads), True)
        else:
            # No non-AMX C3 kernel exists in this release.
            require_non_amx_c3("c3_forward_transform")
        ctx.save_for_backward(hs, weight, rowptr, colidx, scale, schedule)
        ctx.threads = int(threads)
        ctx.amx = os.environ.get("HYBRID_AMX_BACKWARD") == "1"
        return out


class HybridConv(torch.nn.Module):
    def __init__(self, k, d, order, threads):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(k, d))
        self.bias = torch.nn.Parameter(torch.zeros(d))
        self.order, self.threads = order, int(threads)
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, graph):
        fn = AggregateFirst if self.order == "aggregate" else TransformFirst
        return fn.apply(x, self.weight, self.bias, graph.rowptr, graph.colidx,
                        graph.scale.to(x.dtype), graph.schedule, self.threads)


class HybridGCN(torch.nn.Module):
    def __init__(self, threads, layers=2, dropout=0.5, out_dim=40):
        super().__init__()
        if layers not in (2, 3):
            raise ValueError("layers must be 2 or 3")
        dims = [128] + [128] * (layers - 1) + [out_dim]
        self._dims = dims
        self._plans = plan_layers(1, dims, feature_static_first=True)
        self._plans_logged = False
        self.convs = torch.nn.ModuleList([
            HybridConv(dims[i], dims[i + 1],
                       self._plans[i].order,
                       threads)
            for i in range(layers)
        ])
        self.dropout = dropout

    def forward(self, x, graph):
        if not self._plans_logged:
            self._plans = plan_layers(
                int(x.shape[0]), self._dims, feature_static_first=True
            )
            emit_plan_logs(self._plans)
            self._plans_logged = True
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, graph))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, graph)


class ReferenceGCN(torch.nn.Module):
    def __init__(self, layers=2, dropout=0.5, out_dim=40):
        super().__init__()
        dims = [128] + [128] * (layers - 1) + [out_dim]
        self.convs = torch.nn.ModuleList([
            TFSConvCSR(dims[i], dims[i + 1], runtime="reference")
            for i in range(layers)
        ])
        self.dropout = dropout

    def forward(self, x, graph):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, graph))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, graph)


class PyGGCN(torch.nn.Module):
    def __init__(self, layers=2, out_dim=40):
        super().__init__()
        dims = [128] + [128] * (layers - 1) + [out_dim]
        self.convs = torch.nn.ModuleList([
            GCNConv(dims[i], dims[i + 1], add_self_loops=True,
                    normalize=True, cached=True)
            for i in range(layers)
        ])

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, edge_index)


def metrics(a, b):
    d = (a.double() - b.double()).reshape(-1)
    denom = torch.linalg.vector_norm(a.double().reshape(-1)).clamp_min(1e-30)
    return {"relative_l2": float(torch.linalg.vector_norm(d) / denom),
            "max_abs": float(d.abs().max())}


threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
OUT_DIM = int(os.environ.get("HYBRID_OUT_DIM", "40"))
HIDDEN_DIM = int(os.environ.get("HYBRID_HIDDEN_DIM", "128"))
LAYERS = int(os.environ.get("HYBRID_LAYERS", "2"))
if LAYERS not in (2, 3):
    raise ValueError("HYBRID_LAYERS must be 2 or 3")
torch.set_num_interop_threads(1)
torch.set_num_threads(threads)
if torch.get_num_interop_threads() != 1:
    raise RuntimeError("PyTorch inter-op thread count must be exactly 1")
torch.manual_seed(int(os.environ.get("HYBRID_SEED", "101")))
ds = load_ogbn_arxiv_raw(os.environ["ARXIV_ROOT"])
# pandas exposes the official CSV feature matrix with column-major strides.
# Canonicalize it once before any timed method so every framework receives the
# same row-major input and no layer repays a full N x K copy each epoch.
x, labels, graph = ds.x.contiguous(), ds.labels, ds.graph

if os.environ.get("HYBRID_CHECK") == "1":
    if os.environ.get("HYBRID_CHECK_DGL") == "1":
        import dgl
        configure_dgl(dgl, threads)
        rows = torch.repeat_interleave(torch.arange(x.shape[0]), graph.degree)
        edge_index = torch.stack((graph.colidx, rows))
        dg = dgl.add_self_loop(
            dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.shape[0]))
        ref = ReferenceGCN(LAYERS, dropout=0.0, out_dim=OUT_DIM)
        # Arxiv node features are 128-dimensional; the shared wrapper's
        # default input dimension is for Products, so pass it explicitly.
        dgl_model = DGLGCN(
            LAYERS, in_dim=128, hidden_dim=128, out_dim=OUT_DIM, norm="both"
        )
        for ref_conv, dgl_conv in zip(ref.convs, dgl_model.convs):
            dgl_conv.weight.data.copy_(ref_conv.weight.data)
            dgl_conv.bias.data.copy_(ref_conv.bias.data)
        ref.eval(); dgl_model.eval()
        xr = x.clone().requires_grad_(True)
        xd = x.clone().requires_grad_(True)
        yr = ref(xr, graph)
        yd = dgl_model(xd, (dg, None))
        lr = F.cross_entropy(yr[ds.train_mask], labels[ds.train_mask])
        ld = F.cross_entropy(yd[ds.train_mask], labels[ds.train_mask])
        lr.backward(); ld.backward()
        result = {"framework": "dgl_stock", "reference": "tfs_reference_csr",
                  "loss_abs": abs(float(lr) - float(ld)),
                  "logits": metrics(yr, yd), "input_grad": metrics(xr.grad, xd.grad),
                  "parameter_grads": {}}
        for i, (ref_conv, dgl_conv) in enumerate(zip(ref.convs, dgl_model.convs)):
            result["parameter_grads"][f"convs.{i}.weight"] = metrics(
                ref_conv.weight.grad, dgl_conv.weight.grad)
            result["parameter_grads"][f"convs.{i}.bias"] = metrics(
                ref_conv.bias.grad, dgl_conv.bias.grad)
        grad_tol = float(os.environ.get("DGL_GRAD_TOL", "5e-4"))
        forward_tol = float(os.environ.get("DGL_FORWARD_TOL", "2e-5"))
        ok = result["logits"]["relative_l2"] < forward_tol
        ok &= result["input_grad"]["relative_l2"] < grad_tol
        ok &= all(v["relative_l2"] < grad_tol
                  for v in result["parameter_grads"].values())
        result["tolerance"] = {"forward_relative_l2": forward_tol,
                                "gradient_relative_l2": grad_tol}
        result["status"] = "pass" if ok else "fail"
        Path(os.environ["HYBRID_OUTPUT"]).write_text(
            json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        raise SystemExit(0 if ok else 3)
    ref = ReferenceGCN(LAYERS, dropout=0.0, out_dim=OUT_DIM)
    hyb = HybridGCN(threads, LAYERS, dropout=0.0, out_dim=OUT_DIM)
    hyb.load_state_dict(ref.state_dict())
    xr = x.clone().requires_grad_(True); xh = x.clone().requires_grad_(True)
    yr = ref(xr, graph); yh = hyb(xh, graph)
    lr = F.cross_entropy(yr[ds.train_mask], labels[ds.train_mask])
    lh = F.cross_entropy(yh[ds.train_mask], labels[ds.train_mask])
    lr.backward(); lh.backward()
    result = {"loss_abs": abs(float(lr)-float(lh)), "logits": metrics(yr,yh),
              "input_grad": metrics(xr.grad,xh.grad), "parameter_grads": {}}
    for (nr,pr),(nh,ph) in zip(ref.named_parameters(),hyb.named_parameters()):
        assert nr == nh
        result["parameter_grads"][nr] = metrics(pr.grad,ph.grad)
    grad_tol = 3e-2 if os.environ.get("HYBRID_AMX_BACKWARD") == "1" else 2e-5
    forward_tol = 3e-2 if os.environ.get("HYBRID_AMX_FORWARD") == "1" else 2e-5
    ok = result["logits"]["relative_l2"] < forward_tol and result["input_grad"]["relative_l2"] < grad_tol
    ok &= all(v["relative_l2"] < grad_tol for v in result["parameter_grads"].values())
    result["status"] = "pass" if ok else "fail"
    Path(os.environ["HYBRID_OUTPUT"]).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
    raise SystemExit(0 if ok else 3)

rows = torch.repeat_interleave(torch.arange(x.shape[0]), graph.degree)
edge_index = torch.stack((graph.colidx, rows))
path = os.environ["HYBRID_PATH"]
if path == "hybrid":
    model, graph_arg = HybridGCN(threads, LAYERS, out_dim=OUT_DIM), graph
elif path == "tfs_reference":
    model, graph_arg = ReferenceGCN(LAYERS, out_dim=OUT_DIM), graph
elif path == "pyg":
    model, graph_arg = PyGGCN(LAYERS, OUT_DIM), edge_index
elif path in ("dgl", "dgl_stock"):
    import dgl
    configure_dgl(dgl, threads)
    dg = dgl.add_self_loop(dgl.graph((edge_index[0],edge_index[1]),num_nodes=x.shape[0]))
    if path == "dgl":
        from dgl.nn.pytorch import EdgeWeightNorm
        ew = EdgeWeightNorm(norm="both")(dg,torch.ones(dg.num_edges(),dtype=x.dtype))
        model = DGLGCN(
            LAYERS, in_dim=128, hidden_dim=128, out_dim=OUT_DIM, norm="none"
        )
        graph_arg = (dg, ew)
    else:
        model = DGLGCN(
            LAYERS, in_dim=128, hidden_dim=128, out_dim=OUT_DIM, norm="both"
        )
        graph_arg = (dg, None)
else:
    raise ValueError(path)

requested_dtype = os.environ.get("HYBRID_DTYPE", "fp32").lower()
if requested_dtype == "bf16":
    requested_dtype = "bf16_native"
if requested_dtype not in ("fp32", "bf16_native", "bf16_mixed"):
    raise ValueError("HYBRID_DTYPE must be fp32, bf16_native, or bf16_mixed")
if requested_dtype != "fp32" and path not in ("dgl", "dgl_stock", "pyg"):
    raise ValueError("framework BF16 gates are only defined for DGL/PyG")
if requested_dtype == "bf16_native":
    x = x.to(torch.bfloat16)
    model = model.to(torch.bfloat16)
if requested_dtype != "fp32" and path in ("dgl", "dgl_stock"):
    # DGL gspmm requires node messages and edge weights to have identical
    # dtype. Autocast converts the node path but does not cast edge features.
    if graph_arg[1] is not None:
        graph_arg = (graph_arg[0], graph_arg[1].to(torch.bfloat16))
    if requested_dtype == "bf16_mixed":
        # DGL may execute aggregation before its dense transform, so autocast
        # alone leaves the first sparse input FP32. Keep activations/metadata
        # BF16 explicitly while learnable parameters and Adam remain FP32.
        x = x.to(torch.bfloat16)
        model.force_bf16_activations = True

reported_dtype = (
    "bf16_inputs_fp32_accum_fp32_master"
    if path == "hybrid" and os.environ.get("HYBRID_AMX_FORWARD") == "1"
    and os.environ.get("HYBRID_AMX_BACKWARD") == "1"
    else requested_dtype
)


def compute_context():
    if requested_dtype == "bf16_mixed":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()

# Layer hooks are enabled only in separate profile runs. They are deliberately
# absent from formal total-time runs because Python hook overhead would change
# the quantity being measured.
layer_profile = os.environ.get("HYBRID_LAYER_PROFILE") == "1"
layer_backward_ms = [0.0] * LAYERS
layer_backward_begin = [0] * LAYERS
layer_backward_end = [0] * LAYERS
hook_handles = []
if layer_profile:
    for layer_id, conv in enumerate(model.convs):
        # Module full-backward hooks omit most of layer 0 when its data input
        # does not require a gradient. Parameter hooks cover that legitimate
        # boundary without forcing an otherwise-unneeded dX computation.
        for parameter in conv.parameters():
            def parameter_done(gradient, idx=layer_id):
                layer_backward_end[idx] = max(
                    layer_backward_end[idx], time.perf_counter_ns())
                return gradient
            hook_handles.append(parameter.register_hook(parameter_done))

opt = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
warmups=int(os.environ.get("HYBRID_WARMUPS","2")); repeats=int(os.environ.get("HYBRID_REPEATS","7")); seed=int(os.environ.get("HYBRID_SEED","101"))

quality_epochs = int(os.environ.get("HYBRID_TRAIN_EPOCHS", "0"))
if quality_epochs > 0:
    quality_rows = []
    for epoch in range(1, quality_epochs + 1):
        torch.manual_seed(seed * 100000 + epoch)
        train_begin = time.perf_counter_ns()
        model.train(); opt.zero_grad(set_to_none=True)
        with compute_context():
            logits = model(x, graph_arg)
        loss = F.cross_entropy(logits.float()[ds.train_mask], labels[ds.train_mask])
        loss.backward(); opt.step()
        train_end = time.perf_counter_ns()
        eval_begin = time.perf_counter_ns()
        model.eval()
        with torch.no_grad():
            with compute_context():
                logits = model(x, graph_arg)
            logits = logits.float()
            val_loss = F.cross_entropy(logits[ds.valid_mask], labels[ds.valid_mask])
            test_loss = F.cross_entropy(logits[ds.test_mask], labels[ds.test_mask])
            val_acc = (logits[ds.valid_mask].argmax(-1) == labels[ds.valid_mask]).float().mean()
            test_acc = (logits[ds.test_mask].argmax(-1) == labels[ds.test_mask]).float().mean()
        eval_end = time.perf_counter_ns()
        quality_rows.append({
            "path": path, "layers": LAYERS, "dtype": reported_dtype,
            "threads": threads, "seed": seed, "epoch": epoch,
            "train_loss": float(loss.detach()), "val_loss": float(val_loss),
            "val_accuracy": float(val_acc), "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "train_step_ms": (train_end - train_begin) / 1e6,
            "evaluation_ms": (eval_end - eval_begin) / 1e6,
        })
    quality_out = Path(os.environ["HYBRID_OUTPUT"])
    with quality_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(quality_rows[0]))
        writer.writeheader(); writer.writerows(quality_rows)
    print(json.dumps({"status": "pass", "mode": "training_quality",
                      "path": path, "layers": LAYERS,
                      "dtype": reported_dtype, "threads": threads,
                      "seed": seed, "epochs": quality_epochs,
                      "final": quality_rows[-1], "output": str(quality_out)}))
    raise SystemExit(0)

raw=[]
for step in range(1,warmups+repeats+1):
    torch.manual_seed(seed*100000+step)
    layer_backward_ms[:] = [0.0] * LAYERS
    layer_backward_begin[:] = [0] * LAYERS
    layer_backward_end[:] = [0] * LAYERS
    begin=time.perf_counter_ns(); model.train(); opt.zero_grad(set_to_none=True); f0=time.perf_counter_ns()
    layer_forward_ms=[]; activation_ms=[]
    if layer_profile:
        value=x
        with compute_context():
            for layer_id,conv in enumerate(model.convs):
                layer_input=value
                q0=time.perf_counter_ns()
                if path in ("dgl", "dgl_stock"):
                    dg,ew=graph_arg
                    if requested_dtype == "bf16_mixed":
                        value = value.to(torch.bfloat16)
                    value=conv(dg,value,edge_weight=ew)
                else:
                    value=conv(value,graph_arg)
                q1=time.perf_counter_ns(); layer_forward_ms.append((q1-q0)/1e6)
                def layer_enter(gradient, idx=layer_id):
                    layer_backward_begin[idx] = time.perf_counter_ns()
                    return gradient
                value.register_hook(layer_enter)
                if layer_input.requires_grad:
                    def input_done(gradient, idx=layer_id):
                        layer_backward_end[idx] = max(
                            layer_backward_end[idx], time.perf_counter_ns())
                        return gradient
                    layer_input.register_hook(input_done)
                if layer_id + 1 < LAYERS:
                    a0=time.perf_counter_ns(); value=F.relu(value); value=F.dropout(value,p=0.5,training=True)
                    a1=time.perf_counter_ns(); activation_ms.append((a1-a0)/1e6)
        logits=value
    else:
        with compute_context():
            logits=model(x,graph_arg)
    f1=time.perf_counter_ns()
    loss=F.cross_entropy(logits.float()[ds.train_mask],labels[ds.train_mask]); f2=time.perf_counter_ns()
    loss.backward(); f3=time.perf_counter_ns(); opt.step(); end=time.perf_counter_ns()
    if layer_profile:
        for layer_id in range(LAYERS):
            if layer_backward_begin[layer_id] and layer_backward_end[layer_id]:
                layer_backward_ms[layer_id] = (
                    layer_backward_end[layer_id] -
                    layer_backward_begin[layer_id]) / 1e6
    record={"path":path,"layers":LAYERS,"dtype":reported_dtype,"threads":threads,"seed":seed,"step":step,"is_warmup":step<=warmups,
                "elapsed_ms":(end-begin)/1e6,"forward_ms":(f1-f0)/1e6,"loss_ms":(f2-f1)/1e6,
                "backward_ms":(f3-f2)/1e6,"optimizer_ms":(end-f3)/1e6,"loss":float(loss.detach())}
    if layer_profile:
        for layer_id in range(LAYERS):
            record[f"layer{layer_id}_forward_ms"] = layer_forward_ms[layer_id]
            record[f"layer{layer_id}_backward_ms"] = layer_backward_ms[layer_id]
        for activation_id in range(LAYERS - 1):
            record[f"activation{activation_id}_ms"] = activation_ms[activation_id]
    raw.append(record)
out=Path(os.environ["HYBRID_OUTPUT"])
with out.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
print(json.dumps({"status":"pass","path":path,"threads":threads,"seed":seed,"output":str(out)}))
