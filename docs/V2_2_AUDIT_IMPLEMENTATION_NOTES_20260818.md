# final_v2_2 第二轮审计实施记录

## 直接采用

- `v2_2_authority` release profile 对 planner、High-D tile/panel、workspace
  cache、CSR index、pull-only、NUMA first-touch 以及已知负优化开关进行硬锁定；
  污染的外部 shell 值会被覆盖，严格模式不一致时 fail-fast。
- `release_contract.json` 与 Python 生成的
  `runtime_contract_resolved.json` 分开记录，便于复现与审计。
- `LayerExecutionPlan` 增加稳定 `plan_id`、candidate 列表、selected
  implementation、实际 `d_slabs`、实际最大 slab、panel working-set 和
  dtype/sparse tensor metadata。`plan.validate()` 在构造完成时执行。
- Wide-output、wide-aggregate 和 aggregate-saved autograd wrapper 接收并
  保存模型构造时的同一份 plan；正式 `HybridConv` 不在 backward 中重建 plan。
- d-slab 尾部由 planner 预先分区。多 slab 的每段满足 `129 <= width <= T`；
  无法满足时保留单 slab，并按实际宽度收费，不再运行时把尾部合并成超预算 slab。
- full-D、native d-slab、transform stream 和 Python stream 分成独立 candidate；
  增加预算单调性测试，预算增加不会把可用 native 实现降级为 Python。
- row-panel working-set 独立受 `TFS_HIGHD_PANEL_BUDGET_BYTES` 约束。
- 正式 benchmark 将 dtype contract 与 execution plan JSON 分开记录；旧的
  `wide_output_amx_gemm` dtype 拼接已移除。
- native High-D workspace 在 `TFS_PROFILE_NUMA_WORKSPACE=1` 下增加
  `TFS_NUMA_BUFFER_PROFILE` ownership/consumer 记录；只做 instrumentation，
  不改变当前 reduction/dataflow。

## 有条件保留

- benchmark 中的旧重复 wrapper 改名为 `Legacy*`，正式路径只使用
  `python/tfs_train/dimension_dispatch.py` 的 canonical wrapper；旧实现仍保留
  仅用于 provenance，不能被正式 benchmark 调用。
- DGL cold-process wall 继续保留；Graph-ready timer 和三图 200 epoch 重新跑
  属于实验报告阶段，不把计时口径悄悄改成新的数字。

## 本轮明确不默认开启

- 真正单 CSR traversal 的 native transform High-D d-slab kernel；当前仍使用
  Python/ATen correctness stream，待 native sparse slab 和 dense AMX 分阶段 gate。
- 共享 reduction workspace 的 NUMA owner-aware 重新布局；当前仅保留既有 worker
  first-touch，避免在数据流稳定前同时改变 graph scheduling。
- 所有 `torch.matmul` 替换为手写 AMX、扩大 `K<=256` gate、图 partition/halo
  NUMA，以及 fused scale/db 等曾出现回退风险的实验开关。

## 验证

- `tests/test_execution_plan.py` + `tests/test_transform_production_gate.py`:
  34 tests pass, 1 opt-in native test skipped（含 transform production shape、
  d-slab tail、budget monotonicity、panel budget）。
- Python modules and formal benchmark pass `py_compile`。
- 远端 Linux 已重建 extension，且 `tests/test_extension_dimension_generalization.py`
  通过；无 pytest 依赖的 `tests/run_transform_production_gate.py` 也在远端
  对 `N=4103,K=1024,D=257` 完成了 forward/dX/dW/db 数值 gate。完整
  200-epoch authority 性能数字仍需单独提交 paired wall-time gate，本版本
  不继承 v2/v2.1 的旧性能数字。
