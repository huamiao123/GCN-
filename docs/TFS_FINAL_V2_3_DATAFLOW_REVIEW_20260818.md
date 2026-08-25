# final_v2_2 数据流与维度泛化方案审阅记录

日期：2026-08-18

## 结论

方案总体合理，可以作为下一阶段的数据流冻结计划。本轮已落地不改变数学语义、且能在当前 Python 3.9 集群验证的部分；尚未把未经 native 数值/性能 gate 的 Transform-HighD kernel 设为默认路径。

## 已落地

- `KernelCandidate` 改为兼容 Python 3.9 的 keyword-only 初始化，所有 planner 构造统一使用关键字参数。
- 修正 sparse backward metadata：
  - aggregate-first：稀疏 backward 是 `dP`，宽度为 `K/Kp`；
  - transform-first：稀疏 backward 是 `Gs -> dT`，宽度为 `D/Dp`。
- 增加 logical/physical sparse width、sparse tensor、plan hash metadata，并在 `plan.validate()` 中强制校验。
- candidate 类型、selected implementation、workspace 和 sparse-width invariant 纳入 planner 校验。
- planner 明确区分 `dense_d_tile` 与 `sparse_d_slab`；保留 `d_tile` 兼容别名，避免旧 adapter 产生第二套决策。
- release profile 明确锁定 small-single-scan、active-row、aggregate-saved、NUMA 辅助开关及 glue 开关；DGL 分支清除这些 TFS 变量，避免继承 shell 污染。
- paired authority launcher 的 Intel runtime module 改为加载失败即停止，不再静默继续。
- Transform-HighD 新增 direct-BF16 sparse pull primitive；旧 FP32-entry primitive 仍保留为扩展兼容回退。

## 有意调整文档方案的部分

1. 文档建议 `@dataclass(kw_only=True)`，但远端权威环境是 Python 3.9；该语法不可用，因此使用 `@dataclass(frozen=True, init=False)` 加手写 keyword-only initializer，实现相同约束且不改变运行环境。
2. 文档建议新增 `plan_hash`；现有 `plan_id` 已经是稳定 SHA-256 摘要，因此保留 `plan_id` 作为既有日志键，同时提供 `plan_hash` 同值别名，避免生成两套不同身份。
3. 已新增 `c3_backward_transform_highd_amx_v1`：单个 D-slab 在 C++ 内完成 BF16 scale/pull、AMX dW 和 AMX dHs；Python 只负责 planner slab 的 FP32 跨 slab 累加。该入口已通过数值 gate，但性能在小 panel/高线程数上并不单调，因此仍由 `TFS_TRANSFORM_HIGHD_NATIVE=auto|on|off` 控制，未通过 cost gate 的 authority shape 不自动切换。
4. `transform_auto` 的 N/K/D 门槛暂时保留。N 是 workspace/streaming 成本的一部分，不是数据集名称；删除该门槛前需要补齐小 N、大 K 的性能消融，避免把 stream launch overhead 扩散到所有形状。

## 验证结果

- 本地 planner / contract / transform gate 相关测试：`52 passed, 1 skipped`。
  直接运行全量 pytest 时，剩余扩展 smoke 文件在本机因缺少
  `tfs_train_v2_c0_ext` 而无法收集；这不影响上述纯 Python gate，远端扩展
  smoke 另行完成。
- 服务器 Python 3.9：`py_compile` 通过，所有修改脚本 `bash -n` 通过。
- 服务器 `N=4103,K=1024,D=257` Transform-HighD production gate：`PASS`。
- 服务器 direct-BF16 pull 数值 gate（`K=19/128/257`）：`PASS`；在 `N=4096,K=257,T=32` 微基准中，pull-entry 中位时间由约 `15.3 ms` 降至 `0.24 ms`。该结果是 kernel-entry 消融，不等同于端到端加速比。
- 服务器 aggregate High-D V4 与 native d-slab gate 均通过；`N=4096,K=128,D=2983` 的现有 stream gate backward 为约 `2.45×`，`N=1024,K=1024,D=1024` d-slab 数值 gate 通过。
- `N=4103,K=1024,D=257,T=32` 五次中位数：Transform-HighD backward 约 `54.1 ms → 19.4 ms`（`2.79×`），forward 约 `8.59 ms → 7.21 ms`；相对旧 wide-output path 的最大绝对相对误差仍在既有 gate 内。
- 新增 Transform-HighD 形状矩阵（远端 Python 3.9、AMX、4 线程、每条路径独立 warmup）通过：
  - `N=4097,K=512,D=257`：dX `0.00220`、dW `0.00249`；
  - `N=4103,K=1024,D=257`：dX `0.00203`、dW `0.00217`；
  - `N=4099,K=1024,D=513`：三段 slab，dX `0.00200`、dW `0.00333`；
  - `N=4097,K=2048,D=513`：三段 slab，dX `0.00259`、dW `0.00216`。
  四个形状的输出误差均为 `0`，db 误差小于 `7e-8`，均在 production gate 阈值内。矩阵中的耗时仅作 kernel-entry 诊断，不作为端到端加速结论。
- 额外的低内存 planner gate 通过：native High-D workspace budget 设为 `1 byte` 时，planner 选择 `python_stream`，并按 panel budget 验证，不再把 reference fallback 错误判为 native workspace 超预算。
- 远端扩展 smoke 组合通过：基础 shape matrix、IGB `1024→128`/`128→2983` dimension gate、active-row-wide、aggregate cache、aggregate-saved V4、generic single-scan、Hs NUMA replica、locality schedule、padded tail、persistent Hs 共 10 项均 `PASS`。
- 现有 native wide-output（逐 128 列 tile，包含 native sparse pull/dW/dHs）与 streamed 路径完成 `T=1/2/4/8/16/32` 线程数值 gate，四个 Transform-HighD 形状均通过；但重复中位数显示 `N≈4100,K=1024,D=513` 的多-slab streamed backward 在 `T=8/16/32` 仅约 `0.59×/0.68×/0.84×`，属于真实回退，不再允许 auto 选择。
- planner 已收紧：Transform-HighD 只有单 slab 形状才由 `auto` 选择 streamed；多 slab 仍可用 `TFS_TRANSFORM_HIGHD_STREAM_V1=on` 显式 probe。该修改保持按 N/K/D 和 slab 结构决策，不按数据集名称决策。
- 远端 `run_transform_auto_gate.py` 已验证：`D=257` auto 选 streamed，`D=513` auto 选 legacy，显式 `on` 才选 streamed。
- 新增 native Transform-HighD slab gate：`N=4097,K=1024,D=257` 单 slab 与
  `N=4097,K=1024,D=513` 三 slab 的 dX/dW relative-L2 分别约
  `0.001655/0.001653` 与 `0.001660/0.001659`；独享节点上 wide single-scan
  组件与 planner cost gate 已通过。详细的 2 warmup + 5 measured 结果见
  `reports/transform_native_slab_gate_20260818.md`。
- 新增 `TFS_TRANSFORM_HIGHD_NATIVE_GS_BUDGET_BYTES`（默认 256 MiB）并写入
  release runtime contract；native planner/adaptor 在进入 C++ 前共同拒绝
  超预算的 BF16 scaled-gradient slab。
- 追加的宽维度组件矩阵也通过：`D=1025`（5 slabs）、`K=2051` 非 64 对齐尾部、`K=4096,D=1025`，在 4/16/32 线程下输出误差均为 `0`，dX/dW 最大相对误差分别约 `0.00364/0.00307` 以内，db 小于 `1e-7`。该矩阵仍是显式 streamed probe，不能替代真实 fused native kernel 的 gate。
- 本地与服务器关键源码 hash 已核对一致。

### Post-document four-graph E2E (32 threads, 200 epochs)

The paired cold-process wall results are recorded in
`reports/postdoc_e2e_authority_20260818.md`:

| graph/config | TFS (s) | stock DGL (s) | DGL/TFS |
|---|---:|---:|---:|
| Products, 2-layer | 107.460 | 268.674 | 2.500x |
| Arxiv, 2-layer | 14.076 | 23.157 | 1.645x |
| IGB-19, 2-layer | 70.697 | 121.016 | 1.712x |
| IGB-2983, 2-layer | 383.349 | 688.720 | 1.797x |

These runs use the authority profile with `TFS_TRANSFORM_HIGHD_NATIVE=off`;
the new Transform-HighD native symbol is covered by a separate numerical and
component gate, not silently mixed into this E2E denominator.

## 尚未合并为默认路径

- native Transform-HighD 新入口已实现但尚未冻结为所有 shape 的默认路径；只有满足 panel/worker/dense-work cost gate 的维度组合才允许 `auto` 选择，其余仍显式回退到 streamed/legacy；
- dT panel immediate-consume（当前新入口仍 materialize 当前 slab 的 BF16
  Gs/dT 工作集）；
- aggregate High-D single-scan；
- NUMA ownership、halo、replica、scheduler 深度改造。

这些项目应在独立 correctness、peak-RSS 和多图 non-regression gate 通过后再进入 authority profile。
