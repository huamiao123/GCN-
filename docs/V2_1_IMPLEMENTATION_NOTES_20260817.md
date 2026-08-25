# final_v2.1 实施记录

本目录基于 `final_v2` 独立复制，`final_v2` 保持冻结。v2.1 先落实源码问题修改计划中不会改变数学语义的部分，再进入远程扩展重编译和端到端 gate。

## 已实施

1. `scripts/tfs_standard_env.sh` 增加 `TFS_RELEASE_PROFILE=v2_highd_native`。正式 profile 显式锁定 planner、High-D stream/native gate、pull-only backward、BF16 scale、静态 Hs 和 NUMA first-touch，并生成 `TFS_RUNTIME_CONTRACT`。
2. `run_final_numa_on_dgl_igb_20260817.sh` 使用仓库内 profile，Slurm 输出指向 `final_v2_1`，TFS/DGL 分别保存启动环境和 contract；DGL 分支不继承 TFS 的 High-D/缓存开关。
3. `LayerExecutionPlan` 成为 High-D 的形状/线程/预算决策记录：实际 forward sparse width、backward sparse width、layer path、workspace strategy、d_tile、local dW bytes、native gate、fallback reason 和 NUMA hint 都写入 `TFS_PLAN`。
4. `AggregateWideAMX.backward` 使用 planner 生成的 High-D contract，再由 adapter 交给 stream implementation；不再调用另一套独立的 `choose_highd_backward_plan` 作为正式 dispatch gate。缺失 native symbol 或 native 调用异常时，release profile fail-fast，debug/ablation 仍可回退。
5. High-D workspace 改为 bounded LRU cache（默认 512 MiB，可由 `TFS_WORKSPACE_CACHE_MAX_BYTES` 调整）。缓存条目带使用时间戳和字节统计；`TFS_PROFILE_NUMA_WORKSPACE=1` 时输出命中、未命中、淘汰、总字节、local dW、scratch 和 first-touch 时间。
6. High-D workspace 的页首次触碰从创建线程移到持久 worker；local dW/db/scratch 按 worker ownership 初始化，避免 `--localalloc` 下由单一创建线程把大 workspace 错放到一个 NUMA 节点。
7. transform-first High-D 增加 D-slab stream correctness path。正式 profile 使用 shape-only `auto` gate（N>=4096、K<=1024、D>=256）；K=2048/D=513 的实测仍回到旧 AMX tile 路径，避免把当前 Python d-slab 的负优化带入权威运行。
8. aggregate High-D 在 workspace 超预算时增加 bounded native-slice adapter：按 planner 的 d_tile 调用已验证的 native AMX v1，并合并 dX/dW/db；这解决内存上限和 native 可用性，但 C++ 内部仍是每 slab 一次调用，真正单 CSR-scan 的 fused d-slab kernel仍列为后续优化。

## 尚未宣称完成

- transform-first `K>D>128` 的 native 单 CSR-scan High-D kernel；当前 planner 会明确记录 native pending，Python D-slab 只在 shape gate 通过时启用，不会伪装成 native 路径。
- native 单 CSR-scan d-slab dW kernel；当前是 bounded native-slice adapter，已经通过 `N=1024,K=1024,D=1024,threads=32` 数值 gate，但还没有把多个 slab 合并为一次 CSR traversal。
- 远程 AMX extension 需要在 `final_v2_1` checkout 重新编译；在重编译前不使用旧 v2 build 作为 v2.1 性能结论。

## 验证

- `tests/test_execution_plan.py`: 20 passed。
- Python 源码已通过 `py_compile`。
- 远程 v2.1 extension 已重编译（SHA256 记录在 `build/extension.sha256`）。
- 远程 shape/correctness gate：IGB `K=1024,D=128` 通过；aggregate `N=4096,K=128,D=2983` 的 High-D stream 通过（dx 相对误差约 0.13%、dW 约 0.22%）；transform `N=4096,K=128,D=1024` 的 D-slab gate 通过（dx 约 0.23%、dW 约 0.22%）。
- aggregate d-slab native-slice gate：`N=1024,K=1024,D=1024,threads=32` 通过（dx 约 0.56%、dW 约 0.22%）。
- `tests/test_release_profile.sh` 在 clean environment 通过，并验证外部 `TFS_HIGHD_STREAM_V1=0` 会被 release profile 覆盖；未知 profile 会 fail-fast。
- 远程完整三图/200 epoch 的 TFS-vs-native-DGL 冷启动 wall-time gate 尚未在 v2.1 上宣称完成，必须单独提交并记录结果。
