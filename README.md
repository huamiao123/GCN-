# TFS-Train final_pre_numa

这是 TFS 数据流、维度泛化、workspace 和实验口径收口后的唯一正式版本。论文主方法只暴露一个 runtime；`native_c3`、V3、V4、High-D stream/single-scan 等名字仅是内部候选或消融项。

## 正式边界

- planner 只根据 `N/K/D`、层位置、静态性、线程数和内存预算选择，不读取数据集名称。
- 每层只保存一个不可变的 `execution_variant`；forward、backward、cache 与日志均由它派生，dispatch 不一致立即报错。
- 静态 aggregate-first 首层使用已验证 V3；动态 `128→128` 默认恢复 `native_c3`；V4 仍是显式实验候选。
- small-D、wide-K、aggregate High-D、transform High-D、任意尾维度和任意 `L>=2` 均有合法路径；未经 non-regression 验证的 single-scan/native 候选不会被 auto 选择。
- common/High-D workspace 都按真实数据流条件分配并使用有界 LRU；`direct_hs` 不分配 `Hb`，`compute_dx=false` 不分配 `Wt/packedWt`。
- `TFS_NUMA_FIRST_TOUCH=on` 与外部 `localalloc` 是当前基础内存初始化契约；`TFS_NUMA_PRIVATE/REDUCE=off`，locality scheduler、Hs replica 和 NUMA ownership 均关闭。

## 唯一正式入口

```bash
bash scripts/build_extension.sh
sbatch scripts/run_final_pre_numa_gate.sh
sbatch --dependency=afterok:<gate_job_id> scripts/run_final_pre_numa_authority_matrix.sh
```

正式矩阵为 Products、Arxiv、IGB-19、IGB-2983，2/3 层，1/2/4/8/16/32 线程，每个 cell 单独冷进程训练并评估 200 epoch，共 48 个 TFS cell。DGL 不在该 launcher 中重复执行，而是读取冻结的本轮 stock-DGL 结果。

每个 cell 保存：白名单环境、CPU/NUMA/affinity、源码和扩展哈希、严格 runtime contract、逐层计划、200 行训练 CSV、四段 ready 时间、峰值 RSS、冷启动 wall、精度和汇总 JSON。任何少于 200 行、计划不匹配、出现动态 V4 或来源不一致的 cell 都标为失败。

## 目录

- `python/tfs_train/`：唯一 planner、dispatch、High-D、cache 与 DGL helper。
- `csrc/`：正式 AMX 扩展源码。
- `tests/`：数值、维度、尾部、候选和 workspace gate。
- `scripts/`：构建、preflight、唯一 authority launcher 与结果验证器。
- `reports/`：gate、实现审计和最终矩阵报告。
- `runs/`、`slurm/`：运行产物；不属于源码真源。

`scripts/run_*v2_3*` 等旧脚本只用于历史复现，不能作为 authority 入口。
