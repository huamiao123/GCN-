# TFS supervision-scoped shadow planner 验收记录

日期：2026-09-10

## 1. 验收结论

显式shape-aware planner已经接入shadow训练入口，并通过native构建、单元测试、
真实图24-cell同节点A/B以及两组200-epoch收敛检查。它可以作为后续正式融合的
候选，但仍不替换冻结的`final_pre_numa`权威入口。

核心结论：

- 24/24个真实性能cell通过正确性门槛；
- 所有cell均为同一独占节点内authority/candidate交错A/B，不拼接跨节点时间；
- 24-cell训练步加速比几何均值为`1.5848x`；
- IGB-small D=2983的L2/L3六线程几何均值为`2.2406x/2.0329x`；
- Products D=47的L2/L3六线程几何均值为`1.2765x/1.0850x`；
- 两组200-epoch paired训练均通过loss与accuracy门槛。

## 2. 被验收的执行合同

planner输入为：

```text
(selected_rows M, hidden_dim K, output_dim D, threads)
```

输出并显式传入训练路径：

```text
row_tile
in-place log-softmax
fixed-chunk fused db
native compact dW
direct-tail transpose
native compact logits
```

`DenseFusionPlan.validate()`会在执行前核对完整形状与线程合同，防止将一个形状
生成的plan错误用于另一个形状。planner路径调用
`c3_compact_dw_bf16_amx_shadow_v2(..., direct_tail)`，不读取进程级
`TFS_COMPACT_DW_T4`。未传plan的旧入口继续保留，仅用于复现历史实验。

当前保守启用边界：

- native dW：`K=128, threads>=8, D>=512, M/D>=20`；
- direct-tail：native dW已启用且`D mod 32 != 0`；
- native logits：`K=128, threads>=8, D>=2500, M*D/threads>=500000`；
- fused db仅跟随已测得收益的native-dW区域；
- row tile为`min(M, 300000)`。

这些是实测包络内的保守边界，不是通用硬件解析cost model。

## 3. 构建和测试证据

| 项目 | 证据 |
|---|---|
| native构建 | Slurm `10387317`, `COMPLETED`, exit `0:0` |
| 扩展SHA-256 | `79385b1cee4b624264f906b4882d353b2b3c1dcb3b485c0e2990af39bdfdc402` |
| planner/native/scope gate | Slurm `10387533`, `13 passed`, exit `0:0` |
| 最终收紧fallback后的复验 | Slurm `10388041`, `15 passed`, exit `0:0` |
| 首个显式planner同节点gate | Slurm `10387552`, pass |
| 24-cell矩阵 | Slurm array `10387812`, 24/24 completed |
| 200-epoch收敛 | Slurm array `10387962`, 2/2 completed |

首个IGB L2 32T显式planner gate：

- authority：`1310.390 ms`；
- planned：`590.243 ms`；
- ratio-of-medians：`2.2201x`；
- loss绝对误差：`0`；
- dH relative L2：`8.70e-5`；
- dW relative L2：`3.85e-7`；
- db最大绝对误差：`3.37e-6`。

## 4. 24-cell真实图同节点矩阵

线程为`1/2/4/8/16/32`，层数为`2/3`，每个cell为2次warm-up后9次
交错重复。比值定义为：

```text
authority TFS median / planned shadow median
```

| 数据集 / 层数 | 1T | 2T | 4T | 8T | 16T | 32T | 几何均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IGB-small D=2983 L2 | 1.591x | 2.282x | 2.234x | 2.547x | 2.539x | 2.413x | 2.241x |
| IGB-small D=2983 L3 | 1.504x | 2.107x | 2.055x | 2.319x | 2.293x | 2.039x | 2.033x |
| Products D=47 L2 | 1.319x | 1.371x | 1.339x | 1.316x | 1.203x | 1.130x | 1.277x |
| Products D=47 L3 | 1.101x | 1.108x | 1.102x | 1.081x | 1.069x | 1.050x | 1.085x |

全矩阵最大数值差异：

- loss绝对误差：`4.53e-6`；
- dH relative L2：`2.416e-3`；
- dW relative L2：`1.698e-3`；
- db最大绝对误差：`3.088e-5`。

完整逐cell时间与误差见：

- `evidence/planned_dense_matrix_table.md`
- `evidence/planned_dense_matrix_summary.json`
- `evidence/authority_vs_planned_dense_matrix/*.json`

## 5. 200-epoch收敛验收

两臂从相同参数初始化开始，每个epoch使用相同dropout seed，并在同一节点交错
执行；evaluation不计入训练步时间。

| 数据集 | authority稳态 | planned稳态 | 加速比 | 最终loss相对差 | val acc差 | test acc差 |
|---|---:|---:|---:|---:|---:|---:|
| Products D=47 L2 32T | 308.076 ms | 270.843 ms | 1.1375x | 3.67e-5 | 5.08e-5 | 4.97e-5 |
| IGB-small D=2983 L2 32T | 1297.782 ms | 557.743 ms | 2.3268x | 2.79e-6 | 6.50e-5 | 1.40e-4 |

准确率差为绝对比例；例如`1.40e-4`等于`0.014`个百分点。原始epoch曲线和
所有周期性evaluation记录位于`evidence/planned_dense_convergence/`。

## 6. 能宣称和不能宣称的内容

### 6.1 低线程native-logits边界

Slurm array `10388006`额外测了1/2/4T，并保持framework dW，只切换native
logits。训练步速度分别改善`1.162x/1.127x/1.111x`，但三项的dH relative-L2
分别为`1.266e-4/1.245e-4/1.235e-4`，均略高于该独立A/B预设的严格
`1e-4`门槛，所以3个任务按合同全部判fail。绝对误差仍低于`9e-10`，说明这
不是崩溃或明显实现错误；但本次没有放宽门槛，planner继续在低于8T时回退
framework logits。原始负结果保存在`evidence/native_logits_low_threads/`。

### 6.2 论文边界

当前证据支持：

> loss-coupled supervision-scoped terminal dataflow在不改变隐藏层full-graph
> 训练和精确CE目标的情况下，可以消除未监督末层行与High-D宽中间张量；
> shape-aware planner可在已测包络内安全选择bounded CE与AMX dense融合路径。

当前证据不支持：

- 将该结果称为相对DGL-Mixed的新权威矩阵；本次直接基线是冻结authority TFS；
- 声称已获得适用于任意模型、任意监督率和任意CPU的通用cost model；
- 把通用panel、BF16/AMX、软件预取或负载均衡本身称为核心首创；
- 声称Q-panel NUMA push已经实现；它仍然只有设计文档。

## 7. 复现入口

```bash
sbatch slurm/build_planned_dense.sh
sbatch slurm/authority_vs_planned_dense_matrix.sh
sbatch slurm/planned_dense_convergence_32t.sh
```

汇总：

```bash
python scripts/summarize_planned_dense_matrix.py RESULTS_DIR \
  --json evidence/planned_dense_matrix_summary.json \
  --markdown evidence/planned_dense_matrix_table.md
```
