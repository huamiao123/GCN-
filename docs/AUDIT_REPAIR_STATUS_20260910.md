# TFS 源码审计修复状态（2026-09-10）

基线提交：`e399d4f69132983c6cced71a4bde29590e9c2e43`

本轮只在独立 shadow 中修改，不覆盖权威源码。

## 已修复

1. **int32 CSR 缓存身份与生命周期**
   - 不再使用裸 `data_ptr + numel` 作为缓存身份。
   - 缓存持有源 tensor，以 TensorImpl 身份和 version counter 联合校验。
   - validation cache 与 conversion workspace 均限制为最多 8 项，避免无界增长。

2. **`aggregate_saved_v4` backward 元数据**
   - 实际 sparse operand 修正为 `dP`。
   - logical/physical width 修正为 `K/Kp`，不再错误报告为 `Gs/D/Dp`。

3. **SupervisionScope 图身份契约**
   - scope 记录原始 CSR tensor、版本和线程数。
   - 相同 N 的另一张图、CSR 原地修改或线程数变化均 fail-fast，要求重建 scope。

4. **V4 完整张量 BF16→FP32→BF16 往返**
   - `Gs @ W^T` 的 BF16 `dP` 直接进入 `c3_pull_only_bf16_amx_v1`。
   - 删除只为适配旧 FP32 pull 接口而产生的完整 `dP.float()`。
   - 最终 `dX` 仍保持 FP32 合同。

## 已验证但拒绝合入

尝试将 selected/rectangular pull 从 feature-block 外层改成 edge 外层，令每行 CSR
逻辑上只遍历一次。数值在 K=1..128、1/4 线程下全部精确一致，但同节点 A/B 显示
8 个并行 ZMM 累加器造成明显寄存器压力：

- selected BF16：几何均值 `1.050x`，但多线程存在回退；
- rectangular BF16：`0.846x`；
- rectangular scaled FP32：`0.670x`。

因此该实现已撤回。不能把“逻辑单扫”直接等同于更快；后续若继续，应使用编译期
特化、较小 feature group 或两级 reducer，并重新过性能 gate。

原始 A/B 数据位于 `results/selected_rect_single_scan/`。

## 验收

- 本地 planner/contract：66 tests passed。
- 服务器 AMX 扩展：Intel oneAPI 2024.1 编译通过。
- 服务器 native gate：109 tests passed（含 colidx 原地修改失效、
  wide-K K=200/300/500/602/1024 数值合同）。
- `aggregate_saved_v4` forward/backward/autograd smoke：PASS。
- 新旧 BF16 pull 的数值在升格至 FP32 后逐值零误差。

## 尚未处理

- hidden-layer FP32 epilogue→ReLU/Dropout→BF16 bridge；
- native Q / whole-terminal session（执行期 packed-W 基础已完成）；
- cost-based Y-first/Q-first planner；
- source-lifetime/frontier streaming 的真实图可行性 gate。

## Cycle 4：terminal packed-W 生命周期

- 新增显式 weight packing 与 packed-logits native API。
- terminal 只有在 `rows > row_tile` 时才在 panel 循环外 pack 一次；单 panel 继续走原接口。
- packed tensor 仅在当前 autograd forward 内生存，不跨 optimizer step 缓存，避免权重更新后复用旧 packing。
- packed 与 legacy 在 `D=47/129/2983`、尾块和 1/4/8 线程下逐位一致。
- 服务器完整回归：`98 passed`，V4 autograd smoke PASS。
- 同节点 A/B：真实规模趋近下总体为持平到小幅改善；`D=2983`、8 panels、32 threads 为 `1.084x`。该项作为去重和 whole-terminal micro-engine 的基础保留，不作为主要加速贡献。

## Cycle 5：wide-K 静态首层 SX cache

- 允许显式 `TFS_STATIC_AGGREGATE=on` 将静态、无需 dX、`K>128,D<=128`
  的首层从动态 `S(XW)` 改为一次性缓存 `(SX)`，每轮只执行 `(SX)W`。
- `auto` 保持原维度规则不变，未通过真实图性能 gate 前不会自动启用。
- producer/backward 去除了不必要的 `K<=128` 人工限制；现有 AMX dW 内核按
  `Kp` 分块，本身支持宽 K。
- planner 在 dispatch 前检查 `TFS_AGG_CACHE_MAX_BYTES`（其次继承
  `TFS_HS_CACHE_MAX_BYTES`）；预算不足时回退动态路径，避免运行中选中缓存路径后
  再因容量失败。
- K=200/300/500/602/1024 native 数值 gate 已通过。
- 真实图 32T 首层 forward+backward（排除一次性 cache build）性能 gate：
  - Flickr K=500：`5.324x`；cache build 30.7 ms；相对 L2 误差 0.264%。
  - Reddit K=602：`16.504x`；cache build 909.3 ms；相对 L2 误差 0.281%。
  - Yelp K=300：`5.443x`；cache build 414.1 ms；相对 L2 误差 0.282%。
  - AmazonProducts K=200：`29.089x`；cache build 4823.5 ms；相对 L2 误差 0.126%。
- 这些是首层 kernel/dataflow gate，不是端到端训练加速比。大收益来自静态 SX 同时
  消除每轮首层 forward 和 dW backward 的 sparse traversal；由于 BF16 下发生
  `S(XW)` 与 `(SX)W` 重结合，仍需完整 convergence gate 后才能考虑进入 `auto`。

## Cycle 6：D-slab 零拷贝适配

- Python 不再对每个 `grad[:,d0:d1]` 和 `weight[:,d0:d1]` 强制
  `.contiguous()`；native API 接受 `stride(1)=1` 的二维 column slab，并显式使用
  `stride(0)` 访问每一行。
- native 数值 gate 验证 strided view 与旧 contiguous copy 逐值一致。
- 32T、N=100003、degree=8、slab width=257 的同节点 A/B：
  - Transform High-D K=1024：`1.050x`，每 slab 避免约 103.9 MB copy。
  - Aggregate High-D K=128：`2.650x`，每 slab 避免约 102.9 MB copy。
- 这是 adapter 数据搬运修复，不等同于真正的 CSR single-scan；多 D-slab 的重复
  sparse traversal 仍然存在，需作为后续独立周期处理。

## Cycle 7：Aggregate High-D 一次最终 CSR pull

- 每个 D-slab 仍使用已验证的 native AMX dense/dW 内核，但不再各自产生完整
  `part_dx[N,K]`，也不再逐 slab 执行转置 sparse pull。
- Python 只分配一份预算内的 FP32 `dP[N,Kp]`；各 native slab 将 AMX FP32 结果
  原地累加到该缓冲区。所有 slab 完成后只进行一次 FP32→BF16 边界转换和一次
  scaled CSR pull。
- 因此同时消除了逐 slab `part_dx`、`dx.add_(part_dx)` 的 N×K 扫描以及重复 CSR
  遍历；planner 将 bounded dW workspace 和唯一 dP accumulator 纳入同一预算。
- K=65 尾块与 K=128、两 D-slab 数值 gate 均通过；服务器完整回归为
  `112 passed`，V4 autograd smoke PASS。
- 32T、N=100003、degree=12 的无争用交替 A/B：
  - D=513（2 slabs）：`1.515x`；
  - D=2983（12 slabs）：`3.430x`。
- 真实训练 CSR gate：IGB-HOM-small，N=1,000,000、NNZ=12,068,130、K=128、
  D=2983、32T：旧版 `2403.62 ms`，新版 `743.76 ms`，即 `3.232x`；CSR pull
  从 12 次降为 1 次。dW/db 逐值一致，dX 相对 L2 差异 `0.322%`。
- 以上是该 High-D backward 路径的性能 gate，不是完整训练端到端加速比；该分支
  仍保持显式 opt-in，进入 auto 前还需收敛与完整训练 gate。

## Cycle 8：隐藏层 fused bridge 探索（暂不接入）

- 原型将 ReLU、给定 dropout mask、dropout scale、下一层 source scale、BF16-RNE
  合并为一次 native 扫描，并用 1-byte/element 状态完成融合 backward。
- K=47/128/129、1/4 线程下，FP32 activation、BF16 staging、状态和 backward
  梯度均与显式 reference 逐值一致；服务器完整回归为 `119 passed`。
- 静态同 mask 的理想化 kernel gate 曾显示 forward/backward 有收益，但这不能代表
  训练，因为没有计入 dropout 随机数生成。
- 完整训练边界口径（N=1,000,000、K=128、32T）计入 mask 分配与 RNG：PyTorch
  ReLU+Dropout+scaled-BF16+autograd 为 `48.93 ms`，当前 native 组合为
  `62.38 ms`，即 `0.784x`，发生回退。
- 因此当前 bridge 不接入 authority，也不作为有效优化。后续只有在 native
  counter-based RNG 能与融合扫描共同完成、且重新通过完整口径 gate 时才重启；
  当前转向优先级更高且已有成熟 AMX 基础的 terminal Native-Q。

## Cycle 9：terminal Native-Q 探索（不接入 planner）

- 增加 shadow-only 的 packed `W^T` 与 AMX BF16 Q 接口；权重打包可在同一
  terminal forward 的多个 30 万行 panel 间复用，Q 以 BF16 直接写出。
- 尾维 K=47/65/128、D=31/129/2983、1/4 线程数值测试通过；服务器完整回归为
  `126 passed`，V4 autograd smoke PASS。
- 32T、每 panel 30 万行、K=128、D=2983 的 Q 子算子（包含写入全局 Q）测试：
  1/2/3 panels 分别为 `1.677x / 2.044x / 2.224x`；算入一次权重打包后仍为
  `1.594x / 1.988x / 2.184x`，相对 L2 误差约 `0.0027%`。
- 维度 crossover 并不等于 `D>K`：60 万行时 D=512 为 `0.787x`，D=1024 为
  `1.291x`，D=2048 为 `2.481x`。
- 但真实 IGB-HOM-small 完整 terminal forward+CE+backward gate 中，除 Q 实现外
  所有路径保持一致，框架 Q 中位数为 `419.96 ms`，Native-Q 为 `420.75 ms`，
  即 `0.998x`；dH 相对 L2 误差 `0.0029%`，dW/db 一致。
- 结论：Native-Q 的局部 kernel 收益被 terminal 其他阶段完全稀释，暂不接入
  planner/authority。保留独立原型和测量证据；除非后续能与 Gs 生成或 sparse
  consumer 形成真正的跨算子流水，否则不继续微调该孤立 GEMM。

## Cycle 10：普通隐藏层 destination-pull Gs panel（淘汰）

- High-D Aggregate 的现有 `streamed_aggregate_backward` 和 one-final-pull 已经只
  保留一个 `Gs` row panel；本周期针对仍全局物化 `Gs[N,Dp]` 的普通 D≤128
  backward，复用 fused source-scale/BF16 sparse pull 做了独立 shadow ABI。
- 1/4/8 线程数值 gate 通过；8T 大规模 dW 相对 L2 约 `5e-7`，dX 一致。32T
  原型暴露跨 NUMA dW reduction 边界问题，但性能结论已足够明确，不继续修补。
- N=100003、K=128 的 8T 公平 A/B（旧全局 Gs / destination-pull panel）：
  - D=47，degree=4/12/64：`0.534x / 0.328x / 0.095x`；
  - D=128，degree=4/12/64：`0.567x / 0.159x / 0.150x`。
- 根因不是 AMX dense consumer，而是 destination-pull 在每条边访问 source 时重复
  执行 FP32 `grad*scale` 和 BF16 rounding；全局 Gs 虽有写入成本，却把转换摊为
  每个 source 一次。随着平均度上升，panel 方案的重复转换成本线性放大。
- 该 ABI、测试和 launcher 已从 shadow 源码清除，正式路径从未修改。后续若继续
  消除普通层全局 Gs，只考虑 source-driven 分桶/局部归约，使每个 source 的 Gs
  最多生成一次；不再尝试 destination-pull 内按边即时转换。

## Cycle 11：source-row Gs panel + Q（不接入）

- 为避免 Cycle 10 的按边重复转换，测试了按连续 source rows 生成一次 Gs panel，
  立即用于 `pulled_panel^T @ Gs_panel` 累加 dW，并生成 BF16 Q panel；所有 panel
  完成后对全局 K-wide Q 执行一次 sparse pull。该实验只在 benchmark 中实现，
  没有修改任何生产 dispatch。
- 32T 单 socket 独占、N=100003、K=128、degree=12、15 次重复结果：
  - D=47，row panel=25k/50k/100003：`1.010x / 0.918x / 1.016x`；
  - D=128，row panel=25k/50k/100003：`0.950x / 0.923x / 1.110x`。
- 多 panel dW 因 FP32 分段累加顺序改变，相对 L2 约 `0.235%`；dX/db 一致。
- 真正 bounded 的 25k/50k panel 没有稳定收益。D=128 的单个全量 panel 虽有
  `1.110x`，但候选峰值 Gs 已恢复到 24.4 MiB，并同时持有 24.4 MiB Q，不能视为
  全局 Gs 消除；D=47 时 Q 还是原 Gs 的两倍宽。
- 结论：普通 D≤128 层中，单纯把全局 Gs 换成 bounded Gs + 全局 Q 不值得进入
  真实图或正式源码。若未来重启，必须同时消除/缩小下游全局 Q，或与 sparse
  consumer 构成无需全局 accumulator 的新数据布局；否则只是移动中间张量。

## Cycle 12：source-panel Q 直接累加 dH（性能淘汰）

- 使用静态 selected-transpose 将每个 source panel 的边组织成 destination-owned
  rectangular CSR；运行时执行 `Gs panel -> Q panel -> FP32 dH accumulate`，所有
  panel 完成后仅做一次 BF16 rounding 和 output scale。因此不保存全局 Gs，也不
  保存全局 Q，最终 FP32 dH 同时充当必要输出和累加器。
- 新增 shadow-only rectangular BF16-to-FP32 accumulate primitive；目的仅是验证
  数据流下界，不进入 authority planner。服务器回归为 `127 passed`。
- 32T 单 socket 独占、N=100003、K=128、degree=12：
  - D=47，25k/50k source panel：`0.563x / 0.695x`；
  - D=128，25k/50k source panel：`0.670x / 0.669x`。
- dX/db 与原路径一致；分 panel dW 相对 L2 约 `0.235%`。静态分区预处理约
  9--12 ms（不计入迭代），额外静态 metadata 约 12.2--13.7 MiB。
- 性能回退来自每个 source panel 都要扫描 rectangular rowptr、反复读写完整 FP32
  dH accumulator，并承受多个 dense launch。说明在普通 D≤128 层中，同时消除
  Gs/Q 的收益不足以覆盖跨 panel 的全局输出流量。
- 结论：该路线保留为可复现的负结果/下界证据，不进入正式执行路径，也不继续做
  真实图。普通层全局 Gs 不再作为当前优先优化对象。
