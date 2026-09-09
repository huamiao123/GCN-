# Q-panel 流式消费与 NUMA-local dH 设计

状态：仅完成索引与并发设计核验；未修改权威 TFS，未实现正式 kernel。

## 1. 当前路径

当前 bounded terminal 仍会全局生成 `Q[M,K]`，再用 rectangular transpose CSR：

```text
Q[M,K]
  -> transpose CSR pull
  -> dH[N,K]
```

pull 让每个 worker 独占一段 dH rows，因此不需要原子操作，但 Q 必须完整存活。

## 2. 不能直接采用的方案

逐个 Q panel 扫原 selected CSR 并直接 scatter 到 dH，会让多个线程同时更新同一 dH row。
使用 FP32 atomics 会增加每条边、每个 feature vector 的同步成本，不能作为强候选。

## 3. 无原子 NUMA-local push

在一次性 scope build 阶段，将每条 selected edge 转换为：

```text
(q_panel, owner, destination_row, compact_q_row)
```

其中 `owner` 由 destination dH row 的固定连续分区决定。每个
`(q_panel, owner)` bucket 内按 destination row 排序，并保持 compact row 的确定性顺序。

运行时：

```text
produce P panel
  -> classifier + exact CE
  -> produce Q panel
  -> owner workers consume their bucket
  -> accumulate only owner-local dH rows
  -> release P/Q panel
```

每个 dH row 只有一个 writer，因此无需原子操作；dH 由 owner first-touch，可保持单 socket
NUMA domain 本地写。panel 按固定顺序执行，数值归约顺序可复现。

## 4. 成本与风险

- 预处理额外空间接近 `O(E_selected)`；使用两个 int32 索引约 8 bytes/entry。
- dH 可能被每个 Q panel 重访一次，panel 太小时会增加写流量和 launch/barrier。
- bucket 按 destination 排序改善 dH locality，但 Q 访问会变成间接访问。
- 单 socket 的 NUMA locality 仍需用真实 remote/local counter 验证，不能仅凭绑定声明。
- 该路径主要消除全局 Q 生命周期；若 current pull 已完全带宽饱和，性能未必提高。

## 5. 第一阶段 gate

不接完整训练，固定真实 IGB-small selected structure 和现有真实 Q：

1. current global-Q rectangular pull；
2. one-panel destination-owned push；
3. two/four/eight-panel destination-owned push；
4. 1/8/32T；
5. 比较 dH 误差、kernel time、额外结构内存和 dH 写流量。

只有 push 在至少 two-panel 下不慢于 pull，才继续与 terminal producer 做流水融合。
如果只能节省 Q 内存而持续变慢，则保留为低内存模式，不进入默认性能路径。
