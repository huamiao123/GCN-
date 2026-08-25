# `final_pre_numa` package note

本文件原先描述的是 2026-08-14 的三图、混合单步协议旧包，已经不适用于当前版本。当前唯一说明入口是 [README.md](README.md)，版本与来源契约见 [VERSION_MANIFEST.md](VERSION_MANIFEST.md)，实施验收见 `reports/FINAL_PRE_NUMA_IMPLEMENTATION_ACCEPTANCE_20260819.md`。

当前正式协议统一为四类图、2/3 层、1/2/4/8/16/32 线程、每个 cell 一个独立冷进程和完整 200 epoch；不得再引用旧的“2 warmup + 7 measured”单步结果作为本版本权威数据。
