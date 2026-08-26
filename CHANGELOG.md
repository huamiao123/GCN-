# 变更记录

拉取本仓库后请先看这里：每一条改动都配有一份说明报告，写清**改了什么、怎么验证的、
哪些验证不了、发现但没修什么**。报告放在 `docs/`，本文件只做索引。

其他入口：`README.md`（使用说明）、`VERSION_MANIFEST.md`（版本与来源契约）、
`HANDOFF.md`（服务器交接）。

## 分支 `codex/publish-authority-qfirst-20260823`

| 日期 | Commit | 内容 | 报告 |
|---|---|---|---|
| 2026-08-26 | `5bbe605` | 修复 3 个阻塞门禁的源码完整性缺陷；新增扩展符号契约门禁 | [docs/SOURCE_INTEGRITY_AUDIT_20260826.md](docs/SOURCE_INTEGRITY_AUDIT_20260826.md) |
| 2026-08-25 | `c2964fa` | 收录 TFS 核心创新独立评审与 IEEE TC 路线建议 | [docs/TFS_核心创新独立评审与TC路线建议_20260825.md](docs/TFS_核心创新独立评审与TC路线建议_20260825.md) |
| 2026-08-26 | `a354a51` | TC review high-D 候选修改（Q-first BF16 边界、tile-local bias 归约、可选 sparse prefetch） | 见 `VERSION_MANIFEST.md` 的 "2026-08-26 TC-review candidate updates" 一节 |
| 2026-08-26 | this commit | 清理 6 个未引用 `.orig` 历史残留；补充 Python/oneAPI 依赖与构建说明 | [docs/ENVIRONMENT_20260826.md](docs/ENVIRONMENT_20260826.md) |

## 当前需要注意的状态

- **`HANDOFF.md` 记录的门禁通过（Slurm job `10048941`，2026-08-20，exit `0:0`）早于 `c2964fa`
  之后的改动**，不能用来证明本分支当前的门禁状态。引用门禁结论前必须重跑
  `scripts/run_final_pre_numa_gate.sh`。
- `a354a51` 引入的候选修改「numerically gated but must pass separate performance A/B gates
  before replacing any reported authority result」（见 `VERSION_MANIFEST.md`），尚未替换任何权威数据。
- `docs/SOURCE_INTEGRITY_AUDIT_20260826.md` 中记录的 `.orig` 残留和依赖清单
  缺失已在后续提交处理；该报告保留其审计时点的历史事实。
