# 源码完整性审计与修复报告

> 审计日期：2026-08-26
>
> 审计对象：分支 `codex/publish-authority-qfirst-20260823`，起始 commit `c2964fa`
>
> 修复提交：`5bbe605` "Fix three gate-blocking source integrity defects"
>
> 审计范围：157 个受版本控制的文件（88 个 `.py`、28 个 `.sh`、7 个 `.cpp`、14 个 `.h`、13 个 `.md`）

---

## 0 · 摘要

静态 + 可运行审计共发现 **3 个真实缺陷，均已修复**；另有 **2 项遗留问题未处理**，需要维护者决定。

| # | 缺陷 | 影响 | 状态 |
|---|---|---|---|
| 1 | 门禁的两个 planner 检查对同一输入断言相反契约 | `run_final_pre_numa_gate.sh` 按现状**不可能通过** | 已修 |
| 2 | `test_formal_switches_and_plan_log` 在 authority profile 下必挂 | 同上，且裸 shell 下看不出来 | 已修 |
| 3 | 17 处调用三个从未存在的扩展符号 | 潜伏；非 AMX 回退路径必然 `AttributeError` | 已修 |
| A | 6 个 `.orig` 补丁残留被提交进仓库 | 仓库卫生 / 来源歧义 | **未处理** |
| B | 无任何依赖清单 | TC artifact evaluation 硬伤 | **未处理** |

**这三条修复不触碰任何 kernel、planner 决策或已报告的性能数字。**

> ⚠️ 重要：`HANDOFF.md` 记录的门禁通过（Slurm job `10048941`，2026-08-20，exit `0:0`）**早于缺陷 ① 和 ② 进入本分支**。那次 exit 0 不能用来证明当前分支的门禁状态。要引用本分支的门禁结论，必须重跑。

---

## 1 · 缺陷详情

### 1.1 门禁的两个 planner 检查互相矛盾

`scripts/run_final_pre_numa_gate.sh` 连续执行两个检查，它们对**完全相同的输入**断言相反的结果：

输入：`build_layer_plan(4097, 1024, 1024, threads=32, compute_dx=True)`，环境 `TFS_AGGREGATE_DSLAB_SINGLE_SCAN=on`

| 位置 | 断言 |
|---|---|
| `tests/smoke_final_pre_numa_planner.py:75-82` | 必须抛 `ValueError`，且信息含 `"not a one-CSR-scan kernel"` |
| `tests/test_execution_plan.py:477-487` | 必须返回 `execution_variant == "aggregate_highd_single_scan"` |

实测（两种环境差异仅 `TFS_HIGHD_NATIVE_STREAM`，结果完全一致）：

```
smoke 的环境:  variant='aggregate_highd_single_scan' policy='explicit'
test 的环境:   variant='aggregate_highd_single_scan' policy='explicit'
默认 auto:     variant='aggregate_highd_dslab'       policy='auto'
```

两个断言不可能同时成立。进一步地，**`"not a one-CSR-scan kernel"` 这个字符串在整个仓库中不存在**，说明 smoke 期待的那个守卫从未被实现。

**判定 smoke 为过期的一方**，依据是该 variant 事实上已完整实现：

- `python/tfs_train/execution_plan.py` 中该 candidate 的 `reason` 明写 `"explicit reference: one final CSR pull; not an independent native single-scan kernel"`，并标注 `performance_state="experimental"`、`validated_for_auto=False`
- `python/tfs_train/dimension_dispatch.py:206` 有对应的 dispatch 分支
- `python/tfs_train/highd_backward.py:810` 定义了 `streamed_aggregate_single_scan_backward`

即：设计意图是「显式引用实现，永不进 auto」，而非「显式请求要报错」。`test_execution_plan.py` 的测试名 `test_aggregate_dslab_single_scan_is_explicit_reference_only` 也印证这一点。

**修复**：smoke 改为断言真正实现的契约 —— 永不进 auto、只能显式选中、`selection_policy == "explicit"`、`performance_state == "experimental"`、`validated_for_auto` 为假。

### 1.2 `test_formal_switches_and_plan_log` 在 authority profile 下必挂

planner 读取优先级（`python/tfs_train/execution_plan.py:681`）：

```python
local_dw_budget = _nonnegative_int(
    "TFS_MAX_LOCAL_DW_BYTES",
    _nonnegative_int("TFS_LOCAL_DW_BUDGET_BYTES", 0))
```

`TFS_MAX_LOCAL_DW_BYTES` 优先，读不到才回退别名 `TFS_LOCAL_DW_BUDGET_BYTES`。而 `scripts/tfs_standard_env.sh` 的 authority profile **两个都导出为 0**。测试只设置了别名，于是永远读到 profile 的 0，断言 `plan.local_dw_budget == 1048576` 必然失败。

实测：

```
裸 shell:            passed=62 failed=0
authority profile:   passed=61 failed=1   ← 门禁正是这样跑的
```

门禁第 41 行 `python -m pytest -q tests/test_execution_plan.py` 是近期新增的（`scripts/run_final_pre_numa_gate.sh.orig` 中可见该行为新增），加入时未在 profile 下验证过。

**修复**：测试中 `monkeypatch.delenv("TFS_MAX_LOCAL_DW_BYTES", raising=False)`，使别名回退路径真正被覆盖。修复后两种环境均 62/62。

### 1.3 17 处调用从未存在的扩展符号

以下三个符号被调用，但**在 `csrc/` 与 `include/` 中没有任何定义**，也不在 `bindings_aggregate.cpp` 的 32 个 `m.def` 注册中：

| 符号 | 调用点数 |
|---|---:|
| `c3_forward` | 6 |
| `c3_backward_selective` | 6 |
| `c3_forward_transform` | 5 |

分布于 `python/tfs_train/native.py`、`python/tfs_train/authority_autograd.py` 及 4 个 benchmark harness（`tests/hybrid_fullstep_{arxiv,products}[_detailed].py`）。任何一次调用只能从 autograd 内部抛出裸 `AttributeError`。

**可达性分析**：三者全部位于 `HYBRID_AMX_FORWARD != "1"` / `HYBRID_AMX_BACKWARD != "1"` 的 `else` 分支。核查全部 launcher 后确认，`HYBRID_AMX_FORWARD=0` **只**出现在 `dgl_stock` 臂（该臂根本不进入这些模块），TFS 臂一律为 1。因此这是**潜伏缺陷而非现网回归** —— 而这恰恰解释了为何历次数值门禁都无法发现它。

**修复**：这些调用点改为经 `tfs_train.native.require_non_amx_c3` 在决策点抛出可操作的契约错误，指明本 release 只提供 AMX C3 kernel、应设置 `HYBRID_AMX_FORWARD=1` / `HYBRID_AMX_BACKWARD=1`，或改用 C2 路径 `TFSConvCSR(runtime="reference")`。

`TFSConvCSRParallelFunction` 的 `forward` 与 `backward` 两半都是此类调用（即整个类不可用）。保留其导入面（`tfs_train/__init__.py` 与 `modules.py` 仍引用），但改为直接声明契约。经核查，全部现有调用方使用的都是 `runtime="reference"`，无人走 `parallel` 路径。

### 1.4 新增防回归门禁

新增 `tests/test_extension_symbol_contract.py`：源码级交叉核对 `python/tfs_train/` 与 `tests/` 中每一个 `backend().<name>` 是否存在于 `bindings_aggregate.cpp` 的 pybind 导出中。

- **不需要编译扩展，不需要 AMX 硬件**，任何能跑源码的机器都能执行
- 已用注入假符号（`backend().c3_does_not_exist`）做过反向验证，确认能捕获
- 已接入 `scripts/run_final_pre_numa_gate.sh`，位置在 planner smoke 之前（因其不依赖任何构建产物，失败可最早暴露）

当前输出：`extension symbol contract: PASS (17 used / 32 exported)`

---

## 2 · 验证结果

| 检查项 | 结果 |
|---|---|
| 88 个 `.py` 语法编译 | 0 错误 |
| 28 个 `.sh` `bash -n` | 全部通过 |
| 16 个 `tfs_train` 模块 import | 全部通过 |
| 包内 `tfs_train.*` import 符号解析 | 0 处断裂 |
| `csrc` 本地 `#include` 解析 | 0 处未解析 |
| `setup_backward_opt_aggregate_wide.py` 的 7 个源文件 | 全部存在 |
| shell 脚本调用的 `tests/` `scripts/` 目标 | 0 处缺失 |
| 扩展符号契约 | PASS (17 used / 32 exported) |
| `tests/test_execution_plan.py`（裸 shell） | 62 / 62 |
| `tests/test_execution_plan.py`（authority profile） | 62 / 62 |
| `tests/smoke_final_pre_numa_planner.py`（authority profile） | PASS |

### 本地无法验证的部分

审计在 Windows + Python 3.14 + torch 2.10.0+cpu 上进行，以下**未能执行**，需在服务器上重跑门禁确认：

- 所有需要编译扩展的测试（`test_extension_*`、high-D smoke、`hybrid_*_detailed` 的数值段）—— 需 Linux + AMX + oneAPI 构建
- 所有需要 DGL 的测试 —— 本地 `dgl` 安装损坏（`libdgl.dll` 缺依赖）
- `test_execution_plan.py` 用等价的最小 pytest 替身执行（该文件仅用到 `monkeypatch.setenv/delenv`、`raises`、`mark.parametrize/skipif`），服务器上应以真实 pytest 复核

---

## 3 · 发现但未处理的遗留项

### A. 6 个 `.orig` 补丁残留被提交进仓库

```
csrc/experiments/backward_opt_20260814/v6_wide_aggregate_probe.cpp.orig
scripts/run_arxiv_v2_3_auto_200e.sh.orig
scripts/run_final_pre_numa_gate.sh.orig
scripts/run_igb_v2_3_auto_200e.sh.orig
scripts/run_products_v2_3_auto_200e.sh.orig
scripts/run_v2_3_auto_fixed_pair_200e.sh.orig
```

其中 `v6_wide_aggregate_probe.cpp.orig` 与当前版本相差 383 行，包含 worker pool 从「首次请求 right-size」改为硬编码 32 容量的那次改动（与 `tfs_standard_env.sh` 中 `TFS_GLUE_E8_RIGHTSIZE_POOL=0` 的注释对应）。

无法判断这些文件是**有意保留的历史快照**还是 `patch` 未清理的残留。`VERSION_MANIFEST.md` 未提及它们。**删除属不可逆操作，未执行**，请维护者确认。

### B. 仓库无任何依赖清单

无 `requirements.txt`、`environment.yml`、`pyproject.toml`、`conda` 环境文件。但门禁实际依赖：`pytest`、`dgl 2.1.0`、特定版本 torch、oneAPI 编译器（`scripts/build_extension.sh` 中 `module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp`）。

对 IEEE TC 的 artifact evaluation 而言这是硬伤：评审者无法重建环境。

### C. 文档中的失效引用（低优先级）

以下路径被文档引用但不存在于仓库（`reports/gates/*.json` 等运行时产物已排除）：

| 引用目标 | 引用来源 |
|---|---|
| `reports/FINAL_PRE_NUMA_IMPLEMENTATION_ACCEPTANCE_20260819.md` | `PACKAGE_README.md`（作为「实施验收」入口） |
| `scripts/run_conservative_e2e_20260816.sh` | `docs/R5_SOURCE_FAIRNESS_AUDIT_20260816.md` 等 |
| `scripts/run_dimension_thread_matrix_igb_20260816.sh` | `docs/R5_DIMENSION_GENERALIZATION_20260816.md` |
| `scripts/run_numa_e2e_igb_20260816.sh` | `docs/R5_DIMENSION_GENERALIZATION_20260816.md` 等 |
| `scripts/validate_dimension_igb_full_20260816.sh` | `docs/R5_DIMENSION_GENERALIZATION_20260816.md` |

另有 32 个 `tests/` 下的文件不被任何脚本或文档引用，其中包括 `test_extension_worker_pool_capacity.py` —— 而 worker pool 容量语义恰好是 `.orig` 差异中被修改的部分，即**守护该改动的测试不在任何门禁中**。

---

## 4 · 服务器上的下一步

1. 重跑 `scripts/run_final_pre_numa_gate.sh`。新增的符号契约检查排在最前，数秒内出结果；若失败，报错会直接给出 `文件:行号`。
2. 门禁通过后，`HANDOFF.md` 中「Completed acceptance jobs」一节应追加本次重跑的 job id，并注明 `10048941` 对应的是 `c2964fa` 之前的源码状态。
3. 遗留项 A / B 请给出处置意见。
