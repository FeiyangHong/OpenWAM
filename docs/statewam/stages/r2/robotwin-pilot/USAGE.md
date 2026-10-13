# R2 RoboTwin Pilot 简洁用法

Goal ID：`r2/robotwin-pilot`。范围和门槛见 [GOAL](GOAL.md)，进度、历史故障与审计证据见 [REPORT](REPORT.md)。本页只维护已验证入口，不将工程命令当作正式训练/评测用法；三卡生命周期 smoke 已通过，正式协议 `r2-pilot-v1` 已冻结。正式训练和 2040 条配对 rollout 仍未开始。

## 环境与路径

在 `feat/statewam-r2-robotwin-pilot` 分支、OpenWAM 根目录执行：

```bash
cd /gaozt-test1/fyhong/OpenWAM
SWPY=/gaozt-test1/fyhong/envs/statewam/bin/python
SIMPY=/gaozt-test1/fyhong/envs/robotwin/bin/python
R2_ART=/gaozt-test1/fyhong/OpenWAM/statewam_artifacts/robotwin/r2-robotwin-pilot
R2_DATA=/gaozt-test1/datasets/robotwin/r2-robotwin-pilot/clean50-785feb15-s20260912-v1
```

正式任务的物理 GPU allowlist 固定为 `0,1,2,3`，同时最多占用这四张卡；GPU `4–7` 禁止使用。启动训练、评测、渲染或服务前必须按 GPU UUID、compute/graphics 进程、显存、租约和心跳重新核验四张 allowlist GPU 均可安全使用。最近一次阻塞证据为 `statewam_artifacts/robotwin/r2-robotwin-pilot/reports/gpu_allowlist_block_20260913_1535.json`；其中 GPU 1/2/3 存在无法归属的外部上下文，因此当时不得启动正式四卡 run。

RoboTwin worktree 为 `/gaozt-test1/fyhong/RoboTwin-statewam-r1`，沿用固定资产和模拟器环境。数据已准备好，不需重新下载或提供 token；不要覆盖已有数据、checkpoint 或报告。精确 EEF 转换审计使用 SIMPY，不能换成训练 Python。

## 状态与日志

已完成的三卡多服务工程检查（不重复 launch）：

```bash
RUN_ID=p4-pilot-lifecycle-001
"$SWPY" scripts/statewam/r2/gpu_job.py status --run-id "$RUN_ID"
tail -n 30 "$R2_ART/runs/$RUN_ID/console.log"
nvidia-smi
```

该检查曾租用物理 GPU 0/1/2，执行登记的 8 个工程 episodes；GPU 0 共享模拟器，另两卡运行 E0/E3 服务，结果在该 run 的 `evaluation/` 和 `episodes/`。run 已 exit 0，8/8 记录通过独立生命周期审计，但全部结果均为工程阶段 `failure/action_limit`，不是正式成功率。该 run 无故障注入或自动重试，三卡已释放；500 场景收集与 20 项跨进程 reset 审计已完成，不重建原场景目录或重跑已验收项。

已登记 CPU `p4-pilot-lifecycle-audit-001` 有界等待同一 GPU run 正常结束及释放后独立审计；失败或超时即退出，不自动重试或启动训练。查询入口如下，不重复 launch：

CPU 测试/审计用对应入口查询：

```bash
"$SWPY" scripts/statewam/r2/cpu_job.py status --run-id p4-pilot-lifecycle-audit-001
tail -n 20 "$R2_ART/runs/p4-pilot-lifecycle-audit-001/console.log"
```

`status` 同时检查 PID 和创建时间；结合心跳、日志及 GPU 进程核验，不仅看历史状态。首次启动时尚无日志不等于失败，不重复 launch。

## 中断、接管与停止

interrupt 只中断当前助手操作，不主动停止后台任务，任务可能继续占卡；不保证助手继续监控或自动进入下一阶段。接续时可直接说：“继续执行本 R2 Goal，先核验后台任务并接管，只执行原预算剩余部分，不重复启动。”

只有确实要停止指定的存活 GPU 任务时，才执行：

```bash
"$SWPY" scripts/statewam/r2/gpu_job.py stop --run-id "$RUN_ID"
"$SWPY" scripts/statewam/r2/gpu_job.py status --run-id "$RUN_ID"
nvidia-smi
```

CPU 任务将上述 `gpu_job.py` 换为 `cpu_job.py`。STOP 请求不等于已退出：等待有界收尾，检查 `supervisor_live_now=false`、`child_live_now=false`、GPU run 的 `live_processes_now=[]`，再核对 compute/graphics 上下文释放。不得停止其他用户进程；非零退出或未核验释放须如实保留。

已完成的 profiling、batch16 恢复与物理 STOP/SIGKILL 两对工程实验不要重跑。恢复训练必须用完整 checkpoint 和新 run id；评测保留原账本，已完成的成功和失败均跳过，中断 attempt 先核验分类，最多一次同场景重试。正式恢复命令待正式入口验收后补充，不通过修改 R1 profiling 步数实现。

## 证据与性能

产物均在 R2_ART 内，详细路径和哈希见 REPORT：

| 内容 | 入口 |
|---|---|
| 数据与 train-only 统计 | `reports/p0_acceptance.json`、R2_DATA 内 manifest |
| 四组有效容量与限制 | `reports/capacity_report.json`，不宣称 compute-matched |
| E0/E3 各 200 步 profiling | `reports/profiling_paired_001.json` |
| batch16/global64 完整恢复 | `reports/batch16_resume_audit_001.json` |
| 真实物理恢复汇总 | `reports/physical_recovery_001.json` |
| 完整开发/最终场景清单 | `reports/evaluation_scenes_001.json`，100 / 400 scenes |
| 20 项跨进程初态核验 | `reports/cross_process_reset_001.json` |
| 资源总账快照 | `reports/resource_accounting_001.json` |
| 各 run 原始资源与汇总 | `runs/<run-id>/resource_samples.jsonl`、`performance_summary.json` |

训练计时与样本/噪声轨迹在 `training/timing_rank<0..3>.json` 和 `steps_rank<0..3>.jsonl`。资源总账按物理 UUID 区间去重，租卡时长含启动、保存和等待；历史越界及 PID 归属盲区保留，不把未知计算量/能耗填零。

重新生成资源快照可使用新的 run id 和输出路径：

```bash
"$SWPY" scripts/statewam/r2/cpu_job.py launch --run-id my-resource-accounting -- \
  "$SWPY" scripts/statewam/r2/resource_accounting.py \
  --output "$R2_ART/reports/my-resource-accounting.json"
```

`exports/engineering-e3-b16-update4-001/` 是已验证的工程部署导出，不得进入正式 E0–E3 矩阵。正式协议文件为 `$R2_ART/manifests/protocol.yaml`，SHA256 为 `30ae4bd51745a985f1f10b0cb87c0c2d7c1acbbfb3f1d627a446de2cd1b2b83c`。正式结果统一使用冻结 N 的最终 checkpoint；正式训练、2040 个既定 rollout 和最终汇总命令只有在 allowlist 资源核验通过、相应入口门槛满足并实际启动后才会执行。

## 复核

已验证的 CPU 数据复核和测试入口；重跑使用新 run id，测试不启动训练：

```bash
"$SWPY" scripts/statewam/r2/cpu_job.py launch --run-id my-reader-audit -- \
  "$SWPY" scripts/statewam/r2/verify_data.py --data "$R2_DATA" \
  --output "$R2_ART/runs/my-reader-audit/reader_audit.json"

"$SWPY" scripts/statewam/r2/cpu_job.py launch --run-id my-statewam-tests -- \
  "$SWPY" -m pytest tests/statewam -q
```

CPU launcher 隐藏 CUDA，不能拿它启动 GPU 工作负载；CUDA 专项跳过不算通过，已有真实 GPU 补验另见 REPORT。旧工程运行的日志、脱敏命令和配置保留在各 run 的 `job.json`、源码快照与 REPORT，不在 usage 中逐次追加历史状态。
