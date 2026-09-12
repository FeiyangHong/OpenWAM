# R2 RoboTwin Pilot 简洁用法

Goal ID：`r2/robotwin-pilot`。本页随执行更新，目前仅含已验证的 CPU 准备入口。训练、四卡恢复和正式评测尚未完成工程验收，不使用 R1 profiling 命令替代。进度与证据见 [REPORT](REPORT.md)。

## 路径

在 `feat/statewam-r2-robotwin-pilot` 分支、OpenWAM 根目录执行：

```bash
cd /gaozt-test1/fyhong/OpenWAM
SWPY=/gaozt-test1/fyhong/envs/statewam/bin/python
R2_ART=/gaozt-test1/fyhong/OpenWAM/statewam_artifacts/robotwin/r2-robotwin-pilot
R2_DATA=/gaozt-test1/datasets/robotwin/r2-robotwin-pilot/clean50-785feb15-s20260912-v1
```

## 查看与停止

```bash
"$SWPY" scripts/statewam/r2/cpu_job.py status --run-id p0-data-001
tail -n 40 "$R2_ART/runs/p0-data-001/console.log"
tmux list-sessions
```

`status` 同时核验 PID 和创建时间；重新进入对话后先查看状态，不重复启动。同一 run id 禁止再次启动。interrupt 不等于停止任务，独立任务可能继续运行；助手中断期间不保证主动监控或启动下一阶段。

停止仍在运行的 CPU 准备任务：

```bash
"$SWPY" scripts/statewam/r2/cpu_job.py stop --run-id p0-data-001
"$SWPY" scripts/statewam/r2/cpu_job.py status --run-id p0-data-001
```

等待状态变为 `stopped`，并核对 `child_live_now=false`。准备入口隐藏 CUDA，不能用于启动 GPU 工作负载。GPU 训练/评测的安全停止命令将在其验收后补齐。

## 数据准备与审计

当前数据已准备好，不需要重新下载。审计使用新的 run id：

```bash
"$SWPY" scripts/statewam/r2/cpu_job.py launch --run-id my-reader-audit -- \
  "$SWPY" scripts/statewam/r2/verify_data.py --data "$R2_DATA" \
  --output "$R2_ART/runs/my-reader-audit/reader_audit.json"
```

首次准备其他数据版本，使用未完成过的新目录和 run id；失败的准备可在确认旧进程退出后，用新 run id 指向未完成的数据目录续作，已校验归档会复用。已完成的数据版本禁止覆盖：

```bash
"$SWPY" scripts/statewam/r2/cpu_job.py launch --run-id my-data-preparation -- \
  "$SWPY" scripts/statewam/r2/prepare_data.py \
  --output /gaozt-test1/datasets/robotwin/r2-robotwin-pilot/my-data-version \
  --reuse-root /gaozt-test1/datasets/robotwin/upstream-785feb15
```

默认匿名使用 `hf-mirror.com`，不在命令或文件中加入 token。检查数据目录内的 `source_manifest.json`、`episode_manifest.json`、`statistics_provenance.json` 和 `data_summary.json`，以及 run 内的退出状态与性能摘要。

## 局部测试

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 "$SWPY" -m pytest \
  tests/statewam/test_r2_data.py tests/statewam/test_r2_capacity.py \
  tests/statewam/test_r2_cpu_job.py tests/test_robotwin_dataloader.py -q
```
