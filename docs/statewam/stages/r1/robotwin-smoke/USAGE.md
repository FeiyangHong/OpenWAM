# R1 RoboTwin 简洁用法

Goal ID：`r1/robotwin-smoke`。已通过的检查与已知限制见 [REPORT.md](REPORT.md)，范围见 [GOAL.md](GOAL.md)。本页只覆盖 smoke、200–500 步 profiling 和部署，不包含完整训练或全量评测。

## 前提与路径

在 `feat/statewam-r1-robotwin` 分支、OpenWAM 根目录执行。使用已准备的两个独立环境；资产、依赖版本和补丁见报告。运行前用 `nvidia-smi` 检查占用，项目同时使用不超过 4 张卡，不能使用被其他任务占用的卡。

```bash
cd /gaozt-test1/fyhong/OpenWAM
SWPY=/gaozt-test1/fyhong/envs/statewam/bin/python
RTPY=/gaozt-test1/fyhong/envs/robotwin/bin/python
SW_ART=/gaozt-test1/fyhong/OpenWAM/statewam_artifacts/robotwin
SW_DATA=/gaozt-test1/datasets/robotwin/smoke-clean/dataset
SW_BASE=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Clean2Random
SW_INIT="$SW_ART/checkpoints/clean2random-state-init"
SW_TRAINED="$SW_ART/runs/r1-profile-200-001/2026-09-12_18-50-22"
export ROBOTWIN_PATH=/gaozt-test1/fyhong/RoboTwin-statewam-r1
export ROBOTWIN_PYTHON="$RTPY"
export PYTHONUTF8=1 OMP_NUM_THREADS=8
export __EGL_VENDOR_LIBRARY_FILENAMES=/gaozt-test1/fyhong/envs/robotwin/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json
```

以下 `my-*` 输出目录须未使用过；重跑时换 run id，不覆盖已有证据。每个新终端重新设置上述变量。不要把 HF token 写入命令文件；公开资产可匿名读取，需要认证时使用受保护的运行环境。

## 工程检查

```bash
"$SWPY" -m pytest tests/statewam -q
"$RTPY" scripts/statewam/prepare_robotwin_env.py --output "$SW_ART/manifests/my-dependency-patches.json"
CUDA_VISIBLE_DEVICES=0 "$RTPY" scripts/statewam/robotwin_env_smoke.py \
  --seed 100001 --output "$SW_ART/runs/my-env-clean"
```

仅首次转换新权重时执行下方两行；已有 `SW_INIT` 可跳过转换直接复用。输出必须是新目录。

```bash
SW_INIT="$SW_ART/checkpoints/my-state-init"
"$SWPY" scripts/statewam/initialize_checkpoint.py "$SW_BASE" "$SW_INIT"
```

真实权重 forward/backward 检查：

```bash
CUDA_VISIBLE_DEVICES=0 "$SWPY" scripts/statewam/verify_checkpoint.py \
  --baseline "$SW_BASE" --state "$SW_INIT" --dataset "$SW_DATA" \
  --output "$SW_ART/reports/my-real-weight-check"
```

## 4 卡 Profiling

先将 `--updates` 设为 2、使用另一个输出目录做启动检查；再默认运行 200 步，需要确认稳态或定位稳定性问题时可延长至 500 步。默认 per-GPU batch=4、accumulation=1、global batch=16。`--updates` 指 optimizer updates，不是 microsteps。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 "$SWPY" -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  --log-dir "$SW_ART/runs/my-profile/launcher" --redirects 3 --tee 3 \
  scripts/statewam/profile_training.py --checkpoint "$SW_INIT" \
  --dataset "$SW_DATA" --output "$SW_ART/runs/my-profile" \
  --updates 200 --batch-size 4 --workers 4
```

实际 checkpoint 目录记录在 `my-profile/profile/checkpoint_directory.txt`；逐步和汇总数据也在 `profile/`。测试恢复时将 `--checkpoint` 指向该目录、增加 `--resume`，将 `--updates` 设为累计目标（200 步后续跑一步即 201），使用新输出目录保存续训诊断。全状态会保留，不要只拷贝 safetensors 后要求恢复优化器。若采用下方部署命令测试自己的新 run，将 `SW_TRAINED` 改为此目录；当前变量默认指向已验收的 step 201 目录。

可选的完整生成与保存重载检查（单卡，需要额外一份约 25 GB checkpoint 空间）：

```bash
CUDA_VISIBLE_DEVICES=0 "$SWPY" scripts/statewam/verify_generation.py \
  --checkpoint "$SW_TRAINED" --dataset "$SW_DATA" \
  --output "$SW_ART/reports/my-generation" --roundtrip
```

单独检查 compile/cache 时，将 `--roundtrip` 换为 `--compile`，并使用新的输出目录。优化误差的已测范围见报告，不将其视为逐元素等价。

## 策略服务与 Smoke

发布基线服务，终端一运行；保留默认 10 步 sync、compile 和 velocity cache：

```bash
CUDA_VISIBLE_DEVICES=0 "$SWPY" scripts/deploy.py \
  --ckpt-dir "$SW_BASE" --host 127.0.0.1 --port 8851
```

终端二运行，每条件 1 episode，不改原生规划器、筛选和步数上限：

```bash
CUDA_VISIBLE_DEVICES=1 "$RTPY" scripts/statewam/rollout_smoke.py \
  --condition demo_clean --label baseline --port 8851 \
  --output "$SW_ART/runs/my-baseline-clean"
CUDA_VISIBLE_DEVICES=1 "$RTPY" scripts/statewam/rollout_smoke.py \
  --condition demo_randomized --label baseline --port 8851 \
  --output "$SW_ART/runs/my-baseline-randomized"
```

StateWAM 工程 smoke：先停止基线服务，再以训练 checkpoint 启动 eager、无 cache 服务；客户端沿用上面两条命令，把 label 和输出目录分别改为 `statewam`、`my-state-clean` / `my-state-randomized`。

```bash
CUDA_VISIBLE_DEVICES=0 "$SWPY" scripts/deploy.py \
  --ckpt-dir "$SW_TRAINED" --host 127.0.0.1 --port 8851 \
  --compile-enabled false optimization.dit_cache.enabled=false
```

结果以 `smoke_result.json`、`episodes.json` 为准；日志在 `console.log`，视频在 run 内的 `runtime/eval_result/`。上游 `_result.txt` 的分母固定为 100，不用于少量 smoke。

常见问题：UTF-8 报错检查 `PYTHONUTF8=1`；缺 `curobo_left.yml` 时在固定 RoboTwin worktree 根目录运行 `"$RTPY" script/update_embodiment_config_path.py`；首次 compile 有额外等待。`UnStableError` 是上游 seed 筛选，不启用 planner fallback 来伪造通过。
