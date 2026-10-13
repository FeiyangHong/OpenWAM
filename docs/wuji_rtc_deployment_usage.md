# OpenWAM 星尘 + Wuji TrainingRTC 部署

## 1. 启动 Server（172.16.10.32）

将 TrainingRTC 训练输出放到下面的 checkpoint 目录，包含训练保存的 `config.yaml`、
`normalization_stats.npy`、tokenizer 和权重。修改目录与权重文件名即可；不能用普通同步训练的配置冒充 RTC checkpoint。

```bash
cd /home/ubuntu/astritbot_wuji/OpenWAM
CUDA_VISIBLE_DEVICES=0 \
OPENWAM_WUJI_CKPT=/home/ubuntu/astritbot_wuji/OpenWAM/checkpoints/grasp_anything_base_rtc \
bash scripts/deploy_wuji_rtc_server.sh \
  --ckpt-name checkpoint_step_4000.safetensors \
  --port 8848 \
  --compile-enabled true
```

脚本自动使用 `/home/ubuntu/miniconda3/envs/openwam/bin/python`。
看到 `TrainingRTC server started on ws://0.0.0.0:8848` 后启动客户端。
首次编译在客户端预热阶段完成，期间不下发策略动作；需要更长等待时增加客户端 `--timeout`（默认 300 秒）。
先排查非编译路径时用 `--compile-enabled false`。RTC 入口默认关闭 compile，以上命令显式开启；DiT cache 保持关闭。
这里不使用 `--inference-mode async` 或 `--inference-horizon`，动作队列和异步请求由 Orin 管理。

## 2. 启动 Orin Client

```bash
cd /home/astribot/workspace/openwam-wuji-client

./run_rtc_client.sh \
  --server ws://192.168.0.100:8848 \
  --task "Pick up the items from the table and place them in the trash bin." \
  --execution-horizon 16 \
  --rtc-prefix-steps 10 \
  --rtc-warmup-rounds 2 \
  --enable-arms --enable-hands \
  --control-hz 30 \
  --control-way filter \
  --filter-scale 0.35 \
  --max-eef-step 0.03 \
  --workspace-radius 2 2 2 \
  --max-rotation-step-deg 10 \
  --max-hand-step 0.10
```

只验证相机、状态与模型通信：去掉 `--enable-arms --enable-hands`，加 `--max-chunks 2`。
仍需 ROS 和 SDK 正常，但不发送机械臂或手指动作。空格或 Ctrl+C 停止。

| 参数 | 含义 |
| --- | --- |
| `--execution-horizon 16` | 每个 chunk 采用 16 个新动作；不再等 32 步全部执行完才采观测。 |
| `--rtc-prefix-steps 10` | 队列剩余 10 步时采观测并请求下一 chunk，同时继续执行这 10 步，约留出 333 ms。 |
| `--rtc-warmup-rounds 2` | 执行前，两轮预热各覆盖无前缀和有前缀推理；预热结果不执行。 |
| `--max-tick-lag-ms 50` | 控制线程单次调度落后超过该值即停止，不突发补发命令。 |

当前动作 chunk 长度为 32，须满足 `0 < prefix < execution-horizon`、
`prefix + execution-horizon <= 32`，且 prefix 不超过 checkpoint 的训练上限（当前通常为 12）。
200–250 ms 延迟先用 `10/16`；若端到端延迟超过预算，客户端停止，不接入错位 chunk。
修改 prefix 后重新启动客户端，重新预热。更换服务端端口时同步修改 `--server`。

前缀使用已经限幅、即将下发的原始 58D 指令；保留压缩图像字节 + MsgPack 传输、SDK 滤波及相机过期检测。
手指仍检查目标相对实测位置的 `--max-hand-step`：若执行时必须再次改写已承诺的目标，RTC 停止并报告 `hand tracking guard`。

原同步入口仍为 `scripts/deploy_wuji_server.sh` 和 `run_client.sh`，用法见 [同步部署文档](wuji_deployment_usage.md)。
RTC 的 server/client 必须成对使用。CPU 回归与模拟通信测试不代表真机闭环效果或 GPU 编译延迟已验证。
