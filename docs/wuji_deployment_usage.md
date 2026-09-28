# OpenWAM 星尘 + Wuji 部署用法

## 1. 启动 Server

本机地址为 `192.168.0.100`，默认监听 `8848`。启动脚本会自动使用
`/home/ubuntu/miniconda3/envs/openwam/bin/python`，无需手工执行
`conda activate openwam`。

```bash
cd /home/ubuntu/astritbot_wuji/OpenWAM
OPENWAM_WUJI_CKPT=/home/ubuntu/astritbot_wuji/OpenWAM/checkpoints/grasp_anything_base \
bash scripts/deploy_wuji_server.sh \
  --ckpt-name checkpoint_step_4000.safetensors \
  --port 8848 \
  --inference-mode sync \
  --inference-horizon 32
```

首次启动启用了 `torch.compile`，模型加载和第一次推理会明显慢于后续请求。看到
`WebSocket server started on ws://0.0.0.0:8848` 后再启动 Orin 客户端。

这条命令已经包含需要日常修改的参数：

- `OPENWAM_WUJI_CKPT`：checkpoint 目录；其中必须有 `config.yaml`、
  `normalization_stats.npy`、tokenizer 和权重文件。
- `--ckpt-name`：目录中实际加载的权重文件。替换成例如
  `checkpoint_step_3000.safetensors` 即可切换权重。
- `--port`：服务端端口；修改后客户端的 `--server` 也要使用相同端口。
- `--inference-horizon 32`：训练配置为 `num_frames=33`，模型生成的动作 chunk 为
  `33 - 1 = 32` 步，因此这里设为 `32`。
- `--inference-mode sync`：当前按同步方式执行。

启用异步时，把命令中的 `--inference-mode sync` 替换为：

```text
--inference-mode async --inference-delay-steps 8
```

`inference-delay-steps` 必须小于 `inference-horizon`。它表示动作缓冲还剩多少步时启动
下一次后台推理，实际值应根据服务端日志中的推理耗时调整。启用服务端异步执行器时，
客户端还应把 `--execution-horizon 32` 改成 `--execution-horizon 1`，使每次网络请求
对应一个已经发生的真实控制步；否则会提前推进服务端的异步动作时钟。

### 保存每次推理解码的视频

如果要把模型每次实际生成的预测视频块保存为 MP4，在 Server 命令末尾增加配置覆盖：

```bash
OPENWAM_WUJI_CKPT=/home/ubuntu/astritbot_wuji/OpenWAM/checkpoints/grasp_anything_base \
bash scripts/deploy_wuji_server.sh \
  --ckpt-name checkpoint_step_4000.safetensors \
  --port 8848 \
  --inference-mode sync \
  --inference-horizon 32 \
  optimization.decode_video=true \
  optimization.save_video=true \
  optimization.video_output_dir=/home/ubuntu/astritbot_wuji/OpenWAM/debug_output/predicted_videos
```

视频默认写入服务端机器上的：

```text
<video_output_dir>/<run_id>/<session_id>/episode_<episode_id>/prediction_<prediction_id>.mp4
```

其中 `run_id` 每次重启 Server 都不同，`session_id` 每次重新连接 client 都不同，
`episode_id` 在 client 发送 `reset`（开始新任务）后递增，`prediction_id` 对应一次模型
实际生成的视频-动作块。每次生成约 9 帧，Wuji 默认按 7.5 FPS 保存；视频写盘由后台线程
执行，不会同步阻塞动作返回。服务端还会在 run 目录写入 `manifest.jsonl`，记录每个 MP4
的相对路径、帧数和 FPS。

`optimization.decode_video` 必须同时为 `true`，否则没有像素帧可保存。保存器默认使用有界
队列；磁盘或编码速度跟不上时只丢弃视频保存任务，不影响机器人动作推理。停止 Server
时应正常退出，以便后台队列完成写盘。

如果启动脚本提示 loaded/installed NVIDIA module 版本不同，说明驱动升级后尚未重启。
重启本机并确认 `nvidia-smi` 正常后，再重新运行同一条 Server 命令。

## 2. 启动 Orin Client

客户端位于：

```text
/home/astribot/workspace/openwam-wuji-client
```

推荐从默认参数开始：

```bash
cd /home/astribot/workspace/openwam-wuji-client

./run_client.sh \
  --server ws://192.168.0.100:8848 \
  --task "Pick up the items from the table and place them in the trash bin." \
  --execution-horizon 32 \
  --enable-arms --enable-hands \
  --control-hz 30 \
  --control-way filter \
  --filter-scale 0.35 \
  --max-eef-step 0.03 \
  --workspace-radius 2 2 2 \
  --max-rotation-step-deg 10 \
  --max-hand-step 0.10
```

### 机械臂

- `--control-hz`：动作下发频率，默认 `30 Hz`，应与训练数据帧率一致。
- `--control-way filter|direct`：使用星尘 SDK 的滤波控制或直接控制，默认 `filter`。
- `--filter-scale`：传给星尘 SDK 的机械臂和夹爪滤波系数，默认 `0.35`。这是 SDK
  控制器参数，修改后应先观察小范围动作，不要把它当作速度上限。
- `--max-eef-step`：相邻命令允许的最大末端平移，单位米，默认 `0.03`。
- `--max-rotation-step-deg`：相邻命令允许的最大末端旋转，单位度，默认 `10`。
- `--workspace-radius X Y Z`：以客户端启动时左右手末端位置为中心的 XYZ 活动半径，
  默认均为 `0.4 m`。
- `--eef-frame chassis|world`：EEF 坐标系，默认 `chassis`。必须与采集训练数据时一致。
- `--use-wbc`：启用星尘全身控制；只有明确需要且已经验证 WBC 配置时再添加。

调整动作平滑性时，优先一次只改一个参数：先保持 `control-hz=30`，再调整
`filter-scale`；若仍有单步跳变，收紧 `max-eef-step` 和
`max-rotation-step-deg`。安全限幅和 SDK 滤波是两层独立保护。

### Wuji 灵巧手

- `--max-hand-step`：每次命令相对实测手指位置的最大关节变化，单位弧度，默认 `0.10`。
- `--hand-effort-limit`：手部 effort 上限，默认 `1.5`。
- `--hand-lowpass-hz`：一代 `WujiHand` 实时控制器低通截止频率，默认 `5 Hz`。
- `--hand-mit-kp`、`--hand-mit-kd`：二代 `WujiHand2` MIT 控制参数，默认 `3.0/0.05`。
- `--hand-model auto|wuji_hand|wuji_hand2`：默认自动识别；多设备环境可配合
  `--left-hand-serial` 和 `--right-hand-serial` 固定序列号。

## 4. 通信与相机参数

- `--execution-horizon`：每次观测后执行的动作步数，默认 `32`，与训练动作 chunk 一致。
  `--chunk-size` 是同一参数的兼容别名。
- `--timeout`：等待服务端推理响应的秒数，默认 `300`。
- `--max-camera-age`：允许的相机帧最大年龄，默认 `0.25` 秒；任一相机过期即停止。
- `--head-topic`、`--left-wrist-topic`、`--right-wrist-topic`：覆盖三个 ROS 压缩图像话题。

客户端将 ROS `CompressedImage` 原始压缩字节通过 MsgPack 直接发送，不进行 OpenCV
解码、PNG 重编码或 base64 转换。服务端握手还会检查模型输出必须是列式 rot6d 的
58D 布局：

```text
[左 EEF9, 左手 20, 右 EEF9, 右手 20]
```

如果 checkpoint 的动作维度、rot6d 约定或排布不匹配，客户端会在下发动作前退出。
