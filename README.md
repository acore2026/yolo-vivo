新电脑部署与 Sandbox 对接：[交接文档](HANDOFF_NEW_PC_CN.md)。

# yolo-vivo

独立的模拟救护车机器车实时检测服务。服务从摄像头或视频读取画面，使用 `models/robot_car.pt` 检测单类 `ambulance_robot_car`，通过 HTTP、SSE 和 WebRTC 提供实时结果及带框视频。

## 启动

在项目根目录执行：

```bash
python -m service.car_app
```

默认配置在 `service/car_config.yml`：输入 `/dev/video10`，GPU `0`，监听 `127.0.0.1:18767`，输出固定为 640×480。局域网访问时可将 `host` 改为 `0.0.0.0`，再重启服务。

浏览器打开 `http://127.0.0.1:18767/`，点击“启动检测”和“连接画面”。接口包括 `/v1/detections`、`/v1/detections/stream`、`/v1/webrtc/offer` 和 `/v1/monitor/{start|stop}`。

Windows 浏览器访问 WSL 服务时，推荐点击“连接画面（HTTP）”：通过 `/v1/video/frame.jpg` 连续获取带框及文字的 JPEG 画面，约 10 帧/秒，不依赖 WebRTC 的 UDP/ICE 连通性。此接口使用同样的 Token 鉴权，无新鲜画面时返回 503，页面会隐藏旧画面并自动重试。也可点击“连接 WebRTC”使用原视频轨道。如果检测状态为 READY 且 fresh 为 true，但 WebRTC 没有画面，可用 HTTP 预览区分视频传输问题和摄像头采集问题。

连续 10 帧有效推理结果检测到小车后，在检测框右侧显示深蓝至紫色渐变面板，配青紫霓虹渐变圆角描边。面板左上角固定在检测框右上角右移 8 像素的位置，随检测框移动。每 100 毫秒追加一个可见字符，Markdown 标记不参与打字，不会逐个闪现。文本框固定宽度 220 像素（`MarkdownPanel.WIDTH`），正文 14 像素、标题 18 像素，自动换行，高度随内容向下增长。不向左避让；超出画面右侧或底部的部分直接裁剪，检测框贴近右边缘时文字可能完全不可见。多个小车检测框同步显示。

支持常用 Markdown 子集：标题、段落、粗体、强调、无序及有序列表、引用、分隔线、行内代码、围栏代码块和链接文字。强调使用紫色标识，链接使用青色下划线；视频中的链接不可点击。字体保持项目内置中文字体。暂不支持表格、图片、HTML 或复杂嵌套 Markdown；不加载外部资源。渲染器在 `service/markdown_panel.py`，无需浏览器或额外依赖。

当前使用 `service/car_overlay.py` 中 `ArrivalOverlay.MESSAGE` 的长段示例文字测试效果。整段输出后保留约 2 秒，再从首字循环逐字显示；时长参数为 `CHARACTER_SECONDS` 和 `HOLD_SECONDS`。修改示例文字时也需更新内置字体子集，使其包含新增字符。

连续 10 帧未检测到小车后隐藏文字并重置播放进度；下次再次连续 10 帧检测到小车时，从“救”字重新开始。不足 10 帧的短暂漏检保留最后的检测框位置并继续播放，不重置进度。中途出现相反检测结果会清零对应连续计数。计数以实际完成并被接收的推理结果为准，丢弃帧和 WebRTC 重复播放帧不计数；重新启动检测时重置提示。无新鲜画面时仍显示 `NO FRESH VIDEO`。

中文提示使用项目内置的 Noto Sans CJK SC 字体子集（仅包含提示文字），无需安装系统中文字体；字体许可见 `service/fonts/OFL.txt`。

依赖可使用已有环境安装 `service/requirements.txt`。视频、数据集、训练产物和运行日志不属于本项目。

### WebSocket 三类花卉检测与介绍

`service.stream_app` 默认使用 `service.stream_processor:create_processor`，接入保留的
`three_flowers_20261109_160053_774018` 三类 YOLO-World LoRA + CLIP 分类器。
识别名称为芍药（peony）、百合（lily）、菊花（chrysanthemum），不显示桉树叶。
沿用 detector_conf=0.05、classifier_conf=0.7、class-agnostic NMS IoU=0.5；
框上分数是检测置信度乘分类置信度，并非单独的 detector score。

每个目标连续识别 10 个已处理帧后，在框右侧显示 Markdown 霓虹渐变介绍框
（宽 220px、字号 14、标题 18、每字 0.04 秒）。逐字打印一次后保持完整内容；
连续 10 个已处理帧未匹配则清除，下次重新计数、打印。传输丢帧不计作未识别。
每个连接独立状态，断线重连或分辨率变化会重置状态。匹配采用同类 IoU，
快速运动、遮挡或类别抖动仍可能重新触发。右侧超出图像的部分截断。

模型在监听端口前校验权重、加载并预热，失败则启动失败，不会悄悄原样回传。
多路共用一份模型并串行推理，各路介绍状态独立。
`FLOWER_DEVICE=auto`（默认）选择 CUDA 0，否则 CPU；可设 `FLOWER_DEVICE=0` 强制 GPU，
或 `FLOWER_DEVICE=cpu`。CLIP 分类器保持原离线评估的 CPU 推理路径。
`/health/ready` 中的 `model` 提供实际设备、三类名称和分类器哈希。
默认单帧处理超时仍为 2 秒（包含等待共享模型）；高分辨率/多路需实测吞吐，
必要时通过 `--process-timeout` 调整，同时降低输入帧率，避免积压。

```bash
cd /home/huawei/VIVO/yolo-vivo
systemctl --user restart yolo-stream.service
systemctl --user status yolo-stream.service --no-pager
curl http://127.0.0.1:8767/health/ready
```

接口保持 `ws://127.0.0.1:8767/v1/streams/{stream_id}/ws`，通过现有 SSH 隧道访问。
返回原尺寸 JPEG、原时间戳，`processor=three_flowers_introductions`。
调试原样回传可显式选择 `--processor service.stream_processor:create_passthrough_processor`。

```bash
train/.venv/bin/python -m unittest scripts.test_flower_stream_processor scripts.test_stream_service
train/.venv/bin/python scripts/probe_stream_service.py --help
```

2026-11-10 接入验证：本地介绍状态/生命周期 3 项测试通过；真实权重在 CPU 上加载、
预热并对 `test_videos/test.JPG` 推理成功，三类均有输出。
`train/diagnostics/stream_flower_smoke/preview.jpg` 使用该次真实预测模拟打字完成后的展示，
不是实时连续帧跟踪验收。当前执行沙箱禁止监听套接字和访问 systemd 用户总线，
10 项传输测试及服务重启仍需在 pc-17 终端运行；未宣称本次跨机器实测通过。
