# Sandbox 双向视频帧接口

本入口为 `python -m service.stream_app`，独立于现有 `service.car_app`。
默认处理器已接入三类花卉检测、置信度和逐字介绍框。新电脑部署请先读
[HANDOFF_NEW_PC_CN.md](HANDOFF_NEW_PC_CN.md)，恢复权重并核对环境。

链路：服务器 sandbox 解码帧 → JPEG / WebSocket → pc-17 处理器 → JPEG / WebSocket
→ sandbox 恢复原时间戳并编码为手机及前端使用的视频轨道。
本仓库包含 YOLO 服务和参考客户端；云端生产 sandbox 的适配器及配置需在其部署仓库核对。

## 运行

在 `/home/huawei/VIVO/yolo-vivo` 中：

```sh
train/.venv/bin/python -m service.stream_app --host 127.0.0.1 --port 8767
```

复用现有 `train/.venv`，不安装或升级依赖。默认仅监听 WSL 回环地址。
服务不访问摄像头、训练数据或现有摄像头服务，不产生 QoS 保障申请。

安装提供的用户服务：

```sh
mkdir -p ~/.config/systemd/user
cp deploy/yolo-stream.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now yolo-stream.service
systemctl --user status yolo-stream.service
journalctl --user -u yolo-stream.service -n 100 --no-pager
systemctl --user restart yolo-stream.service
systemctl --user stop yolo-stream.service
```

WSL 必须处于运行状态。用户服务的后台生命周期取决于用户 manager；`loginctl show-user huawei -p Linger`
可查看登出后是否保留 manager。开启 linger 不会让已关闭的 WSL 自动启动。

从 sandbox 所在服务器通过现有 SSH 别名建立通道（前台，Ctrl-C 停止）：

```sh
ssh -N -T -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 -L 127.0.0.1:8876:127.0.0.1:8767 pc-17
```

此时服务器访问 `http://127.0.0.1:8876`。不需要新开公网端口，也不依赖 WebRTC UDP 穿透。
SSH 通道断开时，客户端需重建连接；不要把 `pc-17` 的 SSH 地址误用为 HTTP 地址。
验证时通道为临时创建，生产持久通道由 sandbox 接入部署阶段管理。

## HTTP / WebSocket

- `GET /health/live`：进程存活、配置、当前会话数量。
- `GET /health/ready`：默认处理器的模型加载、预热成功后就绪；不代表有新鲜视频。
- `GET /v1/streams`：活动／清理中的会话、实际接收／输出／丢弃帧数、输出新鲜度、错误。
- `GET /v1/streams/{stream_id}/ws`：双向 WebSocket，一条连接对应一路设备。

建议 stream_id 使用 `device-control`、`device-dynamic-qos`；仅允许 1–64 位字母、数字、`_`、`-`。
默认最多 4 个会话；同 stream_id 同时连接返回 HTTP 409 `STREAM_BUSY`，超过容量返回 503 `CAPACITY`。
重连等待旧会话释放，遇到 409/503 根据 `retry_after_ms` 重试并逐渐退避。
每次成功连接分配新的服务端 session_id；不同设备、重连后的检测状态完全独立。

如设置 `STREAM_API_TOKEN`，所有接口（包括 health）要求 `Authorization: Bearer <token>`。
不要把 token 放 URL。监听非回环地址时必须设置 token；直接跨不可信网络还须配置 TLS。
回环 + SSH 模式由 SSH 的认证和加密保护。

## 帧格式 v1

连接后首先收到文本 JSON：

```json
{"type":"ready","version":1,"stream_id":"device-control","session_id":"服务端生成","processor":"service.stream_processor:create_processor","max_packet_bytes":4210692,"idle_timeout_s":15.0}
```

每帧作为**一条完整二进制 WebSocket 消息**发送：

```text
4 字节无符号大端整数 N | N 字节 UTF-8 JSON 头 | JPEG 二进制
```

头最多 16 KiB，JPEG 最多 4 MiB。解码后每边不超过 4096，像素总数不超过 3840×2160。
这是 JPEG 帧传输，不接受 SDP、RTP、H.264 裸流，也不使用 Base64。

请求头示例：

```json
{"version":1,"type":"frame","frame_id":42,"codec":"jpeg","pts":126000,"time_base":[1,90000]}
```

- frame_id 必填，为 0 至 2^53−1 的整数；同一连接内严格递增，允许跳号。
- pts/time_base 可同时省略；若提供则原样回传，sandbox 应按 frame_id 恢复原帧时序。
- pts 为有符号 64 位整数，time_base 为两个正整数；JavaScript 客户端应避免丢失大整数精度。
- 分辨率从 JPEG 自身读取，允许同连接内切换，无需重新协商。

成功结果仍为同格式二进制消息，JSON 头示例：

```json
{"version":1,"type":"result","stream_id":"device-control","session_id":"服务端生成","frame_id":42,"codec":"jpeg","width":1280,"height":720,"processor":"three_flowers_introductions","pts":126000,"time_base":[1,90000],"processing_ms":0.01,"server_total_ms":8.2,"queue_ms":0.3}
```

- JPEG 为处理后的画面，宽高与输入一致；若模型按 640 推理，应把框映射回原画面后绘制。
- processing_ms 只计 process()；server_total_ms 包括服务端排队、初始化、解码、处理及编码，
  不含发送和网络传输。queue_ms 为服务端收到消息至线程开始工作的耗时。
- 用客户端单调时钟测发送至收到对应结果的往返耗时；不要用两机时钟差作为时延。
- 这些都是统计信息，不是 QoS 保障申请，也不能冒充手机端到端时延。

## 背压、失败和重连

推荐客户端每路最多一个在途请求，采集端只保留最新待发帧。两路可并行。
服务端每路最多一个执行中的任务及一个最新待处理帧，新到帧覆盖等待帧并增加 dropped_frames。
等待超过 500 ms 的帧直接丢弃，所以连续推送时不保证每个 frame_id 都有结果。
要获得每帧响应，请按参考客户端等待响应后再发送。

默认处理超时 2 秒（含等待共享模型、解码、推理、绘制、编码），发送超时 2 秒，15 秒没有帧即关闭会话。
默认模型在监听端口前完成加载预热，不计入单帧超时。慢机或多路负载需实测后调整 `--process-timeout`（最高60秒）。
调整时也应使 `--idle-timeout` 大于最慢处理时间，客户端的响应超时应留出网络及服务端处理时间。

失败尽可能先返回文本 JSON，再关闭连接：

```json
{"type":"error","code":"PROCESSING_FAILED","message":"具体模型错误","stream_id":"device-control","session_id":"本次会话","frame_id":42}
```

其他错误包括 `INVALID_FRAME`、`TIMEOUT`、`IDLE_TIMEOUT`。
超大 WebSocket 消息可能直接被协议层关闭／重置，无法保证 JSON 错误到达。
连接断开和无响应同样视为输出不可用，停止展示旧结果；前端应按既有 Hide video 行为处理。
服务不发送虚构结果、备用图或带错误文字的黑屏。

模型线程不能安全强杀：超时会断开媒体连接，但仍保留该流的槽位，状态为 draining，直到后台调用及 close() 完成。
这避免无限堆积 GPU 任务，也意味着永久卡死的模型需要 `systemctl --user restart yolo-stream.service`。
重连不能取消远端已执行的推理；新 session_id 和 frame_id 匹配用于拒绝旧结果。

## 模型替换接口

编辑 `service/stream_processor.py`，或新增模块并通过 `--processor module:factory` 指定：

```python
class MyProcessor:
    mode = 'my-yolo'

    def process(self, frame, metadata):
        # frame: numpy uint8 BGR, H×W×3
        # metadata: stream_id, session_id, frame_id, pts/time_base 等
        # 调用模型，绘制到原分辨率的画面；异常直接抛出
        return frame

    def close(self):
        # 释放会话状态
        pass

def create_processor(*, stream_id, session_id):
    return MyProcessor()
```

工厂每会话调用一次，同一会话 process 串行、在工作线程执行；不同会话可并行。
不保证同一会话总在同一个 OS 线程调用。共享 GPU 模型需自行加锁或实现调度；检测确认计数、
文字动画等必须保存在各自会话，不能共享。close 仅在已开始的 process 完成后调用。
传输层不限定具体模型，也不改变原 `car_pipeline.py`。

修改处理器后重启此服务。使用自定义工厂时，在 systemd 单元 ExecStart 中增加 `--processor`，
然后执行 `systemctl --user daemon-reload` 和 `systemctl --user restart yolo-stream.service`。

## 验证

```sh
# 隔离测试：不访问生产设备、GPU 或摄像头
train/.venv/bin/python scripts/test_stream_service.py
# 对运行服务检查两路实际 JPEG 回传、像素、帧序号和 480p→720p 切换
train/.venv/bin/python scripts/probe_stream_service.py --url http://127.0.0.1:8767 --frames 30
```

`scripts/probe_stream_service.py` 是 sandbox 客户端的最小参考实现。
测试覆盖会话隔离、自定义处理器、错误、身份验证、容量限制、输入大小限制、帧积压、超时隔离、
重连和退出；测试不向生产 QoS 历史写数据。

## 本次部署及实测（2026-11-09）

- WSL huawei 用户的 `yolo-stream.service` 已安装、enable 并启动，监听 `127.0.0.1:8767`。
- 已开启该用户 linger，避免最后一个 SSH 会话退出后用户服务随之停止；仍需要 Windows/WSL 运行。
- 10 项隔离测试通过，日志 `/tmp/yolo-stream-tests.log`。
- 从 sandbox 所在服务器通过临时 SSH 通道连接此服务，两路各发送并收到 30 帧，均完成 480p→720p。
- 两路往返中位数 58.94 / 55.35 ms，最大 122.30 / 123.61 ms；服务端处理总耗时中位数 2.01 / 1.94 ms。
- 使用合成 JPEG 图像并验证像素、分辨率、两路身份、帧序号和时间戳。默认原样回传，以上不含模型推理，
  不是手机端到端时延，也不是实拍复杂场景或持续运行的性能承诺。
- 测试连接已关闭，活动会话归零。测试用 SSH 通道在完成后关闭；pc-17 服务保持运行。
- 生产 sandbox 和手机配置没有切换，下一步由 sandbox 适配器使用上述协议接入。

## 在新电脑恢复三类花卉模型

仓库包含选定的三类分类器、operating_point.json、LoRA adapter 和基础模型。
CLIP ViT-B/32 原文件约 338 MiB，超过 GitHub 普通文件限制，因此保存为
`models/clip-vit-b32/` 下的 6 个分块（每块最多 64 MiB）。不需要 Git LFS。
克隆/下载完整仓库后，先执行：

```bash
python3 scripts/restore_flower_weights.py
```

脚本校验分块及完整文件 SHA256，并原子写入
`train/.cache/weights/clip/ViT-B-32.pt`；已存在且校验通过时不重复写入。
随后按本文安装依赖并启动服务。LoRA 的历史绝对路径会映射到当前项目内的
`train/runs/three_flowers_20261109_160053_774018_full/weights/adapter_best.pt`。
训练视频、数据集、其余训练结果和虚拟环境仍不提交。
