# 新电脑部署与 Sandbox 对接交接

本文件供新 AI 会话接手。目标：在新电脑启动现有三类花卉推理服务，让云端 sandbox 把手机视频帧送来处理，再把检测框和介绍框画面送回手机。不是重新训练，也不是启动本机摄像头。

## 1. 先读与先确认

按顺序阅读：本文件、`STREAMING.md`、`service/stream_app.py`、`service/stream_protocol.py`、`service/stream_processor.py`、`service/flower_runtime.py`、`service/flower_inference.py`、`train/three_flower_cascade.py`、`scripts/restore_flower_weights.py`。

先只读核对：

- 项目实际路径、系统（推荐 Linux / WSL2）、Python、GPU/驱动、可用空间、Git 当前提交。
- 是否已拉取部署权重和分块，不能假设上一台电脑已经推送成功。
- sandbox 实际部署在哪台云服务器、哪个容器、现用视频处理客户端和配置入口。其源码不在本项目内，必须实际查找，不能编造环境变量名或服务名。
- 云端能否 SSH 到新电脑；旧 `pc-17` 别名是否仍指旧电脑；新电脑能否 SSH 到云端。
- sandbox 网络命名空间、已有隧道、端口占用，避免切断旧服务或重复启动同名流。

先给简洁部署计划。安装依赖、修改配置、启动/重启服务、切换生产 sandbox 前取得用户确认；同一批准计划连续执行，不反复确认小步骤。不要重训、改阈值或换模型来掩盖部署错误。禁止打印私钥、令牌。

## 2. 当前行为与模型

入口：`python -m service.stream_app`，默认监听 `127.0.0.1:8767`。
默认工厂：`service.stream_processor:create_processor`，结果头 processor 为 `three_flowers_introductions`。
不要启动 `service.car_app`；它是旧摄像头服务。

识别：芍药 peony、百合 lily、菊花 chrysanthemum；不识别桉树叶。名称延续此前经用户批准的类别映射，不代表重新鉴定植物品种。

- YOLO-World LoRA + CLIP 视觉分类器；阈值 detector=0.05、classifier=0.7、类无关 NMS IoU=0.5。
- 框上分数为检测置信度乘分类置信度。
- 每目标连续匹配 10 个已处理帧后，右侧逐字显示介绍，打完一次后保持；连续丢失 10 个已处理帧后清除，重新识别重新打印。
- 框宽 220px，正文14、标题18，每字0.04秒；超出边缘直接截断。480×640 竖屏支持，但右侧介绍可能被裁掉。
- 每会话独立状态；重连、分辨率变化重置介绍；快速运动、遮挡、类别改变可能重置跟踪。
- 模型共用并串行调用，丢弃的传输帧不计为未识别帧。当前分类器默认随检测设备，可用 `FLOWER_CLASSIFIER_DEVICE=cpu` 单独指定。
- 不承诺所有场景准确率/召回率达到90%；合成图传输测试不能证明识别准确率。

部署资产：

| 文件 | 用途 |
|---|---|
| `models/yolov8s-worldv2.pt` | 基础模型 |
| `train/runs/three_flowers_20261109_160053_774018_full/weights/adapter_best.pt` | LoRA |
| `train/workflows/three_flowers_20261109_160053_774018/classifier/classifier.pt` | 三类及背景拒绝分类器 |
| 同目录 `operating_point.json` | 固定阈值 |
| `models/clip-vit-b32/manifest.json` 与6个 `.partXX` | CLIP 分块 |
| `train/.cache/weights/clip/ViT-B-32.pt` | 还原后的 CLIP |
| `service/fonts/` | 中文介绍字体及许可 |

基础模型 SHA256：`03cad0f87262b635f2a10fc3a4feec2a8cf30220726048e0fd77016234ed769b`。
分类器 SHA256：`e4ee07981f5f50af67665310484a8e45987c8aebbc2bb4e02931757c55efc1c6`。
CLIP SHA256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`。
LoRA 的校验值在分类器 checkpoint 中，加载时严格核验。新版加载器把 checkpoint 的旧绝对路径映射到当前项目，不能修改 checkpoint 后绕过哈希。

## 3. 获取项目、恢复权重、准备环境

仓库：`https://github.com/acore2026/yolo-vivo`。根据实际权限选择 HTTPS 或 SSH 登录，已有项目先检查本地改动再拉取，禁止强制覆盖。

```bash
git clone https://github.com/acore2026/yolo-vivo.git
cd yolo-vivo
git log -1 --oneline
python3 scripts/restore_flower_weights.py
```

此脚本只用标准库，校验分块并还原338 MiB的CLIP文件，无需 Git LFS。文件缺失时先查远端版本，不要下载其他同名模型代替。现有有效文件会直接校验通过。

新电脑不要复制旧 `.venv`。旧环境为 Python3.12、ultralytics8.4.183、torch2.14.1+cu130、torchvision0.29.1；这些是历史环境记录，不保证新机器/当前软件源可用。
`train/requirements-local.lock.txt` 是库存记录，不是跨机器安装锁；`service/requirements.txt` 包含旧 WebRTC 摄像头依赖且版本范围宽，不能直接视为本服务的精确部署锁。

经用户批准后创建环境：

```bash
python3.12 -m venv train/.venv
train/.venv/bin/python -m pip install --upgrade pip
```

先按新 GPU、驱动和官方 PyTorch 安装矩阵安装匹配的 torch/torchvision。5090/Blackwell 需要支持该架构的构建。CPU 可以验证功能，但吞吐需实测。不要仅凭 `nvidia-smi` 判断 PyTorch 可用。
之后安装流服务依赖（在新虚拟环境中）：

```bash
train/.venv/bin/python -m pip install 'ultralytics==8.4.183' 'aiohttp>=3.10,<4' numpy Pillow opencv-python PyYAML ftfy regex tqdm
train/.venv/bin/python -m pip install 'git+https://github.com/openai/CLIP.git'
train/.venv/bin/python -m pip check
train/.venv/bin/python -c "import torch,clip,ultralytics; print(torch.__version__,torch.cuda.is_available(),ultralytics.__version__); print(clip.__file__); print(torch.ones(1,device='cuda') if torch.cuda.is_available() else 'CPU')"
```

必须是 OpenAI CLIP，不能安装 PyPI 上其他同名包冒充。实际安装版本需记录；若固定版本不可用或要替换，先说明兼容性风险并验证模型加载，不能静默换版本。流入口不依赖 aiortc/av；sandbox 如使用 PyAV 则在其自己的环境保留依赖。

## 4. 先前台启动验证，再安装后台服务

项目根目录执行：

```bash
FLOWER_DEVICE=0 FLOWER_CLASSIFIER_DEVICE=auto train/.venv/bin/python -u -m service.stream_app --host 127.0.0.1 --port 8767
```

无 GPU 验证时改为 `FLOWER_DEVICE=cpu FLOWER_CLASSIFIER_DEVICE=cpu`。默认不传设备变量时自动选择 CUDA0 或 CPU，需检查健康接口防止意外退回CPU。
模型先校验、加载、预热，成功后才监听端口。刚启动 curl 拒绝连接可能仍在加载，查看日志后等待，不要循环重启。
另一个终端：

```bash
curl -fsS http://127.0.0.1:8767/health/ready
curl -fsS http://127.0.0.1:8767/v1/streams
train/.venv/bin/python -m unittest scripts.test_flower_stream_processor scripts.test_stream_service
train/.venv/bin/python scripts/probe_stream_service.py --url http://127.0.0.1:8767 --frames 30
```

健康检查要求 ready=true、model.ready=true、三类名称和正确哈希；查看 device/classifier_device。active_sessions=0 表示还没有客户端，不是模型失败。
probe 使用合成图、两路、640×480→1280×720，只验证协议/回传，不验证花卉识别或竖屏。还需真实手机480×640画面及花卉逐帧测试。
默认处理超时2秒，发送超时2秒、空闲15秒、最多4路。若慢机/多路超时，先测耗时与负载，再降低发送帧率或调整 `--process-timeout`，不能伪称实时。

前台验证后停止前台进程，按实际路径修改 `deploy/yolo-stream.service` 的 WorkingDirectory 和 ExecStart，不能保留旧 `/home/huawei/VIVO/yolo-vivo` 路径。需要固定GPU时在 [Service] 加 `Environment=FLOWER_DEVICE=0`，按需设置分类器设备。然后：

```bash
mkdir -p ~/.config/systemd/user
cp deploy/yolo-stream.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now yolo-stream.service
systemctl --user status yolo-stream.service --no-pager -l
journalctl --user -u yolo-stream.service -n 100 --no-pager
```

之后重启/停止：`systemctl --user restart yolo-stream.service` / `systemctl --user stop yolo-stream.service`。
用户服务不使用 `sudo systemctl --user`。WSL须启用systemd且保持运行；需要登出后常驻时，经批准配置该用户 linger（不会让关掉的WSL自动启动）。

## 5. 与云端 Sandbox 建立通道

推荐 YOLO 保持回环监听，通过 SSH 加密隧道对接，不把8767直接暴露公网。

**方案A：云服务器能SSH到新电脑。** 在云服务器执行，将示例 `YOLO_SSH_ALIAS` 换成已验证的新机 SSH 别名/用户地址：

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8876:127.0.0.1:8767 YOLO_SSH_ALIAS
```

**方案B：新电脑在NAT后，只有新电脑能SSH到云服务器。** 在新电脑执行，将 `CLOUD_SSH_ALIAS` 换成实际云端 SSH 地址：

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:8876:127.0.0.1:8767 CLOUD_SSH_ALIAS
```

选一种即可，确保云端8876未占用、SSH允许相应转发。在云端验证：

```bash
curl -fsS http://127.0.0.1:8876/health/ready
```

sandbox 与隧道在同一网络命名空间时，配置客户端地址为：
`ws://127.0.0.1:8876/v1/streams/{stream_id}/ws`。

**容器注意**：bridge容器内127.0.0.1不是宿主机。不能把宿主机隧道地址原样填进容器，也不能认为host.docker.internal一定可访问只监听宿主机回环的端口。先检查网络；可把隧道放到sandbox所在网络命名空间，或按现有部署使用host网络/受控可达端点。改变网络或监听地址需确认，不要简单公开绑定0.0.0.0。

初次前台验证通道，成功后按实际部署将隧道交由systemd等管理，处理断线重连；不要把临时终端存活当作稳定部署。

## 6. Sandbox 客户端契约

链路：手机/WebRTC → sandbox解码 → JPEG/WebSocket → YOLO绘制 → JPEG/WebSocket → sandbox恢复PTS并编码 → 手机/前端。YOLO端点不是浏览器播放器，不接受手机直接发的SDP/RTP/H.264。
实现参照 `scripts/probe_stream_service.py` 和 `service/stream_protocol.py`，详细规格见 `STREAMING.md`。

1. 每路设备一条WS连接，stream_id使用独立标识；连接先收type=ready，保存session_id。
2. 发一条完整二进制消息：4字节大端JSON长度 + UTF-8 JSON + JPEG。不是Base64、multipart或裸JPEG。
3. 请求头：`{"version":1,"type":"frame","frame_id":42,"codec":"jpeg","pts":126000,"time_base":[1,90000]}`。
4. 同连接frame_id严格递增，允许跳号。pts/time_base要么都提供，要么都省略；回传时恢复原时间戳，不能固定假FPS替代真实时序。
5. 返回相同封包结构，type=result，含原frame_id/pts/time_base、stream_id/session_id、实际width/height、耗时和处理后的JPEG。先区分文本ready/error与二进制结果。
6. 用session_id/frame_id拒绝旧连接结果；解码JPEG并替换输出帧，保留时间戳。通过原有编码器输出给手机，不要再把原始帧覆盖到结果上。
7. 每路最多一个在途请求、一个最新待发帧；服务端会丢积压帧，不保证所有发送帧有响应。客户端超时、重连要匹配服务端策略，禁止无限排队。
8. 409同名流占用、503容量满需退避重连；断开/超时停止展示旧结果，遵循sandbox现有Hide video策略，不生成黑屏假成功。
9. 分辨率以实际JPEG为准，480宽640高竖屏无需旋转成横屏；发送端正确处理手机方向信息、别强行拉伸。返回原尺寸，分辨率切换会重置介绍。
10. 如设置STREAM_API_TOKEN，所有HTTP/WS请求需Bearer头。只在安全配置中存储，不写进仓库或URL。

当前仓库不含云端生产sandbox适配器。新AI必须先找到实际配置/代码再切换端点；不知道位置就问用户，不要编造“修改某某变量即可”。

## 7. 验收和故障定位

按层验证：本机健康 → 本机probe → 云端隧道健康 → sandbox同网络内probe → 手机单路真实花卉 → 竖屏与分辨率切换 → 多路及断线重连。
记录实际设备、模型哈希、帧数/丢帧、耗时、截图/短片、介绍状态，说明哪些没有验证。

- 缺权重：检查Git版本、分块、还原输出；检查LoRA路径映射和哈希，不关闭校验。
- 无法连接：检查预热日志、端口、服务退出、隧道和容器网络，分层定位。
- 有回传无检测：检查实际输入画面、模型输出和方向，不能从返回帧数推断识别成功。
- 有框无介绍：需连续10处理帧，同类IoU跟踪可能中断；右侧可能越界截断。
- 频繁409：同名流重复连接或旧推理仍在draining，检查客户端重连管理。
- TIMEOUT：看GPU/分类器设备、串行模型等待与实际耗时。output_age_ms是距上次输出的时间，不是推理耗时。
- 历史日志记录曾有108/108及481/481回传且无错误，但不能作为新电脑部署的验收。

交付应包含：新机项目路径/提交号、环境版本、启动及停止命令、systemd文件、隧道运行位置/地址、sandbox实际改动位置、健康返回、实测证据及未完成项。不要宣称未执行的推送、部署、模型准确率或跨机器测试已成功。
