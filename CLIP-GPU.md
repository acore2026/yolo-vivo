# pc-17 CLIP GPU 迁移（2026-11-10）

远端项目 `/home/huawei/VIVO/yolo-vivo`，服务 `systemctl --user ... yolo-stream.service`。
已重启上线，`/health/ready` 的 `model.classifier_device` 为 `cuda:0`。

改动：

- `service/flower_runtime.py` 将 CLIP encoder 与线性分类头迁到同一设备。
- 新增 `service/flower_inference.py`，预处理后的 batch 转到对应设备，归一化特征、分类与 softmax 在 GPU 上完成，仅将结果搬回 CPU。
- 保留 FP32、batch size 16、候选阈值、裁剪和预处理、权重及原 NMS 逻辑。未修改训练／离线评估代码。
- 显式关闭 CUDA matmul / cuDNN 的 TF32，避免迁移时降低数值精度。此设置也作用于同进程的 GPU 检测器。
- `FLOWER_CLASSIFIER_DEVICE=auto` 为默认，跟随检测器设备；支持 `cuda:0` / `0`，也可显式设 `cpu`。
- 默认无 CUDA 时原有 `FLOWER_DEVICE=auto` 选择 CPU；显式指定不可用 GPU 会在启动预热时失败，不伪造 ready。
- 原 runtime 备份：`runtime/clip-gpu-backup/flower_runtime.py`。

验证与证据：

- 远端 `scripts/verify_clip_gpu.py`：9 组实际图片、两个视频采样及空白图；固定检测器候选比较 CPU/GPU 特征、分数、类别、阈值和 NMS 后保留框。
- 最终类别和框全部相同；最大特征绝对误差 2.63e-6，分类分数误差 2.98e-6。另验证空候选和跨 batch 的 20 个候选。
- CLIP（含 crop 预处理和 H2D）中位数 CPU 142.93 ms → GPU 11.68 ms，CUDA 计时显式同步。
- 远端 3 项 presentation / 启动生命周期回归测试通过。
- 真实服务 480×640 输入、相同尺寸输出，20 帧成功；模型处理中位数 26.44 ms，JPEG WebSocket 往返 84.04 ms，本机编解码及处理调用总计 92.37 ms。
- 相同竖屏样本先前 CPU 在线路径分别为 188.67 / 235.78 / 246.20 ms；不同时间测量的网络条件可能变化，不能将全链路差异全部归因于 GPU。
- 两路隔离 WebRTC → sandbox → 真模型 → 手机轨道、网页观看和预览通过，覆盖 640×480→960×720。
- 测试没有写入生产 QoS 保障申请，也没有修改前端。上述不是手机采集到显示的端到端时延，不保证 30 FPS。

本地证据：`vivo-demo/runtime/pc17-integration/clip-gpu-verification.json`、`clip-gpu-transport.json`、`clip-gpu-portrait.jpg`、`clip-gpu-webrtc/`。
远端报告：`runtime/clip-gpu-verification.json`。

复现（远端）：

```sh
cd /home/huawei/VIVO/yolo-vivo
train/.venv/bin/python scripts/verify_clip_gpu.py
train/.venv/bin/python scripts/test_flower_stream_processor.py
curl http://127.0.0.1:8767/health/ready
systemctl --user status yolo-stream.service
journalctl --user -u yolo-stream.service -n 50 --no-pager
```

验证脚本会独立加载模型、短暂使用 GPU；建议在没有手机推流时运行，避免影响实时性能。
