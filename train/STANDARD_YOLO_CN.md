# 普通 YOLO 四类盆栽基线

这是独立于 YOLO-World / CLIP 的普通 YOLOv8s 检测训练方案。不会覆盖 v3、原始模型、旧训练结果或服务配置。

## 最新执行结果（2026-11-05）

用户已在 GPU 环境完成正式运行：`workflows/standard_yolo_20261105_150113_747705`，33 轮提前停止。助手已核验权重并完成原测试照片、视频 1054 帧的固定参数复测。验证集芍药 P=88.89%，视频抽样向日葵 R=75%、桉树叶 R=81.82%，尚未达到四类 P/R 均超过 90%，没有部署或替换 v3。

详见 [训练与测试对照报告](diagnostics/flower_user_test_20261105_standard_yolo/COMPARISON_CN.md)；状态文件 `STANDARD_YOLO_STATUS.json` 已更新。以下运行命令供复现，重复运行会新建训练批次，不是查看结果的命令。

## 正式运行

在能使用 NVIDIA GPU 的普通 WSL 终端执行：

```bash
cd /home/huawei/VIVO/yolo-vivo
train/.venv/bin/python -u train/standard_yolo.py --run --fetch-weights
```

命令会先核验数据和 CUDA，再从 Ultralytics 官方 GitHub release 下载缺失的 `models/yolov8s.pt`，记录来源及 SHA256；已有权重不覆盖。需要网络可达，脚本不安装依赖、不修改代理、不关闭 TLS 验证。此处哈希用于本地追踪，不冒充发布者独立校验值。

执行流程：

1. 核验现有 280 张训练图、39 张验证图及其标签、来源视频分区和审核记录。
2. 将图像和标签复制到独立目录 `train/datasets/flowers_standard_yolo_v1`，保持类别 ID 和框不变；仅数据 YAML 使用原四类身份名称，不使用 CLIP 的 lily/chrysanthemum 提示词。
3. 从普通 YOLOv8s 预训练权重开始，运行 1 轮完整数据训练检查，验证 checkpoint 能加载和推理。
4. 再从原始预训练权重独立开始正式训练，最多 60 轮，early stopping patience=15。正式训练不继承 smoke 权重。
5. 对最佳正式权重在验证集产生候选预测，扫描预先规定的 12 个全局置信度阈值，选择最小单类 P/R 最高的点；同分用 macro-F1。保存全部阈值结果和每类 TP/FP/FN/P/R。

输出为唯一时间戳目录 `train/workflows/standard_yolo_*`：

- `status.json`：阶段状态、配置、数据检查、最佳模型及哈希。
- `smoke/`：预训练权重的一轮检查。
- `full/results.csv`、训练曲线和混淆矩阵：正式训练过程。
- `full/weights/best.pt`、`last.pt`：新权重。
- `val_predictions.json`、`val_thresholds.json`、`operating_point.json`：验证预测、固定阈值扫描及选定工作点。
- 发生错误时保留 `error.txt`；失败不会自动继续正式训练或部署。

## 基线设置

配置：`train/configs/flowers_standard_yolo.yaml`。

- 普通 YOLOv8s，640 输入，batch=8，AdamW，初始 lr=0.001，seed=42。
- 标准网络微调，不使用 LoRA，不主动冻结主干；框架本身固定的 DFL 投影参数除外。
- AMP 关闭，避免引入自动 AMP 检查下载以及未经验证的混合精度变量。
- 训练增强：小幅 HSV、±12°旋转、平移、缩放、轻量透视、水平翻转、mosaic 概率 0.3；最后 10 轮关闭 mosaic。不用垂直翻转、mixup 或 copy-paste。
- 这是第一组待验证超参数，不宣称最优。增强仅训练时在线生成，验证不随机增强。
- 只训练四类专用输出，不声称保留 COCO 全部类别输出能力；旧模型作为独立能力保留。

## 评估边界

每类单独统计，类别一致且 IoU≥0.5 才计 TP；所有类 P 与 R **严格大于 0.90** 才记录 `passes=true`。选模型和阈值使用过验证集，因此通过只是开发集通过，不代表独立测试达标。

当前分区仍为同一套实物下按视频隔离，非独立场景。该流程不读取 test.MOV、test.JPG 或以前留出的测试图做训练/选阈值，不部署。正式训练完成后才对已暴露的用户测试素材进行固定参数回归比较。

## 本机检查与 CPU 流程验证

```bash
train/.venv/bin/python train/standard_yolo.py --check-only
train/.venv/bin/python train/standard_yolo.py --cpu-smoke
```

`--cpu-smoke` 仅使用按固定位置选择的 16 张训练图和 8 张验证图，320 输入、1 轮、普通 YOLOv8s **随机初始化**，验证训练/保存/加载通路。它不下载预训练权重，不能作为模型识别效果，也不替代正式微调。

初始准备阶段的环境记录：助手环境 CUDA 不可用，当时本地缺少普通 `yolov8s.pt`，下载因 `127.0.0.1:7897` 代理连接失败中断。此后用户已在 GPU 环境下载并完成正式训练，实际最新状态以上面的执行结果为准。CPU smoke 仅是此前的流程检查。
