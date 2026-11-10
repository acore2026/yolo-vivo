# 普通 YOLO 第二轮：补充整盆样本与尺度增强

本轮已准备数据与脚本，尚未执行正式 GPU 训练。旧模型保留，不部署。

## 数据交付与审核

- 审核源集：`train/datasets/flowers_reviewed_standard_v2/`，原图、YOLO 标签、data.yaml、manifest.json、validation_report.json。
- 框架隔离副本：`train/datasets/flowers_standard_yolo_v2/`，相同图像与标签，避免修改源集缓存。
- 原训练280张加20张新帧，共300张/522个框；val39张/93个框原样保留。
- 新帧来自 flower1 的12帧和 flower4/5/6/7 各2帧，排除flower2验证视频、flower3测试视频及用户测试图片/视频。
- 新框49个：芍药13、原兰花14、原向日葵10、桉树叶12；整盆含盆，可辨且多数可见的截断目标按可见范围框选，少量模糊边缘局部不标。
- 新帧独立手工绘框，没有使用旧帧坐标或模型候选继承真值；逐张查看640×360原图与叠框缩略图，修订后再次查看。标记为assistant_visual，非人工验收；未声称全部全分辨率复核。
- 预览/审核记录：`train/diagnostics/flower_hard_batch_v2/` 的 raw_01..05.jpg、boxes_01..05.jpg、review_decisions.json、reviewed_manifest.json。pending_manifest.json保留抽帧时状态，不代表最终审核结果。
- manifest记录源视频SHA256、帧号/FPS/尺寸、原图及标签SHA256；源视频已核对哈希。格式、坐标、原始319张不变、val不变、图像/帧唯一及视频分区检查通过。
- 同一套实物/场景且新帧邻近旧帧，不能声称新增独立场景或已经解决泛化；四类沿用用户命名，未做植物学鉴定。

## 训练方案

配置 `train/configs/flowers_standard_yolo_v2.yaml`：从官方原始yolov8s.pt重新开始，不从旧微调权重接训；640像素、AdamW、batch8、60轮上限、patience20、seed42。

训练在线增强：随机缩放从0.35扩大到0.5，保留12度旋转、0.1平移、轻透视、水平翻转、适度HSV；仅训练集增强。关闭Mosaic，避免第一轮提前停止时尚未进入无Mosaic阶段的问题，并观察真实画面分布训练的表现。其他参数维持原方案。此次同时修改数据与增强，是组合实验，不能据此拆分各项收益。

运行：

```bash
cd /home/huawei/VIVO/yolo-vivo
train/.venv/bin/python -u train/standard_yolo.py --config train/configs/flowers_standard_yolo_v2.yaml --run
```

输出独立时间戳目录 `train/workflows/standard_yolo_v2_*`。先1轮GPU smoke，再从原始预训练权重正式训练；保存best、曲线、status、验证预测及仅从val选择的工作阈值。best选择仍采用Ultralytics默认fitness，不能保证同时优化每类P/R。

## 已实际验证

- 新旧配置preflight均通过；模型与训练基线权重哈希复核通过。
- CPU随机初始化1轮小集冒烟成功：`train/workflows/standard_yolo_v2_20261105_152852_622309/`。
- 冒烟16训练/8验证张、320像素；仅验证训练管线与输出加载，不用于准确率结论。实际查看train_batch0增强预览，非所有随机变体。
- 当前助手执行环境CUDA不可用，正式60轮训练尚未运行；需在用户CUDA可用的WSL终端执行以上命令。

## 完成训练后的评估

先在同一39张val上对照旧模型，报告四类P/R、TP/FP/FN和置信度阈值；冻结阈值后在原test.JPG/test.MOV及旧flower3回归集上评估。测试素材已多次查看，仅作为回归，不宣称独立未知场景验收。任何一类P或R不超过90%即不判通过。val原兰花仅10个目标，统计量很小；无法保证新场景90%。保留旧模型，只有比较结果支持后才另行决定部署。
