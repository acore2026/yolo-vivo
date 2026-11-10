# 花卉增强对照实验

配置：`train/configs/flowers_augmented.yaml`。沿用原始训练/验证/测试划分、LoRA 配置、学习率、60 epoch 上限和 patience=15，从原始底模重新训练，不续训旧 adapter。

训练集在线增强：旋转 ±10°，平移 10%，缩放 30%，亮度扰动 0.3，水平翻转 50%，Mosaic 概率 20%（最后 10 轮关闭；提前停止可能在关闭前发生）。色相 0.01、饱和度 0.2，MixUp/上下翻转关闭。模糊、噪声、JPEG 各独立 10% 概率，分别使用 3×3 Gaussian、标准差 1–4、质量 75–95。使用现有 OpenCV/NumPy，无新增依赖。

`flower_augmentation.py` 使用官方几何增强及框同步逻辑，仅加入不改变坐标的拍摄退化。自定义 Dataset 在 close_mosaic 重建流程后保留这些退化；验证走原始非增强 Dataset。

已完成：数据 preflight；560 个增强样本数值/类别文本检查；关闭 Mosaic 后退化仍存在；非增强模式无退化；源 manifest 哈希不变；Python 编译。预览为 `train/diagnostics/augmentation_preview_v3/contact_sheet.jpg` 和 14 张独立图，检查报告为同目录 `report.json`。已实际查看 14 张缩略图及单张 sample_100；这不是全部样本人工审核。Mosaic 可产生边缘截断。

预览脚本前两次运行分别发现并修复了预览器对文本结构和 Dataset 属性的错误假设，v1/v2 为未完成调试输出；最终采用 v3。未修改原始标签。

工具进程 CUDA 不可用，训练启动已在 GPU 检查处停止，没有开始 smoke/full。普通 WSL 终端执行：

```bash
cd /home/huawei/VIVO/yolo-vivo
train/.venv/bin/python -u train/run_workflow.py --config train/configs/flowers_augmented.yaml --eval-split val
```

自动执行 GPU/数据检查、1 epoch smoke、产物核验、重新开始正式训练、核验和验证集评估。每次新建时间戳目录，保留旧结果。评估输出逐类 P/R/AP，并默认在以上命令中使用 val，避免反复用 test 调参。

对照参考修复结构后的旧模型：验证 AP50 0.69366；逐类 AP50 芍药约 0.966、兰花 0.161、向日葵 0.762、桉树叶 0.885（详见 diagnostics/DIAGNOSIS_CN.md）。新评估中的 base 指未微调底模，并非上一轮模型。P/R 是框架选取的工作点，不等于固定部署阈值下的最终验收。

增强收益尚未测得，不能据此声称达到 90%。最终需在验证集选定阈值，再报告每类 precision/recall；同一套实物、同一场景的视频划分仍不能证明独立场景泛化或原类别能力保留。
