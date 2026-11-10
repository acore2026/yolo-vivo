# 新图片与视频检测

运行（自动选择CUDA，不可用则CPU）：

```bash
cd /home/huawei/VIVO/yolo-vivo
train/.venv/bin/python train/test_flowers.py
```

默认输入 `test_videos/test.JPG` 和 `test_videos/test.MOV`，加载 `adapter_best.pt` 与原始基础模型，提示词依次为 potted peony/orchid/sunflower/eucalyptus。阈值0.25、IoU0.7、imgsz640。输出完整尺寸叠框JPG、无音频MP4、逐帧JSONL与summary.json。每次默认生成新的时间戳目录；指定已存在的输出目录会拒绝覆盖。

可用参数：`--sources <图片> <视频>`、`--device 0`、`--conf 0.25`、`--imgsz 640`、`--output <新目录>`。未启用类别无关NMS，保留多类别重叠预测以呈现真实错误。视频逐帧处理，不抽帧。

本次结果目录 `train/predictions/test_flowers_20261103`：图片完成，视频1054帧全部完成。工具环境CPU执行；没有修改模型或训练。另使用ffmpeg将原输出转为H.264 `02_test_h264.mp4` 以便播放，内容与原MP4相同。

图片视觉复核：兰花和桉树叶有正确框；向日葵被预测为桉树叶；芍药位置同时输出芍药、兰花、桉树叶，存在类别冲突。不能说四类都已正确稳定识别。

视频输出芍药992帧、兰花935帧、向日葵8帧、桉树叶1051帧。这是模型产生该类别预测的帧数，不是人工验证的正确帧数，也不是召回率。查看向日葵最高分所在帧，仍存在与桉树叶重叠预测。视频未做逐帧人工标注，不能计算真实精度/召回率。
