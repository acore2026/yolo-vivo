# 已执行：数据导出与独立环境准备

本文件记录用户确认后实际完成的工作，替代早期文档中的“尚未安装/尚未导出”状态。正式训练、在线增强、小批训练试跑均未执行。

## 数据

目录：`train/datasets/flowers_video_holdout/`

- 训练：flower1、flower4–7，280张，473框。
- 验证：flower2，39张，87框。
- 测试：flower3，80张，320框。
- 原始审核清单哈希保持不变。排除的模糊图没有导出。
- 按视频隔离，不随机拆相邻帧；保留原拍摄组记录。
- 显式策略为 `same_setup_video_holdout`，配置和数据YAML必须匹配。原有 `capture_group` 严格策略仍有效，测试确认它会拒绝同组跨集合；新策略会拒绝同一来源视频跨集合。
- 哈希、标签坐标回算、图片标签配对检查通过。每个集合覆盖四类。报告在数据目录 `validation_report.json`。
- 评测仅为同实物、同房间的视频留出测试，不能推断新场景/新实例泛化或旧类别能力保留。

## 环境

`train/.venv/` 为新建 Python 3.12 环境，以独立文件副本复用现有服务环境的包，再从参考训练环境补齐 CLIP、ftfy、regex、tqdm、wcwidth。不共享 site-packages，不修改服务环境，没有联网下载。

该方法复用了现有环境的86个包，不是最小依赖集合。包清单见 `requirements-local.lock.txt`，复制来源见 `environment_setup.json`。入口统一使用 `.venv/bin/python`，没有复制旧环境的CLI启动器。已检查代表包文件不是原环境的同一文件。

关键版本：PyTorch 2.14.1+cu130、Ultralytics 8.4.183、CLIP 1.0。`uv pip check` 检查86个包兼容性通过。

CLIP 权重已复制到 `train/.cache/weights/clip/ViT-B-32.pt`，哈希与参考权重及已核对的CLIP下载地址哈希一致。`runtime.py` 在模型入口设置本项目缓存并校验权重，自动安装关闭。CPU 四类文本编码得到有限的 4×512 特征，训练器可导入。报告见 `ENVIRONMENT_CHECK_REPORT.json`。

## 普通 WSL 终端验证 GPU

用户的普通终端已看到 RTX 5090；当前工具进程仍不可见。请在普通 WSL 终端运行以下命令，只进行GPU计算检查，不启动训练：

```bash
cd /home/huawei/VIVO/yolo-vivo
train/.venv/bin/python train/check_gpu.py
```

返回 `cuda_available: true` 和 `cuda_matmul: PASS` 才能确认新环境可用。`nvidia-smi` 可用本身不等于该 PyTorch 环境可执行 CUDA。

数据与配置预检查（已通过）：

```bash
train/.venv/bin/python train/train.py
```

后续获准训练时使用 `train/.venv/bin/python train/train.py --run`。运行前仍建议先进行小批训练链路检查，评估保存/验证路径与显存占用；本轮没有执行，也未声称已验证。配置为最多60轮、batch8、640分辨率、LoRA rank8。原类别回归评测、断点恢复、真实准确率及完整训练循环仍未验证。
