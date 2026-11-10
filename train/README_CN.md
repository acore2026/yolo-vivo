# 四类盆栽 YOLO-World LoRA 微调

**当前执行状态以 [READY_CN.md](READY_CN.md) 为准**：已获准导出280/39/80张数据并准备独立训练环境；CLIP已就绪、CPU文本编码和依赖校验通过。下文保留的第一阶段说明不再代表当前安装/导出状态。正式训练仍未启动。

后续检查更新：已完成 CPU 合成张量的实际模型零初始化、LoRA 合并及适配器加载测试，见 `LORA_CHECK_REPORT.json`；环境核验与新增独立素材计划见 `NEXT_STEPS_CN.md`。以下第一阶段描述中的“未运行数值测试”已由该报告更新，完整训练链路仍未执行。

当前阶段只交付脚本、配置与静态检查，没有安装依赖、生成增强图、启动训练或切换服务。参考来源：`/home/huawei/compute_network/yolo/train/README.md`、`yolo_world_lora.py`、`train.py`。LoRA Linear/Conv2d 实现由参考代码改编，Trainer 钩子按本机 Ultralytics 8.4.183 调整；未迁移关闭 HTTPS 验证的代码。

## 目标与能力边界

训练后诊断发现，初版 LoRA checkpoint 未保存 WorldModel 的重建结构。原始 `.pt` 模型的 SPPF 激活为 Identity，而 YAML 初始化的结构为 SiLU；旧 `model_io.py` 只在原模块树内替换 LoRA，造成推理结构不同于训练结构。已在 v2 checkpoint 中记录训练 YAML/nc/ch，并由加载器先按训练方式重建结构。旧 adapter v1 与旧 merged best.pt 仍保留不动；手工迁移检查点在 `diagnostics/adapter_best_architecture_v2.pt`，未来训练器保存 v2。详见 `diagnostics/DIAGNOSIS_CN.md`。当前验证集主要指标与旧 merged checkpoint 基本一致；花卉图片检测脚本之后应显式指定修正版适配器以对齐训练计算结构。

基础权重固定为 `models/yolov8s-worldv2.pt`，不覆盖。四类顺序为芍药、兰花、向日葵、桉树叶，文本提示分别为 `potted peony`、`potted orchid`、`potted sunflower`、`potted eucalyptus`。提示强调整盆目标，英文提示与中文显示名分离；提示质量仍需基线测试确认。

冻结基础参数、文本编码器和 BatchNorm 统计量，只训练指定视觉/融合层的 LoRA。低秩训练不能保证不遗忘。必须对原类别和花卉分别用有标注的独立数据做基线/微调对比。教师蒸馏与原类别数据回放本阶段未实现，待确定原类别数据与验收要求后单独设计。

推荐原模型保留原类别识别路径，LoRA 分支识别四类花。`predict.py` 分别输出两路带框结果和 JSONL，原类别路径不加载适配器。当前尚未实现实时服务接入、合并展示或跨分支去重，避免通用 `potted plant` 与具体花卉未经确认就互相抑制。需要时可按帧 index 合并业务展示。双路计算开销须实测。

## 文件

- `check_env.py`：只读检查包版本、基础模型 SHA256、CLIP 模块和 GPU 查询结果。
- `prepare_data.py`：校验审核清单；显式提供拍摄组分配后才导出新的训练目录。不抽帧、不修改原标注、不随机拆帧。
- `configs/flowers.yaml`：模型、四类提示、LoRA 参数与拟采用的在线增强配置。
- `train.py`：默认只检查；只有 `--run` 才进入训练。未配置数据、分组不合格、版本不匹配时拒绝训练。
- `lora.py` / `trainer.py`：适配器实现、参数冻结、保存和冻结状态检查。
- `model_io.py`：校验基础模型哈希和适配器张量名称后加载。
- `evaluate.py`：相同提示、阈值、分辨率下，对基础模型和适配器模型评测并输出差值。
- `predict.py`：原模型/花卉分支分别输出视频或图像结果。
- `export.py`：生成新的合并权重，不覆盖基础模型。

## 环境与安全默认值

建议独立训练虚拟环境，不直接更改现有摄像头服务环境。静态检查使用现有 `/home/huawei/VDF/yolo-pose/.venv/bin/python`。当前发现 Ultralytics 8.4.183，GPU 的 NVML 查询被系统阻止，CLIP 模块未找到。不能据此宣称 CUDA 训练可用。

`requirements-reference.txt` 只是已观察到的版本记录，不是已验证安装锁文件，不应直接用于升级服务环境。CLIP 的安装来源及权重缓存、CUDA 可用性、训练环境依赖兼容性需后续确认。脚本关闭 Ultralytics 自动安装，但 CLIP/框架仍可能尝试下载权重；正式运行前必须准备好依赖及缓存。本阶段没有下载或联网执行训练。

## 只读检查（本阶段允许）

在项目根目录运行：

```bash
/home/huawei/VDF/yolo-pose/.venv/bin/python train/check_env.py
/home/huawei/VDF/yolo-pose/.venv/bin/python train/prepare_data.py
/home/huawei/VDF/yolo-pose/.venv/bin/python train/train.py
```

当前数据为 399 张/880 框，拍摄组仅 `flower_same_setup_01`，全部 `review`。默认训练检查会明确报告数据未分配；这是防止泄漏的阻断，不是自动创建随机验证集。所有原图/标签均保留。

## 后续获准才执行：分组与训练

1. 提供独立场次验证/测试素材并标注，确认旧类别回归测试集合与可接受下降阈值。现有数据审核方式是助手缩略图逐图审核，建议先人工全分辨率抽检。
2. 合并来源清单后，以真实拍摄组编写 JSON 映射。例如 `{"capture_a":"train","capture_b":"val","capture_c":"test"}`；这些是格式示例，不是当前数据的真实组。不可人为把同组改名绕过隔离。
3. 执行 `prepare_data.py --manifest <批准的清单> --groups <分组JSON> --output <全新目录>`。输出目录必须不存在，数据 YAML 使用绝对路径，标签来自清单中的最终框。
4. 将配置的 `data` 设置为生成的 `data.yaml`。默认 rank=8、alpha=16、AdamW、lr=1e-4、640 分辨率、batch=8、最多60轮、patience=15；batch 待显存实测。
5. 显式执行 `python train/train.py --run`。增强只有进入训练时才在线执行，验证集无随机增强。温和 HSV、平移/缩放和水平翻转；mosaic/mixup/垂直翻转关闭。增强不会创造新的真实拍摄场景。

训练产物：`adapter_best.pt` / `adapter_last.pt` 包含独立 LoRA 张量、原模型哈希与层配置；框架的 `best.pt` / `last.pt` 为合并后的推理权重；`training_state_last.pt` 另存未合并模型/EMA/优化器/调度器/scaler及随机数状态。

**断点恢复尚未开放**：保留完整状态供后续实现与测试，不能直接把合并的 `last.pt` 当 LoRA 断点。当前训练入口没有 `--resume`，不声称已实现或验证恢复。混合精度、DDP、多GPU和不同 Ultralytics 版本也未验证。

## 后续获准才执行：评测、推理、导出

```bash
python train/evaluate.py --adapter <adapter_best.pt> --data <花卉独立测试.yaml> --output train/evaluations/flowers --split test
python train/evaluate.py --adapter <adapter_best.pt> --data <旧类别独立测试.yaml> --output train/evaluations/original --split test
python train/predict.py --adapter <adapter_best.pt> --source <本地视频> --old-prompts person chair bottle --output train/predictions/demo
python train/export.py --adapter <adapter_best.pt> --output train/exports/flowers_merged.pt
```

提示词顺序必须与各评测数据标签一致。`evaluate.py` 只评测，不证明测试集来源独立；独立性需来源清单审核。输出 JSON 报告 mAP 等指标及差值，不把无标注视频的检出频率当作准确率。旧类别列表与下降阈值尚待用户确定，当前没有自动“保留能力通过”的结论。

## 正式训练前需要的运行验证

静态检查不等于训练验证。后续在获准且依赖齐备时检查：零初始化适配器与原模型输出一致；适配器合并前后输出误差；优化器仅更新 LoRA；基础参数和 BN 统计不变；适配器保存/加载输出一致；一小批数据训练、保存和验证能贯通。训练器已加冻结检查，模型输出数值测试尚未执行。
