# 普通YOLO第三轮：背景负样本小批次对照

本轮完成审核、数据准备、校验与CPU冒烟；正式GPU训练未运行。旧权重、旧数据及部署保持不变。

## 实际检查发现

1. 实际逐图查看训练集中125张含桉树叶的原图/叠框配对缩略图（320×180，8张总览表）。未发现把整块地面标为桉树叶的明显系统性错误，因此保留现有标签；不是全分辨率逐像素重标，也未重新审核全部其他类别。125张都来自flower1、flower7的同套桌面实物，场景覆盖有限。
2. 从训练集父图裁出20张背景图，逐张按原生320/640×240像素查看：桌面、地砖、桌腿、墙面、椅背和杂物，无可见目标花卉或花盆残片。先pending，再记录assistant_visual接受为负样本；非human验收。没有使用模型的“无检测”作为空标签依据。
3. 用第二轮best及其固定验证阈值conf=0.05检查这20张背景图：1张发生桉树叶误检，置信度约0.484，来源flower1_f001933的地面/桌腿背景裁图。说明训练来源背景也能复现误检，但这不是独立评估；其余19张不能据此宣称模型已经学会排除背景，也不能把20张全部称为模型挖掘的困难负样本。

## 数据与追溯

- `train/datasets/flowers_reviewed_standard_v3/`：300张原正样本、20张新增背景裁图、39张原验证图；原339条记录和文件保持一致。训练仍为522个正目标。
- `train/datasets/flowers_standard_yolo_v3/`：框架使用的隔离副本，含data.yaml。
- `train/diagnostics/standard_yolo_v3_audit/`：audit_01..08原图/叠框配对预览、positive_audit_review.json、negative_01..03背景预览、negative_review.json、negative_pending.json、negative_v2_baseline.json、validation_report.json。
- manifest记录源视频哈希/帧号/FPS、父图ID与哈希、裁剪像素范围、尺寸、原图和空标签哈希、审核依据。背景裁图与父图始终同属train，不能计作新的独立拍摄帧。
- 仅使用flower1/4/5/6/7。flower2验证、flower3回归及test.JPG/test.MOV均未加入训练；原39张验证未扩大或调标签。
- 裁图筛选遇到大目标没有足够背景时跳过不合适区域，最后以negative_pending与negative_review清单为准。候选目录中保留失败尝试期间生成的少量未入清单裁图，它们未导出训练，不自动当作已审核数据，也未删除。

## 脚本与验证

`train/standard_yolo.py`增加严格背景裁图支持：必须配置显式允许、人工视觉审核状态明确无目标、train父图存在、父图哈希/帧号/来源一致、裁剪可按像素重建且与正目标框留20像素距离。未经审核空标签仍拒绝。全图与其裁图以不同样本身份追溯，视频分区规则保留。

已通过：原数据与新增数据哈希、图文配对与坐标格式、裁图像素重建、父来源隔离；未审核背景、错误父图哈希和越界裁剪三个故障用例均被正确拒绝。旧v2配置preflight通过。

CPU随机初始化1轮冒烟目录：`train/workflows/standard_yolo_v3_20261105_185201_257551/`。16张train中实际包含1张背景，8张val；数据加载、训练、验证、保存与重载推理成功。已实际查看train_batch0增强预览；此检查不代表准确率或正式训练。

## 正式训练

```bash
cd /home/huawei/VIVO/yolo-vivo
train/.venv/bin/python -u train/standard_yolo.py --config train/configs/flowers_standard_yolo_v3.yaml --run
```

配置：`train/configs/flowers_standard_yolo_v3.yaml`。从原始models/yolov8s.pt开始，不接训第二轮权重；与v2完全相同的60轮上限、patience20、640像素、batch8、seed42、scale0.5、mosaic0等超参数，仅改变训练数据。在线增强仍仅train使用。新增数据会改变优化步数与随机序列，因此是数据补充对照，不保证曲线逐步可比。

输出独立时间戳目录standard_yolo_v3_*：GPU smoke后重新加载预训练权重正式训练，仅val选择工作点，再冻结后做既有测试回归。旧模型不自动替换。当前助手环境CUDA不可用，需要在用户CUDA可用终端启动。

## 未完成与限制

没有新增实拍俯视正样本、新物体或独立场次背景；已有视频中可安全裁出的背景不能替代这些素材。此小批次优先检验误检抑制，不能保证解决flower3芍药漏检，不能保证四类P/R超过90%。没有修改原标签、安装依赖、执行新正式GPU训练或部署。已有测试素材多次暴露，仍只作回归，不作为未知场景验收。
