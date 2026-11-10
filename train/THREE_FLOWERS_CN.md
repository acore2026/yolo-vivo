# 三类花卉迁移：芍药、百合、菊花

## 已完成

新数据 `train/datasets/flowers_three_v1/`：268张train（248张正图＋20张已审核背景裁图），39张val。训练框为芍药137、百合143、菊花117；验证框为32、10、33。排除52张只有桉树叶的原训练图。多盆同框图保留前三类坐标，删除桉树叶标签。所有原始四类文件、权重、输出保留。

类别为0芍药/potted peony、1百合/potted lily、2菊花/potted chrysanthemum。名称依照此前模型匹配结果和用户确认，不是植物学鉴定。源标签中原ID0/1/2不变，ID3移除。数据YAML、名称和检测提示词已同步。

`train/datasets/flowers_three_test_v1/`是原flower3的80张审核回归图转换副本，每类80个目标。图像不变，桉树叶框删除；桉树叶若被预测为其他花仍算FP。已有test.JPG/test.MOV审核框在配对评估时同样过滤第3类，保持旧三类框坐标不变。

`train/datasets/flowers_three_test2_samples/`新增12张均匀抽样原图PNG、YOLO标签、annotations.json、data.yaml、来源哈希、复核预览、验证报告。实际查看640×360原图及手工框叠图后记录assistant_visual，非human。35个目标（芍药11、百合12、菊花12）；近景第416帧只有百合/菊花，桉树叶为非目标，不作为第四类。原视频已在此前查看，不宣称新独立未知场景验收。

旧训练/验证/flower3标签复用已有审核记录，转换过程中没有重新声称全量逐图审核；conversion元数据记录来源清单及标签哈希、排除项。脚本核对保留坐标、源图哈希、图文配对、类别范围、视频隔离、审核背景空标签。背景裁图仍附原父图与视频追溯，部分父图被排除，但裁图来自原已审核纯背景，不将整盆桉树叶图变成空标签。

## 训练与测试流程

```bash
cd /home/huawei/VIVO/yolo-vivo
train/.venv/bin/python -u train/run_three_flower_workflow.py
```

仅检查：加 `--check-only`。无需重新执行数据转换脚本；目标目录已有文件时转换拒绝覆盖。

1. 数据preflight和CUDA实际计算检查。
2. 三类YOLO-World LoRA 1轮GPU smoke、权重重载核验。
3. 从原始models/yolov8s-worldv2.pt重新正式LoRA训练（60轮上限、patience15、rank8、lr0.0001），未从旧四类适配器继续训练。使用此前保留模型的保守训练增强，仅train增强；保持原基座/CLIP/BN冻结校验。
4. 新训练配套冻结CLIP线性分类器。三种花＋一个内部背景拒识输出，只有三类可以输出到画面；reject不是桉树叶类别。使用三类训练框、20张审核背景和可追溯的旧审核困难裁图。排除旧桉树叶裁图，不把它们重命名为背景；旧困难裁图仅选父图仍在train且哈希一致的条目。旧review状态显式复用，未伪称本轮新审核。
5. 分类器300轮上限，按val选epoch；仅val扫描固定全局阈值网格，冻结工作点。与旧v3的分类型阈值方案不完全相同，本轮属于新三类系统比较，不能把所有差异归因于移除桉树叶一个因素。
6. 完整测试test.JPG、test.MOV、test2.MOV；flower3回归；对照片及两视频固定审核抽样帧做新旧三类配对P/R比较。检测置信度不是准确率。
7. 生成三类“芍药、百合、菊花”介绍视频。逐字打印一次后保持，连续漏检10帧重置。转码H.264输出。

所有输出在新时间戳 `train/workflows/three_flowers_*`，主状态status.json，逐阶段*.log；classifier保存分类头/特征来源/阈值；test保存JSONL/完整视频/flower3_metrics.json/paired_sample_metrics.json/COMPARISON_CN.md；introductions保存介绍效果视频和状态日志。不会自动部署或替换旧摄像头服务。

## 实际验证和当前阻塞

当前助手进程CUDA不可用。正式GPU训练和新正式模型的测试尚未执行，需在GPU可用WSL终端运行上面命令。

实际完成的CPU验证：
- `train/runs/three_flowers_cpu_smoke_20261109`：从原始预训练基座实际训练1轮，fraction0.06、320像素、batch2；用完整39张val验证。冻结参数与BN不变、LoRA更新有限数值、适配器和合并权重重载推理通过。仅小集管线检查，不是正式模型。
- `train/runs/three_flowers_cascade_cpu_smoke_20261109`：用上述smoke适配器完成分类器训练/验证选点，实际验证特征抽取、拒识输出及三类阈值流程。开发指标未全部超过90%，不用于替换保留模型。
- `train/diagnostics/three_flower_pipeline_smoke_20261109_v2`：用真实存在的一张val图验证推理输出。第一次冒烟误写不存在的帧文件名而失败，保留调试输出；已补充输入文件preflight，第二次使用正确路径验证。
- 检查新三类介绍仅接受0/1/2，全文打印后保持；新字形字体子集覆盖验证通过。新字体许可沿用service/fonts/OFL.txt。

## 限制与后续判断

移除识别类别不等于已经解决混淆。混合图里的桉树叶是非目标背景，仍可能被误认为三类花；这会如实计入误检。物体和场景覆盖仍有限，测试均是已暴露回归，不能保证真实新场景P/R>90%。CPU冒烟不证明正式训练效果，正式测试前不宣布新模型优于旧模型。
