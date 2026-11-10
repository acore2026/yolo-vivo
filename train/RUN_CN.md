# 已批准的训练执行流程

用户已批准：1轮链路试跑，通过后最多60轮正式训练，最终测试。无需再确认其中小步骤。当前工具进程无法访问CUDA，未实际启动训练；用户普通WSL终端已通过CUDA矩阵运算检查。

在普通WSL终端运行一次，保持终端打开：

```bash
cd /home/huawei/VIVO/yolo-vivo
train/.venv/bin/python -u train/run_workflow.py
```

流程自动执行：

1. 检查GPU与数据，失败则不启动训练。
2. 用280张训练图进行1轮试跑，39张验证图参与验证。不读取80张测试图做推理。
3. 检查冻结参数和BatchNorm不变、LoRA参数有限且已更新、记录峰值显存、验证指标有限、保存的适配器和合并权重能够加载，并在一张验证图上推理。
4. 检查通过后，从原始基础模型重新开始最多60轮正式训练，patience=15。试跑权重不作为断点续训，因为当前断点恢复接口尚未开放。
5. 正式训练同样做产物检查，然后只在最终阶段对80张测试图运行基础模型/最佳LoRA模型对比。测试结果不用于选最佳轮次。

任何阶段失败立即停止后续步骤；不会绕过检查自动继续，也不会自动重复训练或覆盖历史实验。训练器冻结检查是模型参数完整性检查，不代表准确率或旧类别能力保留验收。原类别评测集合仍未提供。

每次运行生成独立目录：

- `train/workflows/<时间戳>/status.json`：当前阶段及完成记录。
- 同目录 `smoke.log`、`full.log`、`*_verify.log`、`final_test.log`：阶段日志；失败时另有 `error.txt`。
- 同目录 `smoke.yaml`、`full.yaml`：实际配置快照。
- 同目录 `test_evaluation/comparison.json`：最终基础模型/LoRA测试对比。
- `train/runs/flowers_<时间戳>_smoke/` 和 `_full/`：训练结果，含 `weights/adapter_best.pt`、`weights/best.pt` 等。

主配置 `train/configs/flowers.yaml`：640分辨率、batch8、LoRA rank8/alpha16、AdamW lr1e-4。在线增强只发生在训练阶段。

已完成脚本语法和CLI检查；当前工具内GPU前置检查明确失败，所以尚未验证完整训练循环、保存后推理或最终准确率。运行后的日志与报告才是实际完成依据。
