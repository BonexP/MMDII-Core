# Dataset v0.2 稳健性复验计划

## 当前阶段

`d036a2b` 已把第一批 B0/E0/E1 聚合实验和训练主机环境输出纳入 Core 主线。该批次仍只有一个 seed、固定 20 epochs、固定 0.5 阈值，因此结果用于基线定位，不作为最终模型结论。

## 分阶段任务

1. **结果冻结与校验**：保留原始输出和报告；使用 `scripts/validate_tracked_outputs.py` 检查每次运行的 5 折、OOF 行数、唯一 sample_id、概率范围和 `.complete` 标记。
2. **随机性复验**：对 B0 与 E1c 使用 seed `7, 17, 27`，保持 release、分组折、窗口和指标协议不变，汇总 pooled OOF 均值和标准差。
3. **训练稳定性**：在固定 seed 子集上比较 10/20/40 epochs；再开启 `early_stopping_patience`。当前早停监控训练损失，只用于工程稳定性试验，不宣称为无偏超参选择。
4. **优化与正则化**：一次只改变一个因素，比较 AdamW/Adam、weight decay 和 gradient clipping；所有配置通过命令行覆盖并写入 `run_config.json`。
5. **阈值校准**：下一阶段增加只使用训练折预测的 per-class 阈值选择，再在外层 OOF 评估；禁止直接用 OOF 选择阈值。
6. **弱监督研究轴**：当前窗口继承焊缝标签且使用 MIL，已经属于焊缝级弱监督。后续可比较 attention 约束、top-k/ranking、正例-未标注学习和时序平滑，但必须与主基线分开注册实验。
7. **模型升级**：只有 B0/E1c 在多 seed 和阈值协议下稳定后，才替换 ModernTCN encoder（如 PatchTST-style），一次只改变 encoder。

## 解释边界

窗口数量增加不等于独立样本数量增加；统计显著性仍以焊缝和图片组为单位。attention 权重只能作为候选窗口排序证据，不能称为缺陷起止位置定位。
