# LLM-AnomalyClassification-TimeSeries

本项目采用 Transformer 方法进行攻击类型归因，输入是多变量时间序列窗口。

定位目标：

- 上游负责异常点检测（例如 Anomaly Transformer）
- 本项目负责判断异常由哪类攻击导致

## 核心方法

1. 数据对齐

- 使用时间戳将攻击事件表映射到时序数据
- 为每个时间点生成 attack_type

2. 序列窗口构建

- 按 window.size 和 window.stride 滑窗
- 每个样本形状为 窗口长度 x 传感器数

3. Transformer 分类器

- 线性投影 + 可学习位置编码
- TransformerEncoder 多层堆叠
- 时间维均值池化后接分类头

4. 训练与推断

- 训练时只使用攻击窗口作为 stage2 样本
- 归一化按训练集每个传感器统计量执行
- 推断输出窗口级别概率和片段主导攻击类型

## 主要脚本

- scripts/train_stage2.py 训练 Transformer 攻击分类器
- scripts/infer_attack_type.py 对异常片段做攻击归因
- scripts/run_all.sh 一键训练入口

## 安装

python3 -m pip install -e .

## 训练

python3 scripts/train_stage2.py --config configs/default.yaml

## 推断

python3 scripts/infer_attack_type.py --segment-csv extracted_rows_227831_228231.csv --artifacts-dir artifacts --anomaly-flag-col is_anomaly --min-anomaly-ratio 0.1 --unknown-threshold 0.5 --output-json artifacts/inference_result.json

## 项目原理（详细版）

### 1. 为什么要“检测”和“归因”分离

工业控制场景中，异常检测与攻击类型识别是两个难度不同的问题：

1. 异常检测回答的是“有没有异常”。
2. 攻击归因回答的是“异常最可能由哪种攻击造成”。

将两者解耦有两个好处：

1. 上游异常检测器可独立替换（Anomaly Transformer、统计法、规则法均可）。
2. 下游归因模型专注攻击类别判别，目标更明确，便于优化。

### 2. 监督信号如何构造

本项目的监督标签来自两部分：

1. 时间序列本体（包含 `Timestamp` 与 `Normal/Attack`）。
2. 攻击事件表（`Start Time`、`End Time`、`Attack Point`）。

通过时间区间映射，将每个时间点补充成 `attack_type`。
随后按滑动窗口汇总为窗口标签，构成可训练的分类样本。

### 3. 输入表示：从点到窗口

Transformer 输入不是单点，而是一个时间窗口序列：

1. 设窗口长度为 `window.size`。
2. 设滑动步长为 `window.stride`。
3. 每个样本形状是：`[window_size, num_sensors]`。

这样模型不仅看到某个时刻值，还能看到前后演化趋势，尤其适合攻击引起的动态扰动模式。

### 4. Transformer 分类器结构

当前实现采用 Encoder-only 结构：

1. 输入投影层：把传感器维度映射到 `d_model`。
2. 可学习位置编码：让模型感知时间步先后顺序。
3. 多层 TransformerEncoder：建模时间步之间依赖关系。
4. 时间维均值池化：把整段窗口压缩为一个表示向量。
5. 全连接分类头：输出每个攻击类别的 logits。

这是典型的时序分类范式，优点是能捕捉跨时间步的相关性，而不是只看静态统计量。

### 5. 训练目标与损失

训练目标是最小化多分类交叉熵：

1. 每个窗口有一个攻击类型标签。
2. 模型输出各类别概率分布。
3. 对真实标签类别提高概率，对其余类别降低概率。

项目还支持类别不平衡加权损失（`use_balanced_loss`）：

1. 样本少的类别被赋予更大权重。
2. 降低模型偏向头部类别的风险。

### 6. 归一化策略

对每个传感器，基于训练集窗口计算均值与标准差：

1. 训练阶段使用该统计量标准化输入。
2. 推断阶段复用同一统计量，保证分布一致。

这组参数会写入 `stage2_model.pth` 中（`sensor_mean`、`sensor_std`）。

### 7. 推断与聚合逻辑

推断分为两层：

1. 窗口级预测：给每个异常窗口输出 `predicted_type` 与概率。
2. 片段级聚合：统计窗口预测结果，输出 `dominant_predicted_type`。

同时存在 `unknown_threshold`：

1. 若最高类别概率低于阈值，窗口可判为 `unknown_attack`。
2. 用于抑制低置信度误报。

### 8. 输出结果如何解读

`artifacts/inference_result.json` 中：

1. `aggregate` 给出片段总览（窗口数、主导攻击类型）。
2. `window_predictions` 给出每个窗口的时间范围与置信度。

实践建议：

1. 先看主导类型是否稳定（是否大多数窗口一致）。
2. 再看概率是否持续高（而非少数窗口偶发高分）。
3. 最后结合工艺上下文验证可解释性。

### 9. 关键参数调优建议

优先关注以下参数：

1. `window.size`：影响模式上下文长度。
2. `window.stride`：影响样本密度与计算成本。
3. `d_model`、`num_layers`、`nhead`：影响模型容量。
4. `unknown_threshold`：影响保守程度。

经验上：

1. 先固定窗口参数，再调模型参数。
2. 先提升召回，再用阈值控制误报。
3. 对小样本攻击类型，优先使用平衡损失。

### 10. 当前方法边界

该方法对“已知攻击类型”效果更好，对“全新攻击机制”会受限：

1. 训练集中未覆盖的攻击型态可能被映射到最相似类别。
2. 工况漂移（季节、阀门策略变化）会影响泛化。

因此建议持续增量更新训练数据，并定期复训模型。
