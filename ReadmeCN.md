# LLM-AnomalyClassification-TimeSeries

用于 ICS（工业控制系统）时间序列攻击分析的攻击类型分类流程。

你已经用外部异常检测器（例如 Anomaly Transformer）得到异常点，本项目只负责判断这些异常由哪一类攻击导致。

本项目专为 SWAT/WADI 类型的数据集设计，此类数据集通常包含：

- 包含传感器数值和二元标签（`正常` vs `攻击`）的时间序列数据行。
- 包含攻击类型及对应时间区间的攻击事件表（支持 csv 或 xlsx）。

## 1) 预期数据格式

### `train_timeseries.csv`

所需列：

- `Timestamp`：可解析的时间日期格式（本项目已按你当前文件适配）
- `Normal/Attack`：攻击标签列（字符串 `Normal`/`Attack`）
- 数值型传感器列（所有其他数值列均被视为特征）

你当前的 `train_timeseries.csv` 具有 SWAT 常见格式：

- 第一行是 `Unnamed:*` 占位列名。
- 第二行才是真实列名（例如 `Timestamp`、`FIT101`、`LIT101`、`Normal/Attack`）。

代码已自动适配这种结构：当检测到第一行表头不匹配时，会自动按第二行作为表头重新读取。

### 攻击事件文件（csv/xlsx）

所需列：

- `start_time`：可解析的时间日期格式
- `end_time`：可解析的时间日期格式
- `attack_type`：字符串格式的类别名称

该文件用于将已知的攻击时间区间映射至对应的攻击类型。

## 2) 项目结构

- `configs/default.yaml`：主配置文件
- `scripts/train_stage2.py`：训练攻击类型分类器
- `scripts/infer_attack_type.py`：根据异常点对攻击类型进行推断
- `artifacts/`：保存的模型文件与评估指标

## 3) 安装指南

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 4) 配置路径

编辑 `configs/default.yaml` 文件：

- `data.train_csv`
- `data.attack_events_file`（可直接配置为 `List_of_attacks_Final.xlsx`）
- `data.attack_start_col`、`data.attack_end_col`、`data.attack_type_col`
- `inference.anomaly_flag_col`（异常检测器输出的异常标记列）

对于 `List_of_attacks_Final.xlsx`，默认已配置为：

- `attack_start_col: "Start Time"`
- `attack_end_col: "End Time"`
- `attack_type_col: "Attack Point"`

## 5) 训练

```bash
python scripts/train_stage2.py --config configs/default.yaml
```

或使用辅助脚本：

```bash
bash scripts/run_all.sh configs/default.yaml
```

输出文件位于 `artifacts/` 目录下：

- `stage2_model.pth`
- `pipeline_meta.json`
- `stage2_metrics.json`

## 6) 方法说明（项目核心思路）

本项目采用“异常检测与攻击归因解耦”的设计：

1. 异常检测由外部模型完成（你当前使用 Anomaly Transformer）。
2. 本项目只做攻击归因：给定异常点，判断更可能是哪一类攻击导致。

训练阶段具体流程：

1. 读取训练时序数据（`train_timeseries.csv`）。
2. 读取攻击事件表（`List_of_attacks_Final.xlsx`），从 `Start Time` 到 `End Time` 的区间映射到 `Attack Point`。
3. 将时序数据按滑动窗口切片（`window.size` 与 `window.stride`）。
4. 对每个窗口提取统计特征：均值、标准差、最小值、最大值、首尾差、斜率。
5. 仅保留攻击窗口用于训练攻击分类器（PyTorch MLP）。
6. 训练完成后保存 `stage2_model.pth`（包含权重、类别映射、特征顺序、归一化参数）。

推断阶段具体流程：

1. 输入包含异常点标记的片段 CSV（`is_anomaly` 列默认 0/1）。
2. 仅对异常占比达到阈值的窗口进行分类（`min_anomaly_ratio_per_window`）。
3. 使用 `.pth` 模型输出每个窗口的攻击类型概率。
4. 按窗口投票得到主导攻击类型 `dominant_predicted_type`。

## 7) 关键参数建议

- `window.size`：建议先用 60；若攻击影响较慢可增大到 120。
- `window.stride`：建议 10 或 20；越小越精细但计算越多。
- `inference.min_anomaly_ratio_per_window`：
	- 异常点稀疏时可降到 0.05。
	- 希望更严格可升到 0.2。
- `inference.unknown_threshold`：
	- 当最高类别概率低于该值时，输出 `unknown_attack`。

## 8) 与你当前数据对应的默认配置

`configs/default.yaml` 已默认设置为：

- `data.train_csv: "train_timeseries.csv"`
- `data.timestamp_col: "Timestamp"`
- `data.binary_label_col: "Normal/Attack"`
- `data.binary_attack_value: "Attack"`
- `data.normal_label_name: "Normal"`

如果你后续换成其他数据文件，只需修改以上字段即可。

## 9) 推断异常片段的攻击类型

输入的片段 CSV 文件应包含：

- `timestamp` （时间戳）
- 与训练集相同的传感器列
- 异常标记列（默认 `is_anomaly`，取值 0/1）

运行命令：

```bash
python scripts/infer_attack_type.py \
--segment-csv data/raw/anomalous_segment.csv \
--artifacts-dir artifacts \
--anomaly-flag-col is_anomaly \
--min-anomaly-ratio 0.1 \
--unknown-threshold 0.5 \
--output-json artifacts/inference_result.json
```

输出结果包含：

- 综合判定结果 (`dominant_predicted_type`)
- 仅异常窗口的预测结果及概率值

## 10) 注意事项

- 如果时间戳与事件标签存在偏移，请在训练前将其对齐。
- 若存在类别不平衡问题，请调整类别权重（class weights）及阈值。
- 在生产环境中应用时，请针对不同攻击类型，验证事件层面的各项指标及混淆矩阵。

## 11) 底层原理详解（为什么这个方案可行）

本项目的核心目标不是“检测是否异常”，而是“解释异常由谁造成”，因此采用了监督式攻击归因思想。

1. 监督信号来源
- 通过攻击事件表（开始时间、结束时间、攻击点）为时间序列打上攻击类型标签。
- 通过 `Normal/Attack` 或外部异常检测器输出，确定哪些时间点属于异常范围。

2. 时序到样本的映射
- 原始数据是连续时间序列，模型训练需要固定维度样本。
- 使用滑动窗口将序列切分为大量局部片段，每个窗口对应一个训练样本。

3. 攻击表征学习
- 每个窗口提取统计与趋势特征（均值、标准差、极值、首尾差、斜率）。
- 这些特征近似描述了物理过程在窗口内的“状态”和“变化方向”，可反映攻击模式。

4. 分类决策逻辑
- 模型输出每类攻击的概率分布。
- 使用最大概率类别作为窗口攻击类型预测。
- 当最高概率低于阈值时输出 `unknown_attack`，避免过度自信误判。

5. 片段级归因
- 单窗口预测可能有噪声，因此对片段内多个窗口进行投票或统计。
- 最终输出 `dominant_predicted_type` 作为片段主导攻击类型。

## 12) 工具链与方法清单

本项目用到的关键工具和职责如下。

1. Python 3
- 项目运行环境与脚本执行入口。

2. PyTorch
- 实现攻击类型分类器（MLP）。
- 使用交叉熵损失进行多分类训练。
- 模型以 `.pth` 格式保存（权重、类别映射、归一化参数等）。

3. pandas
- 读取训练数据、攻击事件文件（csv/xlsx）。
- 时间字段解析、列名清洗、窗口数据组织。

4. NumPy
- 特征矩阵处理、标准化、概率后处理。

5. scikit-learn
- 评估指标计算（分类报告、混淆矩阵）。

6. openpyxl
- 支持读取 `List_of_attacks_Final.xlsx`。

7. YAML 配置管理
- 在 `configs/default.yaml` 中统一管理路径、窗口参数、训练参数、推断阈值。

## 13) 模型与训练机制（PyTorch MLP）

1. 模型结构
- 输入层：窗口特征向量。
- 隐层：两层全连接 + ReLU + Dropout。
- 输出层：攻击类别 logits。

2. 训练目标
- 损失函数：CrossEntropyLoss。
- 可选类别平衡权重：针对不均衡攻击类别自动加权。

3. 标准化策略
- 在训练集上计算 `x_mean` 和 `x_std`。
- 训练和推断均使用同一组统计量，避免分布漂移。

4. 可复现性
- 通过随机种子固定 `random_state`，减少重复训练波动。

5. 工程化产物
- `stage2_model.pth` 包含：
- `state_dict`（模型权重）
- `classes`（类别名顺序）
- `feature_columns`（特征顺序）
- `x_mean`、`x_std`（标准化参数）
- `model_params`（网络结构参数）

## 14) 端到端详细工作流

以下流程可直接对应你的工程脚本。

1. 数据准备
- 准备训练集 `train_timeseries.csv`。
- 准备攻击事件表 `List_of_attacks_Final.xlsx`。
- 若使用外部异常检测，准备带 `is_anomaly` 的片段 CSV。

2. 配置校准
- 在 `configs/default.yaml` 填写数据路径与列名映射。
- 设定窗口参数（`size`、`stride`）。
- 设定训练参数（`epochs`、`learning_rate`、`dropout` 等）。

3. 训练阶段
- 执行 `scripts/train_stage2.py`。
- 读取并清洗数据。
- 按事件区间映射攻击类型。
- 构造滑窗特征并过滤训练窗口。
- 训练 MLP 并保存 `.pth` 模型。
- 生成评估结果 `stage2_metrics.json`。

4. 推断阶段
- 执行 `scripts/infer_attack_type.py`。
- 读取 `.pth` 模型与 `pipeline_meta.json`。
- 对输入片段进行窗口特征提取与标准化。
- 输出每窗口攻击概率、预测类别、片段主导攻击类型。

5. 结果解读
- 若 `dominant_predicted_type` 明确且概率较高，可作为主判定。
- 若大量窗口为 `unknown_attack`，说明该片段可能是未见攻击模式或数据分布偏移。
- 建议结合过程变量趋势和工艺知识做最终确认。

## 15) 与异常检测模块的协同方式

本项目可与任意异常检测器组合，推荐协同方式如下。

1. 上游异常检测器负责“找出异常时段”。
2. 下游本项目负责“解释异常类型”。
3. 上下游通过统一字段衔接。
- 时间戳列：`Timestamp`
- 异常列：`is_anomaly`（0/1）

这样可以把“检测能力”和“归因能力”分离优化，降低系统耦合度，便于后续升级。
