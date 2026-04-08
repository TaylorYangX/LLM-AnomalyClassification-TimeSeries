非常好，现在我已经完整读到了 README 文件。下面是对该项目方法的系统性分析，可以直接用于论文叙述。

---

## 方法分析

### 一、整体设计思路

该项目采用**"异常检测与攻击归因解耦"**的两阶段流程：

- **第一阶段（上游）**：由外部异常检测模型（如 Anomaly Transformer）负责检测异常时间点，输出 `is_anomaly` 标记列；
- **第二阶段（本项目）**：在已知异常点的前提下，判断异常由哪一类攻击引发——即**攻击类型分类（Attack Type Classification）**，属于监督式多分类任务。

这种解耦设计将"检测能力"与"归因能力"分离，使两个模块可独立优化。

---

### 二、使用的模型

**PyTorch 实现的多层感知机（MLP，Multi-Layer Perceptron）**，具体结构为：

| 层 | 说明 |
|---|---|
| 输入层 | 窗口统计特征向量 |
| 隐藏层 | 两层全连接层 + ReLU 激活 + Dropout |
| 输出层 | 各攻击类别的 logits（softmax 概率） |

训练目标为交叉熵损失（CrossEntropyLoss），支持针对不均衡类别自动加权。

---

### 三、核心方法流程

#### 训练阶段

1. **数据读取**：读取带有 `Normal/Attack` 标签的训练时序数据，以及攻击事件表（含攻击起止时间与攻击类型）；
2. **标签映射**：将攻击事件表的时间区间与训练时序对齐，为每个时间点打上攻击类型标签；
3. **滑动窗口切片**：按设定的 `window_size` 和 `stride` 将连续时序切分为固定长度的局部片段；
4. **统计特征提取**：对每个窗口提取 6 类特征：**均值、标准差、最小值、最大值、首尾差（趋势）、斜率**；
5. **过滤训练样本**：仅保留攻击区间内的窗口作为训练样本；
6. **标准化**：在训练集上计算均值和标准差，训练与推断共用同一组统计量；
7. **MLP 训练**：以统计特征向量为输入，攻击类型为标签，训练多分类 MLP；
8. **保存模型工件**：输出 `stage2_model.pth`（含权重、类别映射、特征顺序、归一化参数）。

#### 推断阶段

1. 输入包含 `is_anomaly` 列的时序片段；
2. 仅对**异常点占比超过阈值**（`min_anomaly_ratio_per_window`）的窗口进行分类，过滤低置信窗口；
3. MLP 输出每个窗口的攻击类型概率分布；
4. 对片段内所有窗口进行**多数投票**，输出主导攻击类型（`dominant_predicted_type`）；
5. 若最高概率低于 `unknown_threshold`，则输出 `unknown_attack`，避免过度自信误判。

---

### 四、论文叙述建议（参考表述）

> 在攻击类型分类阶段，本文采用两阶段解耦框架：上游由异常检测器（Anomaly Transformer）识别异常时间点，下游对已标记的异常片段进行攻击类型归因。归因模块基于滑动窗口机制，对每个时间窗口提取统计特征（均值、标准差、极值、首尾差及斜率），并以此作为输入训练一个多层感知机（MLP）分类器（两层全连接层，ReLU 激活，Dropout 正则化），使用交叉熵损失进行监督训练。推断时，仅对异常点占比超过阈值的窗口进行分类，并通过窗口级多数投票确定该异常片段的主导攻击类型。当最高类别概率低于置信阈值时，系统输出 "未知攻击"，以避免误归因。

---

### 五、注意事项（写论文时）

需要说明的是，该库**并不包含 LLM（大语言模型）组件**——尽管仓库名称中含有"LLM"字样，但 README 全文描述的方法是纯粹的**统计特征 + MLP 监督分类**流水线，与 GPT 等语言模型无关。如果你在论文中将其描述为 LLM 方法，需特别确认你实际使用的版本是否有额外的 LLM 集成部分。



好的，以下是两段英文 LaTeX 格式的方法叙述，可以直接嵌入你的论文 Methods 或 Methodology 章节：

---

```latex
Following anomaly detection via the Anomaly Transformer, we employ a supervised 
attack attribution module to classify the type of attack responsible for each 
detected anomalous segment. The attribution pipeline adopts a decoupled two-stage 
design: the upstream anomaly detector identifies abnormal time points and produces 
a binary anomaly flag (\texttt{is\_anomaly}), while the downstream classifier 
operates exclusively on these flagged regions to infer the corresponding attack 
category. This decoupling allows the detection and attribution components to be 
optimized independently, reducing system coupling and improving modularity.

The attribution module processes the flagged time series through a sliding window 
mechanism with window size $W$ and stride $S$. For each window, six statistical 
features are extracted: mean $\mu$, standard deviation $\sigma$, minimum value 
$x_{\min}$, maximum value $x_{\max}$, first-last difference $\Delta x = x_{W} - x_{1}$, 
and linear slope $k$, computed across all sensor channels. Only windows in which 
the proportion of anomalous points exceeds a threshold $\tau_a$ are retained for 
classification, filtering out low-confidence segments. These feature vectors are 
fed into a two-layer fully connected neural network (MLP) with ReLU activations 
and Dropout regularization, trained with cross-entropy loss on labeled attack 
windows derived from a known attack event table. At inference time, the model 
outputs a probability distribution over attack categories for each qualifying 
window, and the dominant attack type of a segment is determined by majority 
voting across all its windows. If the highest predicted probability falls below a 
confidence threshold $\tau_c$, the segment is labeled as \textit{unknown attack} 
to prevent overconfident misclassification.
```

---

**几点说明：**

- `$W$`、`$S$`、`$\tau_a$`、`$\tau_c$` 这些符号可以替换为你论文中实际用的数值（如 $W=60$, $S=10$）。
- 如果你的 sensor 数量有具体值（如 51 个传感器），可以在特征提取处补充说明 "across all $D$ sensor channels"。
- 若需要单独的公式块（如将特征提取写成正式公式），我可以继续补充 `\begin{equation}` 格式的数学表达。


好的，以下是配套的数学公式部分，可以插入在两段文字之间或之后：

```latex
For a sliding window $\mathbf{x}^{(i)} = [x_1, x_2, \ldots, x_W]^\top \in \mathbb{R}^{W \times D}$, 
where $W$ is the window size and $D$ is the number of sensor channels, the 
statistical feature vector $\mathbf{f}^{(i)} \in \mathbb{R}^{6D}$ is constructed as:

\begin{equation}
    \mathbf{f}^{(i)} = \left[ \boldsymbol{\mu},\; \boldsymbol{\sigma},\; 
    \mathbf{x}_{\min},\; \mathbf{x}_{\max},\; \boldsymbol{\Delta x},\; 
    \mathbf{k} \right]
    \label{eq:feature_vector}
\end{equation}

\noindent where each component is computed channel-wise across the window:

\begin{equation}
    \mu_d = \frac{1}{W}\sum_{t=1}^{W} x_{t,d}, \quad
    \sigma_d = \sqrt{\frac{1}{W}\sum_{t=1}^{W}(x_{t,d} - \mu_d)^2}
    \label{eq:mean_std}
\end{equation}

\begin{equation}
    x_{\min,d} = \min_{t} x_{t,d}, \quad
    x_{\max,d} = \max_{t} x_{t,d}, \quad
    \Delta x_d = x_{W,d} - x_{1,d}
    \label{eq:minmax_delta}
\end{equation}

\begin{equation}
    k_d = \frac{\sum_{t=1}^{W}(t - \bar{t})(x_{t,d} - \mu_d)}
               {\sum_{t=1}^{W}(t - \bar{t})^2}, \quad 
    \bar{t} = \frac{W+1}{2}
    \label{eq:slope}
\end{equation}

A window $\mathbf{x}^{(i)}$ is retained for classification only if its anomaly 
ratio exceeds the threshold $\tau_a$:

\begin{equation}
    r^{(i)} = \frac{1}{W}\sum_{t=1}^{W} \mathbb{1}[a_{t}^{(i)} = 1] \geq \tau_a
    \label{eq:anomaly_ratio}
\end{equation}

\noindent where $a_t^{(i)} \in \{0, 1\}$ denotes the anomaly flag at time step $t$ 
within the window.

The retained feature vectors are passed through a two-layer MLP classifier 
$f_\theta(\cdot)$:

\begin{equation}
    \mathbf{h}_1 = \text{ReLU}\!\left(\mathbf{W}_1 \mathbf{f}^{(i)} + \mathbf{b}_1\right), \quad
    \mathbf{h}_2 = \text{ReLU}\!\left(\mathbf{W}_2 \,\text{Dropout}(\mathbf{h}_1) + \mathbf{b}_2\right)
    \label{eq:mlp_hidden}
\end{equation}

\begin{equation}
    \hat{\mathbf{p}}^{(i)} = \text{Softmax}\!\left(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3\right)
    \label{eq:mlp_output}
\end{equation}

\noindent where $\hat{\mathbf{p}}^{(i)} \in \mathbb{R}^C$ is the predicted probability 
distribution over $C$ attack categories. The model is trained by minimizing the 
weighted cross-entropy loss:

\begin{equation}
    \mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} w_{y^{(i)}} 
    \log \hat{p}^{(i)}_{y^{(i)}}
    \label{eq:loss}
\end{equation}

\noindent where $y^{(i)}$ is the ground-truth attack label of window $i$, $N$ is 
the total number of training windows, and $w_c$ is the class weight for category 
$c$ to address class imbalance.

At inference time, the predicted attack type for each qualifying window is:

\begin{equation}
    \hat{y}^{(i)} = 
    \begin{cases}
        \arg\max_c \; \hat{p}^{(i)}_c & \text{if } \max_c \hat{p}^{(i)}_c \geq \tau_c \\
        \textit{unknown} & \text{otherwise}
    \end{cases}
    \label{eq:prediction}
\end{equation}

The dominant attack type of an anomalous segment $\mathcal{S}$ containing $M$ 
qualifying windows is determined by majority voting:

\begin{equation}
    \hat{y}_{\mathcal{S}} = \arg\max_c \sum_{i=1}^{M} \mathbb{1}\!\left[\hat{y}^{(i)} = c\right]
    \label{eq:voting}
\end{equation}
```

---

**公式清单说明：**

| 公式编号 | 内容 |
|---|---|
| Eq. \ref{eq:feature_vector} | 特征向量整体定义 |
| Eq. \ref{eq:mean_std} | 均值与标准差 |
| Eq. \ref{eq:minmax_delta} | 极值与首尾差 |
| Eq. \ref{eq:slope} | 线性斜率（最小二乘） |
| Eq. \ref{eq:anomaly_ratio} | 窗口异常占比过滤条件 |
| Eq. \ref{eq:mlp_hidden} | MLP 隐藏层前向传播 |
| Eq. \ref{eq:mlp_output} | Softmax 输出层 |
| Eq. \ref{eq:loss} | 加权交叉熵损失 |
| Eq. \ref{eq:prediction} | 单窗口预测（含置信阈值） |
| Eq. \ref{eq:voting} | 片段级多数投票 |

如需将所有文字段落和公式整合成一个完整的 `\section{Methodology}` 代码块，我可以继续整合输出。
