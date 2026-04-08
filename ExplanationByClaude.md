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
