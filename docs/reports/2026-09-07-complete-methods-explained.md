# MMDII 方法与实验工作完整说明

从问题到结果，从传统方法到深度学习方法，本文档尝试把整个工作流程和实验设计的思路、代码实现和结果分析完整地串起来。希望读者在阅读后能对现有的研究方法有一个清晰的整体理解。

## 1. 我们要做什么：我们要解决的是什么问题？

我们的目标是做一个**焊缝级缺陷识别系统**。输入一条焊缝在加工过程中的力信号，输出它是否出现过一种或多种缺陷。目前正式训练的三个目标是：

- `flash`：飞边/飞溅类问题；
- `blur`：另一类图像或焊缝缺陷标签；
- `tunnel`：隧道缺陷。

这里没有`pore`，其仍保留在数据发布的元数据中，因为正例只来自一个图片组，实在不太好下手，所以暂时不作为训练目标。

每条样本有三条同步力传感器通道：前向力 `af`、侧向力 `sf` 和轴向力 `axialf`，另外还有时间戳、焊缝 ID、图片组和人工确认的标签。

数据集方面，Dataset v0.2.1 的主协议把 101 条焊缝作为独立观测，按焊缝进行 5 折交叉验证；同时保留按图片来源分折的对照协议。也就是说，主结果回答“模型能否识别新焊缝”，对照结果回答“模型能否泛化到未见过的图片来源”。

这三条力信号属于**多变量（或多通道）时间序列**，不等于严格意义上的多模态数据。只有把图片、力信号和结构化工艺参数分别编码后再融合，才更接近通常所说的多模态学习。当前的人工缺陷标签是监督信号，不是输入模态。

标签只告诉我们“整条焊缝是否出现过某种缺陷”，没有告诉我们缺陷在时间轴上的起点和终点。因此，窗口实验属于弱监督的多实例学习（Multiple Instance Learning, MIL）：一条焊缝是一个 bag，切出的多个时间窗口是 instances，标签只在 bag 这一层提供。窗口继承焊缝标签，并不表示每个窗口都真的含有缺陷。

## 2. 从原始信号到模型输入：原始信号如何变成模型输入？

### 2.1 采样率、重采样与归一化

代码先检查信号是否为有限的二维数组、时间戳是否严格递增。源采样率由时间戳间隔的中位数估计：

\[
f*s \approx \frac{1}{\operatorname{median}(t*{i+1}-t_i)}.
\]

不同焊缝的采样点数和采样率可能不同，所以训练前会重采样到统一的目标采样率。窗口实验使用 `5400 Hz`；完整信号实验随后还会把整条信号插值到固定的 `4096` 个点。重采样的目的不是增加信息，而是让卷积核看到具有可比较时间尺度的输入。

归一化只使用当前训练折的信号计算每个通道的均值和标准差：

\[
x'*{c,i}=\frac{x*{c,i}-\mu*{c,\mathrm{train}}}
{\sigma*{c,\mathrm{train}}},
\qquad
\sigma\_{c,\mathrm{train}}=0\text{ 时取 }1.
\]

验证折只能使用训练折得到的 \(\mu\) 和 \(\sigma\)。如果把验证样本也用于计算归一化参数，就会把验证集的分布信息提前泄漏给模型。

### 2.2 标签向量与窗口

一条焊缝可能有多个缺陷，因此使用标签向量是一个合理的选择。我们使用三个独立的二分类标签。例如 `flash+tunnel` 会编码成 `[1, 0, 1]`，而不是互斥的单一类别。使用窗口划分的实验把信号切成 2 秒窗口，窗口起点每次移动 1 秒；因此相邻窗口重叠 1 秒。一个窗口在 5400 Hz 下包含 10800 个采样点。

短于 2 秒的信号会在尾部补零，同时生成 `sample_mask`；不同焊缝的窗口数不同时，会在 bag 层补齐并生成 `window_mask`。模型只在有效采样点和有效窗口上做平均或聚合，补零不会被当成真实信号。

这些数据处理主要实现在 [`training_dataset.py`](../src/mmdii/data/training_dataset.py)。

## 3. 传统手工统计基线B0：手工统计到底统计了什么？

B0 是我们的非深度学习基线。它不切窗口，也不直接看每个时间点，而是把每条焊缝压缩成一组可解释的统计量。对每个通道 \(x_1,\ldots,x_N\)，代码计算六个数（平均值，标准差，RMS，最小值，最大值，峰峰值）：

\[
\operatorname{mean}=\frac1N\sum_i x_i,
\]

\[
\operatorname{std}=\sqrt{\frac1N\sum_i(x_i-\operatorname{mean})^2},
\qquad

\operatorname{RMS}=\sqrt{\frac1N\sum_i x_i^2},
\]

以及

\[
\operatorname{minimum}=\min_i x_i,\quad
\operatorname{maximum}=\max_i x_i,\quad
\operatorname{peak_to_peak}=\max_i x_i-\min_i x_i.
\]

> RMS（Root Mean Square，均方根）是衡量一组数值“平均大小/有效幅值”的常用指标。
> 它的意义在于：对于波形、误差、噪声、信号强度等数据，RMS 能更准确地反映整体强度，而不是简单地取算术平均值。
> 因为平方会放大大值，RMS 对较大幅度的样本更敏感，因此常用于描述“整体能量水平”或“平均振幅”。

三条通道各有 6 个统计量，总共得到 18 个特征。直观地说，均值描述整体受力水平；标准差和 RMS 描述波动或信号能量；最小值、最大值和峰峰值更容易捕捉突发冲击、偏移或异常幅度。它们不是缺陷的“物理定律”，而是把可能有用的信号形状压缩成低维、可检查的摘要。

随后，代码先对 18 维特征做 `StandardScaler`，再为每个目标分别拟合一个 one-vs-rest Logistic Regression。其输出是：

\[
p(y_k=1\mid x)=\sigma(w_k^\top x+b_k)
=\frac1{1+e^{-(w_k^\top x+b_k)}}.
\]

`one-vs-rest` 的意思是：`flash`、`blur`、`tunnel` 各自有一个二分类器，三个概率可以同时为高。`class_weight="balanced"` 会按当前训练折的类别数量调整损失，让少数类不至于完全被多数类淹没。

B0 的价值在于提供一个强而透明的参考：如果 18 个统计量已经包含大量判别信息，那么更复杂的深度模型必须证明自己带来了额外收益，而不能只因为“用了神经网络”就被默认认为更好。相关实现见 [`statistical.py`](../src/mmdii/models/statistical.py)。

## 4. 从 TCN 到 ModernTCN

### 4.1 TCN 的基本直觉

传统序列模型常用 RNN 或 LSTM，按时间步递推隐藏状态。TCN（Temporal Convolutional Network）则用一维卷积在时间轴上滑动一个小窗口：

\[
y*t=\sum*{j=0}^{K-1}W*jx*{t-j}.
\]

卷积可以同时处理许多时间位置，训练时容易并行；堆叠多层后，一个输出能看到更长的历史，这个范围叫感受野。膨胀卷积在相邻卷积位置之间留出间隔，可以在不把卷积核做得很大的情况下扩大感受野。

TCN 的经典经验研究来自 Bai、Kolter 和 Koltun 的 *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*：[论文链接](https://arxiv.org/abs/1803.01271)。需要注意 causal convolution 与本项目实现的区别：因果卷积只使用当前及过去信息，适合在线预测；本项目的 `Conv1d` 使用对称 padding，属于非因果的“same padding”，因此一个窗口内部的前后时间信息都可以参与表示学习。

### 4.2 本项目的 ModernTCN-style encoder

ModernTCN 是一种面向一般时间序列的纯卷积结构，原论文发表于 ICLR 2024：[论文链接](https://openreview.net/pdf?id=pElYHYU2y)。仓库中的 `ModernTCNSmall` 是受其设计思想启发的轻量实现，并不是官方模型的逐层复刻。它的一个输入窗口经过以下步骤：

1. `1×1 Conv1d`（pointwise projection）把三条输入通道投影到隐藏通道；
2. depthwise 大核时间卷积分别提取各通道的局部时间模式；
3. `GroupNorm` 稳定中间表示；
4. 两个 pointwise 卷积、GELU 和 Dropout 完成跨通道混合并增加非线性；
5. 加上残差连接，避免深层变换破坏原始信息；
6. 最后投影到 embedding 维度，并沿时间维做平均池化。

可以把它理解为：大核卷积负责“在时间上看得更宽”，pointwise 卷积负责“重新组合不同传感器通道”，残差负责“保留一条稳定的原路”。代码位于 [`modern_tcn.py`](../src/mmdii/models/modern_tcn.py)。

## 5. E0 与 E1：完整信号、窗口和聚合

所有深度路线都可以画成同一条流水线：

```text
力信号 → ModernTCN 编码器 → 一个或多个窗口 embedding → 聚合器 → 三个焊缝级概率
```

### E0：完整信号

每条焊缝的完整信号先被重采样并插值成 4096 点，然后由一个 ModernTCN-small 直接输出焊缝级结果。它保留整段上下文，但没有显式的窗口级证据。

### E1a：masked mean

每个 2 秒窗口独立编码成 \(h_n\)，然后对所有有效窗口平均：

\[
\bar h=\frac{\sum_n m_nh_n}{\sum_nm_n},
\]

其中 \(m_n\) 是 `window_mask`。这个聚合假设整条焊缝的状态由所有窗口共同决定，比较稳健，但局部异常可能被很多正常窗口稀释。

### E1b：max 与 top-k mean

max 聚合对每个 embedding 维度保留最大窗口响应：

\[
z*d=\max*{n:m*n=1}z*{n,d}.
\]

它对应“一个强异常窗口就足以代表整条焊缝”的假设，容易受噪声峰值影响。top-k mean 则取每个维度响应最高的 3 个窗口再平均，试图在全局平均与单点最大之间折中。两者都是逐维操作，得到的向量不一定对应某一个真实窗口。

### E1c：gated attention MIL

E1c 允许不同缺陷关注不同窗口。对窗口 embedding \(h_n\)，代码先计算门控表示：

\[
g_n=\tanh(Vh_n)\odot\operatorname{sigmoid}(Uh_n),
\]

再对目标类别 \(k\) 计算注意力权重：

\[
a*{k,n}=\operatorname{softmax}\_n(w_k^\top g_n),
\qquad
z_k=\sum_n a*{k,n}h_n.
\]

这里的 \(a\_{k,n}\) 表示“为了判断第 k 类缺陷，第 n 个窗口在聚合时有多大权重”。因此 `flash` 和 `tunnel` 可以学习不同的窗口组合。它是一个很有用的候选窗口排序接口，但由于没有真实的窗口级起止标签，不能把 attention 直接称为已经验证的缺陷定位。

聚合器实现见 [`mil.py`](../src/mmdii/models/mil.py)。

## 6. 训练目标与评价指标

### 6.1 加权 BCE

模型输出的是 logits，经过 sigmoid 后才是概率。训练使用每个目标独立的加权二元交叉熵：

\[
\mathcal L=-\left[w_{+}y\log\sigma(z)+(1-y)\log(1-\sigma(z))\right].
\]

正类权重由当前训练折计算：

\[
w*{+}=\frac{N*{negative}}{N\_{positive}}.
\]

这一步只使用训练折标签。优化器是 AdamW，基线统一使用 20 epochs、学习率 `0.001` 和 weight decay `0.0001`，以便比较时主要改变模型路线而不是训练协议。

### 6.2 分类指标

给定阈值后，概率被转成 0/1 预测。核心指标为：

\[
\operatorname{Precision}=\frac{TP}{TP+FP},\quad
\operatorname{Recall}=\frac{TP}{TP+FN},
\]

\[
F1=\frac{2\cdot\operatorname{Precision}\cdot\operatorname{Recall}}
{\operatorname{Precision}+\operatorname{Recall}}.
\]

当前 F1、Precision 和 Recall 使用固定阈值 0.5；*关于此0.5的阈值选择仍待考虑，也就是说还需要再讨论0.5的阈值是否适合的问题*。Average Precision（报告中称 PR-AUC）使用概率排序，不依赖单一阈值。随机排序的 AP 基准大致等于正例比例，所以 `flash` 的 91.1% 正例比例必须被同时报告；只看高 F1 会误判模型能力。

`macro` 是先对各缺陷分别计算再平均，三个类别权重相同；`micro` 是把所有类别的 TP、FP、FN 合并后计算，更容易被样本多的类别影响。Brier score 是概率误差的均方形式：

\[
\operatorname{Brier}=\frac1M\sum_i(p_i-y_i)^2,
\]

越低代表概率通常越接近真实标签。Exact match 则要求一条焊缝的三个标签全部同时预测正确。

如果某个验证折中 `tunnel` 没有正例，PR-AUC、Recall 和 F1 没有定义，代码会记录为 `null`，并带有 `valid_positive_count=0`。这不是模型得了 0 分，而是该折没有足够事实来计算该指标。评价代码见 [`multilabel.py`](../src/mmdii/evaluation/multilabel.py)。

## 7. 第一批结果说明了什么？

下面这张表记录的是此前 Dataset v0.2.0 的图片组分折历史结果，保留它是为了说明已有基线；它不能直接当作 v0.2.1 焊缝独立主协议的结果，也不能和新协议分数混合排序。v0.2.1 的六组主协议实验需要在统一新分折上重新运行后再填入独立结果表。

| 方法                    |     宏 AP |     宏 F1 |     微 F1 | Brier（越低越好） | Exact match |
| ----------------------- | --------: | --------: | --------: | ----------------: | ----------: |
| **B0 statistical**      | **0.785** |     0.676 | **0.803** |         **0.146** |       0.455 |
| E0 full signal          |     0.690 |     0.670 |     0.746 |             0.164 |       0.406 |
| E1a mean                |     0.700 |     0.660 |     0.741 |             0.171 |       0.386 |
| E1b max                 |     0.611 |     0.574 |     0.660 |             0.208 |       0.267 |
| E1b top-k mean          |     0.659 |     0.600 |     0.705 |             0.198 |       0.337 |
| **E1c gated attention** |     0.634 | **0.683** |     0.769 |             0.167 |   **0.485** |

对这组历史结果，最稳妥的读法是：

- B0 是当前整体最强的排序和概率基线，说明幅值、波动和极值统计已经携带了不少可分信息；
- E1c 的固定阈值 macro F1 和 exact match 最高，`blur` 的 Recall/F1 也较好，说明按类别分配窗口权重可能有局部价值；
- E1c 的宏 AP 仍低于 B0，因此不能说它整体胜出；
- max 和 top-k 没有支持“单个极端窗口代表焊缝”的简单假设；
- `flash` 的正例比例很高，全部预测为正的 F1 基准已达 0.953；
- `tunnel` 只有 10 个正例，且只来自 3 个图片组，现有结果最多说明“可能存在信号”，不能证明稳定检测，更不能证明时间定位。

后续使用多个种子重新训练，发现在对 B0 和 E1c 使用 seed `7/17/27` 的复验中，B0 作为确定性统计模型结果完全相同；E1c 的 macro F1 平均约为 `0.7432 ± 0.0163`，宏 PR-AUC 约为 `0.8355 ± 0.0137`。这说明深度模型对初始化有一定敏感性，但没有改变“当前没有证据证明 E1c 整体优于 B0”的判断。

## 8. 当前工作的边界与下一步

现在我们知道深度学习方法在这个问题上还没有明显超过传统统计基线。现在方法仍需要进一步验证和改进。
我正在考虑以下几点：

1. 0.5 还不是最终生产阈值，对于阈值选择仍需要进一步讨论；
2. 生产工艺，包括各种参数还没有很好融入进模型，暂时还没有思路；
3. 现在给窗口使用的注意力是否可以更换其他注意力机制，或者使用其他 encoder 结构；
4. 编码器是否可以更换其他风格；
5. 采样频率和窗口划分等模型前输入是否可以改进，此举能否带来更多对于模型的提升；

## 参考资料与代码索引

- Bai, Kolter, Koltun. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. 2018. [arXiv](https://arxiv.org/abs/1803.01271)
- Luo, Wang. *ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis*. ICLR 2024. [OpenReview PDF](https://openreview.net/pdf?id=pElYHYU2y)
- 数据读取、重采样、窗口和归一化：[`src/mmdii/data/training_dataset.py`](../src/mmdii/data/training_dataset.py)
- B0 统计特征和逻辑回归：[`src/mmdii/models/statistical.py`](../src/mmdii/models/statistical.py)
- ModernTCN-style 编码器：[`src/mmdii/models/modern_tcn.py`](../src/mmdii/models/modern_tcn.py)
- MIL 聚合器：[`src/mmdii/models/mil.py`](../src/mmdii/models/mil.py)
- 加权 BCE 与多标签指标：[`src/mmdii/evaluation/multilabel.py`](../src/mmdii/evaluation/multilabel.py)
- 第一批基线结果与限制：[`2026-09-03-overnight-baseline-analysis.md`](2026-09-03-overnight-baseline-analysis.md)
