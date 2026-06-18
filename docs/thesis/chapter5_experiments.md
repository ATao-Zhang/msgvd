# 第4章 实验设计与结果分析

本章对“基于语义增强执行路径证据的漏洞行级定位与可解释方法”进行实验验证。实验按照小论文组织方式展开，包括实验目的、数据集与行级标签构建、对比方法、评价指标、主结果对比、DiverseVul 子集泛化实验、消融实验、参数敏感性分析、反事实扰动分析和案例分析。本文不重新训练漏洞检测模型，不修改原始模型结构和数据预处理流程；未完成真实运行的外部基线、DiverseVul-subset 结果和案例级结果仅保留待填位置，不编造数值。

## 4.1 实验目的

本文实验主要回答五个问题。第一，语义增强执行路径证据是否能够提升漏洞行级定位效果。第二，本文方法相对于 Random、API-Heuristic、梯度归因、图解释方法和行级检测模型等外部基线是否具有竞争力。第三，本文方法能否在来源于真实开源项目漏洞修复提交的 DiverseVul-subset 上保持泛化能力。第四，Attention Score、Semantic Risk Score 和 Counterfactual Score 三类分数各自对定位结果有何贡献。第五，反事实扰动模块更适合作为最终主排序方法，还是作为候选漏洞行可信度验证和 Top-5 召回增强模块。

## 4.2 数据集与行级标签构建

### 4.2.1 SARD 数据集

SARD 测试集以 XFG 文件形式组织，每个样本包含图结构、节点 token、源代码行号和样本级漏洞标签。本文采用 `xfg_stem` 策略构造 SARD 行级标签，即从 XFG 文件名中提取漏洞行号，例如：

```text
116.xfg.pkl -> true_vul_lines = [116]
```

| 数据集 | 总样本数 | 正样本数 | 负样本数 | 有行级标签的正样本数 | 定位评估样本数 | 行级标签来源 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SARD test | 2648 | 589 | 2059 | 589 | 589 | xfg_stem |

负样本不生成漏洞行标签，仅正样本参与 Top-k、MRR 和 IFA 定位评估。因此，SARD 上共有 589 个正样本用于漏洞行级定位评价。

### 4.2.2 DiverseVul-subset 数据集

DiverseVul 来源于真实开源项目漏洞修复提交，代码结构比 SARD 更复杂。由于原始 DiverseVul 主要提供函数级标签，本文采用补丁差异构造近似行级标签：根据漏洞修复提交提取修复前后 diff，将被删除或被修改的代码行映射回漏洞函数内部，作为近似漏洞行标签。为提高标签可靠性，本文过滤补丁过大、修改范围过宽、无法稳定映射以及 tangled patch 严重的样本。

| 数据集 | 总样本数 | 正样本数 | 负样本数 | 有行级标签的正样本数 | 定位评估样本数 | 行级标签来源 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DiverseVul-subset | 1500 | 500 | 1000 | 500 | 500 | patch diff 近似行级标签 |

DiverseVul-subset 的行级标签是近似标签而非人工逐行审计标签，因此主要用于泛化趋势验证，不与 SARD 的 `xfg_stem` 精确行号标签完全等同。

## 4.3 对比方法

（1）Random。随机排列候选代码行，作为最低参考基线。该方法不使用模型输出、代码语义或扰动信息。

（2）API-Heuristic。仅根据危险 API 是否出现排序，例如 `strcpy`、`memcpy`、`sprintf`、`gets`、`scanf` 等，用于验证本文方法不是简单危险 API 匹配。

（3）Attention-only。仅使用模型节点级或语句级注意力证据分数，是本文的内部基础对照方法。

（4）Gradient Saliency。基于梯度归因的解释方法，对漏洞类别输出相对于输入 token embedding 或节点特征计算梯度，将同一代码行内的梯度绝对值累加得到行级分数。

（5）Integrated Gradients。经典梯度积分归因方法，从基线输入到真实输入进行路径积分，计算输入特征对漏洞预测结果的贡献，并聚合为行级分数。

（6）GNNExplainer。通用图神经网络解释方法，通过优化子图结构和特征掩码寻找对预测最关键的节点和边，再将节点重要性映射到源代码行。

（7）LineVul。面向行级漏洞预测的 Transformer 基线方法，主要在 DiverseVul-subset 上作为行级检测基线。

（8）Semantic-only。仅使用本文语义风险分数排序，用于评估危险 API、内存操作、输入源、数组指针访问、边界函数和算术操作等语义规则的独立贡献。

（9）Counterfactual-only。仅使用反事实扰动后的漏洞概率下降分数排序。对于候选行 $v_i$，反事实分数定义为：

$$
C(v_i)=\max\left(0, P_{\mathrm{vul}}(G)-P_{\mathrm{vul}}(G_{\setminus i})\right)
$$

（10）Ours, A+S。本文主方法，融合 Attention Score 和 Semantic Risk Score。对于候选代码行 $v_i$，最终分数为：

$$
S(v_i) = (1-\lambda)\hat{A}(v_i) + \lambda\hat{R}(v_i)
$$

其中，$\hat{A}(v_i)$ 表示归一化后的注意力证据分数，$\hat{R}(v_i)$ 表示归一化后的语义风险分数，$\lambda$ 为语义风险权重。最终设置 $\lambda=0.9$。

（11）Ours+CF。在 A+S 基础上加入反事实扰动验证，主要用于可信度验证和 Top-5 召回增强，不作为最终主排序方法。

## 4.4 评价指标

本文采用 Top-1 Accuracy、Top-3 Accuracy、Top-5 Accuracy、MRR 和 IFA 评价漏洞行级定位效果。所有指标仅在具有行级漏洞标签的正样本上计算。

Top-k Accuracy 表示排序前 $k$ 个候选代码行中是否命中任意真实漏洞行：

$$
Top\text{-}k=\frac{1}{N}\sum_{i=1}^{N}I(R_i^k\cap Y_i\neq\varnothing)
$$

MRR（Mean Reciprocal Rank）用于衡量第一个真实漏洞行在排序列表中的平均倒数排名：

$$
MRR=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{rank_i}
$$

IFA 表示 Initial False Alarm，即找到第一个真实漏洞行之前需要检查的非漏洞候选行数量。若真实漏洞行排名为第 $r$ 位，则该样本的 IFA 为 $r-1$。IFA 越低，说明人工审计成本越低。

## 4.5 主结果对比

主结果对比用于展示 Ours, A+S（$\lambda=0.9$）与内部基线和外部基线之间的性能差异。当前已完成并确认真实数值的是 SARD 上的内部方法结果，外部基线结果需要在相同协议下复现后填入。

| 方法 | 数据集 | Top-1 Acc | Top-3 Acc | Top-5 Acc | MRR | IFA | 说明 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Random | SARD | 待填 | 待填 | 待填 | 待填 | 待填 | 最低参考基线 |
| API-Heuristic | SARD | 待填 | 待填 | 待填 | 待填 | 待填 | 危险 API 启发式 |
| Gradient Saliency | SARD | 待填 | 待填 | 待填 | 待填 | 待填 | 梯度归因基线 |
| Integrated Gradients | SARD | 待填 | 待填 | 待填 | 待填 | 待填 | 积分梯度基线 |
| GNNExplainer | SARD | 待填 | 待填 | 待填 | 待填 | 待填 | 图解释基线 |
| Attention-only | SARD | 0.2920 | 0.5772 | 0.7759 | 0.4867 | 3.3141 | 内部基础对照 |
| Ours, A+S ($\lambda=0.9$) | SARD | 0.4041 | 0.8353 | 0.9474 | 0.6272 | 1.3447 | 本文主方法 |

在 SARD 上，Ours, A+S（$\lambda=0.9$）相比 Attention-only 明显提升：Top-1 Accuracy 提升 11.21 个百分点，Top-3 Accuracy 提升 25.81 个百分点，Top-5 Accuracy 提升 17.15 个百分点，MRR 提升 14.05 个百分点，IFA 约降低 59.4%。

## 4.6 DiverseVul 子集泛化实验

DiverseVul-subset 泛化实验用于检验本文方法在真实开源项目漏洞样本上的适用性。由于当前对话未提供 DiverseVul-subset 上的真实定位指标，本文不编写具体数值。该实验完成后，应报告 Random、API-Heuristic、Attention-only、LineVul、Ours, A+S（$\lambda=0.9$）以及必要外部基线的 Top-k Accuracy、MRR 和 IFA。

| 方法 | Top-1 Acc | Top-3 Acc | Top-5 Acc | MRR | IFA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 待填 | 待填 | 待填 | 待填 | 待填 |
| API-Heuristic | 待填 | 待填 | 待填 | 待填 | 待填 |
| Attention-only | 待填 | 待填 | 待填 | 待填 | 待填 |
| LineVul | 待填 | 待填 | 待填 | 待填 | 待填 |
| Ours, A+S ($\lambda=0.9$) | 待填 | 待填 | 待填 | 待填 | 待填 |

若 DiverseVul-subset 结果低于 SARD，应结合真实项目代码复杂度、补丁差异标签噪声和跨数据分布差异进行分析，而不应简单解释为方法失效。

## 4.7 消融实验

消融实验用于分析不同分数模块的独立作用。SARD 上已完成的消融结果如下。

| 方法 | Top-1 Acc | Top-3 Acc | Top-5 Acc | MRR | IFA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention-only | 0.2920 | 0.5772 | 0.7759 | 0.4867 | 3.3141 |
| Semantic-only | 0.3531 | 0.7912 | 0.9168 | 0.5791 | 1.6486 |
| Counterfactual-only | 0.2767 | 0.6180 | 0.8014 | 0.4936 | 3.0153 |
| Ours, A+S ($\lambda=0.9$) | 0.4041 | 0.8353 | 0.9474 | 0.6272 | 1.3447 |
| Ours+CF, Top5-best | 0.3922 | 0.8251 | 0.9542 | 0.6211 | 1.3463 |

Semantic-only 明显优于 Attention-only，说明语义风险特征与真实漏洞行高度相关。Ours, A+S（$\lambda=0.9$）在 Top-1、Top-3、MRR 和 IFA 上取得最佳主排序效果。Ours+CF 的 Top-5 Accuracy 最高，但 Top-1 和 MRR 略低于 Ours, A+S，因此本文最终仍采用 Ours, A+S（$\lambda=0.9$）作为主方法。

## 4.8 参数敏感性分析

本文对 A+S 方法中的语义权重 $\lambda$ 进行参数敏感性分析。融合公式为：

$$
S(v_i) = (1-\lambda)\hat{A}(v_i) + \lambda\hat{R}(v_i)
$$

| semantic_weight $\lambda$ | Top-1 Acc | Top-3 Acc | Top-5 Acc | MRR | IFA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 0.3447 | 0.6944 | 0.8846 | 0.5569 | 2.1121 |
| 0.5 | 0.3565 | 0.7997 | 0.9321 | 0.5877 | 1.5637 |
| 0.7 | 0.3786 | 0.8065 | 0.9491 | 0.6120 | 1.3922 |
| 0.9 | 0.4041 | 0.8353 | 0.9474 | 0.6272 | 1.3447 |

随着 $\lambda$ 从 0.3 增加到 0.9，Top-1 Accuracy、Top-3 Accuracy、MRR 和 IFA 整体持续改善。$\lambda=0.7$ 的 Top-5 Accuracy 略高于 $\lambda=0.9$，但 $\lambda=0.9$ 在 Top-1、Top-3、MRR 和 IFA 上表现更优，因此作为最终主方法参数。

## 4.9 反事实扰动分析

反事实扰动模块通过局部 mask 候选行对应节点，观察漏洞预测概率下降程度，从而验证候选行是否真正影响模型预测。反事实分数定义为：

$$
C(v_i)=\max\left(0, P_{\mathrm{vul}}(G)-P_{\mathrm{vul}}(G_{\setminus i})\right)
$$

SARD 上的结果表明，Counterfactual-only 的 Top-1 Accuracy 为 0.2767，低于 Ours, A+S 主方法；但 Ours+CF Top5-best 的 Top-5 Accuracy 达到 0.9542，高于 Ours, A+S（$\lambda=0.9$）的 0.9474。这说明反事实模块有助于提高前 5 个候选漏洞行集合的覆盖率，但并不能稳定提升首位排序质量。因此，本文将 CF 定位为可信度验证和 Top-5 召回增强模块。

## 4.10 案例分析

案例分析用于展示本文方法如何辅助人工理解模型定位结果。每个案例建议包含原始代码片段、真实漏洞行、Attention-only 排名、Ours, A+S 排名、语义风险标签以及必要的反事实概率下降信息。当前对话未提供具体样本级案例输出，因此本文档不编造案例内容。最终论文可选择成功案例和失败案例各若干，分别说明语义风险增强的作用和方法边界。

## 4.11 实验小结

本章按照小论文实验结构，对本文提出的漏洞行级定位方法进行了整理。SARD 测试集包含 2648 个样本，其中 589 个正样本具有通过 `xfg_stem` 构造的行级标签；DiverseVul-subset 包含 1500 个样本，其中 500 个正样本通过补丁差异构造近似行级标签。

在已完成的 SARD 实验中，Ours, A+S（$\lambda=0.9$）相比 Attention-only 明显提升了定位性能。消融实验和参数敏感性分析表明，语义风险分数是提升漏洞行级定位效果的关键因素，模型注意力证据提供了必要的执行路径上下文补充。反事实扰动不作为最终主排序方法，而作为候选行可信度验证和 Top-5 召回增强模块。外部基线、DiverseVul-subset 完整结果和案例分析需要在后续真实实验后补充。