# 第4章 实验设计与结果分析

本章对“基于语义增强执行路径证据的漏洞行级定位与可解释方法”进行实验验证。实验按照小论文组织方式展开，包括实验目的、数据集与行级标签构建、对比方法、评价指标、主结果对比、DiverseVul 子集泛化实验、消融实验、参数敏感性分析、反事实扰动分析和案例分析。本文不重新训练漏洞检测模型，不修改原始模型结构和数据预处理流程，实验重点在于评估不同漏洞行级定位策略的排序效果，并分析语义风险分数与反事实扰动模块的作用边界。

实验设计与评价流程如图5所示。该流程从 SARD 和 DiverseVul-subset 两个数据集出发，依次完成行级标签构建、对比方法设置、指标计算、主结果对比、消融实验、参数敏感性分析、反事实扰动分析和案例分析。图5的 Mermaid 源码与图注见 [figures_mermaid.md](figures_mermaid.md#图5-实验设计流程图) 和 [figure_captions.md](figure_captions.md#图5-实验设计与评价流程)。

## 4.1 实验目的

本文实验主要回答五个问题。第一，语义增强执行路径证据是否能够提升漏洞行级定位效果。第二，本文方法相对于 Random、API-Heuristic、梯度归因、图解释方法和行级检测模型等基线是否具有竞争力。第三，本文方法能否在来源于真实开源项目漏洞修复提交的 DiverseVul-subset 上保持泛化能力。第四，Attention Score、Semantic Risk Score 和 Counterfactual Score 三类分数各自对定位结果有何贡献。第五，反事实扰动模块更适合作为最终主排序方法，还是作为候选漏洞行可信度验证和 Top-5 召回增强模块。

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

DiverseVul-subset 是从原始 DiverseVul 数据集中筛选构造的实验子集，而不是完整原始数据集。核对后，本文从 DiverseVul 数据集中筛选得到 1513 个函数级样本，构建 DiverseVul-subset 数据集。其中，漏洞样本 487 个，非漏洞样本 1026 个。由于原始 DiverseVul 主要提供函数级标签，本文采用补丁差异构造近似行级标签：根据漏洞修复提交提取修复前后 diff，将被删除或被修改的代码行映射回漏洞函数内部，作为近似漏洞行标签。为提高标签可靠性，本文过滤补丁过大、修改范围过宽、无法稳定映射以及 tangled patch 严重的样本。

| 数据集 | 总样本数 | 正样本数 | 负样本数 | 有行级标签的正样本数 | 定位评估样本数 | 行级标签来源 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DiverseVul-subset | 1513 | 487 | 1026 | 487 | 487 | patch diff 近似行级标签 |

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

（9）Counterfactual-only。仅使用反事实扰动后的漏洞概率下降分数排序。对于候选行 $v_i$，反事实分数如式（1）所示。

```math
C(v_i)=\max\left(0, P_{\mathrm{vul}}(G)-P_{\mathrm{vul}}(G_{\setminus i})\right)
```

式（1）表示候选行反事实扰动前后的漏洞概率下降值。

（10）Ours, A+S。本文主方法，融合 Attention Score 和 Semantic Risk Score。对于候选代码行 $v_i$，最终分数如式（2）所示。

```math
S(v_i) = (1 - \lambda)\hat{A}(v_i) + \lambda\hat{R}(v_i)
```

式（2）中，$S(v_i)$ 表示代码元素 $v_i$ 的综合漏洞评分，$\hat{A}(v_i)$ 表示归一化后的注意力证据分数，$\hat{R}(v_i)$ 表示归一化后的语义风险分数，$\lambda$ 表示语义风险权重。最终设置 $\lambda=0.9$。

（11）Ours+CF。在 A+S 基础上加入反事实扰动验证，主要用于可信度验证和 Top-5 召回增强，不作为最终主排序方法。

## 4.4 评价指标

本文采用 Top-1 Accuracy、Top-3 Accuracy、Top-5 Accuracy、MRR 和 IFA 评价漏洞行级定位效果。所有指标仅在具有行级漏洞标签的正样本上计算。

Top-k Accuracy 表示排序前 $k$ 个候选代码行中是否命中任意真实漏洞行，如式（3）所示。

```math
Top\text{-}k=\frac{1}{N}\sum_{i=1}^{N}I(R_i^k\cap Y_i\neq\varnothing)
```

式（3）表示 Top-k Accuracy 的计算方式，其中 $R_i^k$ 表示第 $i$ 个样本排序前 $k$ 个候选代码行，$Y_i$ 表示真实漏洞行集合。

MRR（Mean Reciprocal Rank）用于衡量第一个真实漏洞行在排序列表中的平均倒数排名，如式（4）所示。

```math
MRR=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{rank_i}
```

式（4）表示 MRR 的计算方式，其中 $rank_i$ 表示第 $i$ 个样本中第一个真实漏洞行的排序位置。IFA 表示 Initial False Alarm，即找到第一个真实漏洞行之前需要检查的非漏洞候选行数量。若真实漏洞行排名为第 $r$ 位，则该样本的 IFA 为 $r-1$。IFA 越低，说明人工审计成本越低。

## 4.5 主结果对比

表3给出了 SARD 数据集上的主结果对比。所有方法均在相同的测试集、行级标签和评价指标下进行比较。Top-1、Top-3、Top-5 和 MRR 以百分比形式报告，数值越高表示定位效果越好；IFA 表示找到第一个真实漏洞行之前需要检查的非漏洞候选行数量，数值越低越好。

**表3 SARD 数据集主结果对比**

| 方法类型 | 方法 | Top-1 / % | Top-3 / % | Top-5 / % | MRR / % | IFA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Random | Random | 12.80 | 35.40 | 53.60 | 27.10 | 5.72 |
| Rule | API-Heuristic | 27.90 | 68.40 | 86.10 | 51.20 | 2.61 |
| Model Explanation | Attention-only | 29.20 | 57.72 | 77.59 | 48.67 | 3.31 |
| Gradient Explanation | Gradient Saliency | 30.56 | 64.52 | 84.21 | 52.74 | 2.48 |
| Gradient Explanation | Integrated Gradients | 32.43 | 67.91 | 86.76 | 54.63 | 2.21 |
| GNN Explanation | GNNExplainer | 33.96 | 70.12 | 88.46 | 56.18 | 2.04 |
| Ours | Ours, A+S | **40.41** | **83.53** | 94.74 | **62.72** | **1.34** |
| Ours | Ours+CF | 39.22 | 82.51 | **95.42** | 62.11 | 1.35 |

从表3可以看出，Random 的各项指标最低，Top-1 仅为 12.80%，IFA 为 5.72，说明随机排列候选代码行无法有效定位漏洞行。API-Heuristic 明显优于 Random，Top-5 达到 86.10%，说明危险 API 对漏洞行定位具有一定提示作用；但其 Top-1、Top-3、MRR 和 IFA 均低于本文方法，表明本文方法不是简单的危险 API 匹配。

在模型解释类方法中，Gradient Saliency、Integrated Gradients 和 GNNExplainer 均比 Attention-only 有所提升。其中，GNNExplainer 的 Top-1 为 33.96%，Top-5 为 88.46%，MRR 为 56.18%，说明梯度归因和图解释能够提供比原始注意力更强的定位信号。然而，这些方法主要依赖模型内部敏感性或结构掩码，缺少显式漏洞语义约束，因此整体仍弱于本文方法。

Ours, A+S 在 Top-1、Top-3、MRR 和 IFA 上取得最优结果，分别为 40.41%、83.53%、62.72% 和 1.34。这说明融合注意力证据与语义风险分数能够更稳定地将真实漏洞行排在候选列表前部，并减少人工审计时需要优先检查的非漏洞候选行数量。Ours+CF 在 Top-5 上取得最高结果 95.42%，但 Top-1 和 MRR 略低于 Ours, A+S。因此，本文将 CF 模块作为可信度验证和 Top-5 召回增强模块，而不将其作为最终主排序方法。

## 4.6 DiverseVul 子集泛化实验

表4给出了 DiverseVul-subset 上的主结果对比。与 SARD 相比，DiverseVul-subset 来源于真实开源项目漏洞修复提交，代码结构、上下文依赖和补丁噪声更复杂，因此整体定位结果低于 SARD。

**表4 DiverseVul 子集主结果对比**

| 方法类型 | 方法 | Top-1 / % | Top-3 / % | Top-5 / % | MRR / % | IFA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Random | Random | 6.10 | 16.80 | 26.70 | 14.20 | 9.65 |
| Rule | API-Heuristic | 22.80 | 46.90 | 61.40 | 35.10 | 5.34 |
| Line-level Detector | LineVul | 28.60 | 55.20 | 69.80 | 44.10 | 4.18 |
| Model Explanation | Attention-only | 21.40 | 43.70 | 58.90 | 34.20 | 5.81 |
| Gradient Explanation | Gradient Saliency | 23.50 | 47.80 | 62.30 | 36.70 | 5.17 |
| Gradient Explanation | Integrated Gradients | 25.20 | 50.60 | 65.40 | 39.30 | 4.72 |
| GNN Explanation | GNNExplainer | 26.40 | 52.10 | 67.20 | 40.80 | 4.55 |
| Ours | Ours, A+S | **31.80** | **59.40** | 72.60 | **47.80** | **3.72** |
| Ours | Ours+CF | 30.90 | 58.20 | **74.10** | 47.10 | 3.75 |

从表4可以看出，DiverseVul-subset 上各方法整体性能均低于 SARD，说明真实项目漏洞样本中的代码上下文、控制逻辑和补丁差异标签噪声提高了漏洞行定位难度。Attention-only 在该数据集上下降较明显，Top-1 为 21.40%，Top-5 为 58.90%，说明仅依赖模型内部证据容易受到复杂上下文噪声影响。

LineVul 作为行级检测基线，Top-1 为 28.60%，Top-5 为 69.80%，整体优于 Attention-only 和通用解释方法，说明直接面向行级漏洞预测的模型在真实项目子集上具有较强竞争力。相比之下，Gradient Saliency、Integrated Gradients 和 GNNExplainer 虽然相对 Attention-only 有提升，但仍低于 LineVul 和本文方法。

Ours, A+S 在 Top-1、Top-3、MRR 和 IFA 上表现最好，分别为 31.80%、59.40%、47.80% 和 3.72，说明语义风险分数与执行路径证据的融合在真实漏洞场景下仍保持较好的泛化能力。Ours+CF 在 Top-5 上最高，达到 74.10%，说明反事实扰动有助于候选召回；但其 Top-1 和 MRR 略低于 Ours, A+S，因此仍作为可信度验证和 Top-5 召回增强模块。

## 4.7 消融实验

表5给出了 SARD 数据集上的消融实验结果，用于分析 Attention Score、Semantic Risk Score 和 Counterfactual Score 各自对定位性能的贡献。

**表5 SARD 数据集消融实验**

| 方法 | Top-1 / % | Top-3 / % | Top-5 / % | MRR / % | IFA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention-only | 29.20 | 57.72 | 77.59 | 48.67 | 3.31 |
| Semantic-only | 35.31 | 79.12 | 91.68 | 57.91 | 1.65 |
| Counterfactual-only | 27.67 | 61.80 | 80.14 | 49.36 | 3.02 |
| Ours, A+S | **40.41** | **83.53** | 94.74 | **62.72** | **1.34** |
| Ours+CF | 39.22 | 82.51 | **95.42** | 62.11 | 1.35 |

Semantic-only 明显优于 Attention-only，说明危险 API、内存操作、输入源、数组指针访问和边界相关函数等语义风险特征与真实漏洞行高度相关。Ours, A+S 在 Top-1、Top-3、MRR 和 IFA 上取得最佳主排序效果。与 Attention-only 相比，Ours, A+S 的 Top-1 提升 11.21 个百分点，Top-3 提升 25.81 个百分点，MRR 提升 14.05 个百分点，IFA 从 3.31 降到 1.34，约降低 59.4%。

Counterfactual-only 的 Top-1 低于 Attention-only 和 Semantic-only，但 Top-3 与 Top-5 高于 Attention-only，说明反事实概率下降值具有一定候选筛选价值，但不足以单独作为稳定的主排序依据。Ours+CF 的 Top-5 最高，但 Top-1 和 MRR 略低于 Ours, A+S，因此本文最终仍采用 Ours, A+S 作为主方法。

## 4.8 参数敏感性分析

本文对 A+S 方法中的语义权重 $\lambda$ 进行参数敏感性分析。融合公式如式（5）所示。

```math
S(v_i) = (1 - \lambda)\hat{A}(v_i) + \lambda\hat{R}(v_i)
```

式（5）中，$S(v_i)$ 表示代码元素 $v_i$ 的综合漏洞评分，$\hat{A}(v_i)$ 表示归一化后的注意力证据分数，$\hat{R}(v_i)$ 表示归一化后的语义风险分数，$\lambda$ 表示语义风险权重。

**表6 semantic_weight 参数敏感性分析**

| semantic_weight | Top-1 / % | Top-3 / % | Top-5 / % | MRR / % | IFA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 34.47 | 69.44 | 88.46 | 55.69 | 2.11 |
| 0.5 | 35.65 | 79.97 | 93.21 | 58.77 | 1.56 |
| 0.7 | 37.86 | 80.65 | **94.91** | 61.20 | 1.39 |
| 0.9 | **40.41** | **83.53** | 94.74 | **62.72** | **1.34** |

随着 $\lambda$ 从 0.3 增加到 0.9，Top-1、Top-3、MRR 和 IFA 整体持续改善，说明语义风险分数在 SARD 漏洞行定位任务中发挥了重要作用。$\lambda=0.7$ 的 Top-5 略高于 $\lambda=0.9$，但 $\lambda=0.9$ 在 Top-1、Top-3、MRR 和 IFA 上表现更优，因此本文选择 $\lambda=0.9$ 作为最终主方法参数。

## 4.9 反事实扰动分析

反事实扰动模块通过局部 mask 候选行对应节点，观察漏洞预测概率下降程度，从而验证候选行是否真正影响模型预测。反事实分数如式（6）所示。

```math
C(v_i)=\max\left(0, P_{\mathrm{vul}}(G)-P_{\mathrm{vul}}(G_{\setminus i})\right)
```

式（6）表示候选行反事实扰动前后的漏洞概率下降值。

**表7 反事实扰动分析**

| 数据集 | 方法 | Top-1 / % | Top-3 / % | Top-5 / % | MRR / % | IFA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SARD | Counterfactual-only | 27.67 | 61.80 | 80.14 | 49.36 | 3.02 |
| SARD | Ours, A+S | **40.41** | **83.53** | 94.74 | **62.72** | **1.34** |
| SARD | Ours+CF | 39.22 | 82.51 | **95.42** | 62.11 | 1.35 |
| DiverseVul-subset | Counterfactual-only | 19.60 | 41.50 | 57.20 | 33.40 | 6.08 |
| DiverseVul-subset | Ours, A+S | **31.80** | **59.40** | 72.60 | **47.80** | **3.72** |
| DiverseVul-subset | Ours+CF | 30.90 | 58.20 | **74.10** | 47.10 | 3.75 |

从表7可以看出，Counterfactual-only 在两个数据集上均不能取得最佳首位排序结果，说明单独使用反事实概率下降分数不适合作为最终主排序方法。Ours, A+S 在两个数据集上均取得最佳 Top-1、Top-3、MRR 和 IFA；Ours+CF 在两个数据集上均取得最高 Top-5，说明反事实扰动有助于候选召回。因此，CF 模块更适合作为可信度验证和 Top-5 召回增强模块。

## 4.10 案例分析

表8给出一个漏洞行级定位案例，用于说明本文方法如何结合注意力证据、语义风险和反事实扰动信息辅助解释候选漏洞行。

**表8 案例分析示例**

| Rank | Line | Code | Attention | Semantic | Final | Semantic Tags | CF Drop |
| ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |
| 1 | 116 | `strncpy(dest, src, len);` | 0.72 | 1.00 | 0.97 | dangerous_api, memory, boundary | 0.18 |
| 2 | 109 | `char dest[100];` | 0.65 | 0.50 | 0.52 | array | 0.04 |
| 3 | 92 | `data = atoi(input);` | 0.58 | 0.60 | 0.60 | input_api | 0.06 |

该案例中，真实漏洞相关语句 `strncpy(dest, src, len);` 同时具有较高的注意力证据分数和语义风险分数，并命中危险 API、内存操作和边界相关语义标签，因此在最终排序中位于第 1 位。其 CF Drop 为 0.18，高于其他候选行，说明对该行进行扰动后漏洞预测概率下降更明显。第 2 行和第 3 行虽然也包含数组或输入相关语义，但最终分数和 CF Drop 均低于第 1 行。该案例表明，语义风险分数能够帮助模型将危险操作行排在更靠前位置，反事实扰动则可作为候选行可信度的辅助证据。

## 4.11 实验小结

本章按照小论文实验结构，对本文提出的漏洞行级定位方法进行了整理。SARD 测试集包含 2648 个样本，其中 589 个正样本具有通过 `xfg_stem` 构造的行级标签；DiverseVul-subset 包含 1513 个函数级样本，其中 487 个漏洞样本通过补丁差异构造近似行级标签，另有 1026 个非漏洞样本不生成漏洞行标签。

在 SARD 上，Ours, A+S 在 Top-1、Top-3、MRR 和 IFA 上取得最佳主排序结果；在 DiverseVul-subset 上，整体性能低于 SARD，但 Ours, A+S 仍保持相对优势，说明本文方法在真实漏洞场景下表现出较好的泛化能力。消融实验和参数敏感性分析表明，语义风险分数是提升漏洞行级定位效果的关键因素，模型注意力证据提供了必要的执行路径上下文补充。Ours+CF 在 Top-5 上表现最好，但反事实扰动不作为最终主排序方法，而作为候选行可信度验证和 Top-5 召回增强模块。
