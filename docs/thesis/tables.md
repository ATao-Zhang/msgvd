# 小论文实验表格汇总

本文档整理第4章实验部分可直接引用的表格。所有数值均来自已确认的实验结果。

## 表1 数据集与行级标签统计

**表1 实验数据集与行级标签构建统计**

| 数据集 | 总样本数 | 正样本数 | 负样本数 | 有行级标签的正样本数 | 定位评估样本数 | 行级标签来源 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SARD test | 2648 | 589 | 2059 | 589 | 589 | xfg_stem |
| DiverseVul-subset | 1513 | 487 | 1026 | 487 | 487 | patch diff 近似行级标签 |

表1说明：SARD 采用 `xfg_stem` 策略从 XFG 文件名中提取漏洞行号，例如 `116.xfg.pkl -> true_vul_lines=[116]`。负样本不生成漏洞行标签，只有正样本参与 Top-k、MRR 和 IFA 定位评估。DiverseVul-subset 从原始 DiverseVul 数据集中筛选得到 1513 个函数级样本，其中漏洞样本 487 个、非漏洞样本 1026 个；行级标签由补丁 diff 中被删除或被修改的代码行映射回漏洞函数内部得到，并过滤补丁过大、修改范围过宽、无法稳定映射和 tangled patch 严重的样本。

## 表2 对比方法说明表

**表2 本文实验中的对比方法**

| 方法 | 方法类型 | 说明 |
| --- | --- | --- |
| Random | 随机基线 | 随机排列候选代码行，作为最低参考基线 |
| API-Heuristic | 启发式基线 | 根据 `strcpy`、`memcpy`、`sprintf`、`gets`、`scanf` 等危险 API 是否出现排序 |
| Attention-only | 内部基线 | 仅使用模型节点级或语句级注意力证据分数 |
| Gradient Saliency | 梯度归因基线 | 对 token embedding 或节点特征计算梯度，并聚合为行级分数 |
| Integrated Gradients | 梯度积分基线 | 从基线输入到真实输入进行路径积分，并聚合为行级贡献 |
| GNNExplainer | 图解释基线 | 优化子图结构和特征掩码，寻找关键节点和边 |
| LineVul | 行级检测基线 | 面向行级漏洞预测的 Transformer 方法，主要用于 DiverseVul-subset |
| Semantic-only | 内部消融 | 仅使用本文语义风险分数 |
| Counterfactual-only | 内部消融 | 仅使用反事实扰动后的漏洞概率下降分数 |
| Ours, A+S | 本文主方法 | 融合 Attention Score 与 Semantic Risk Score，最终设置 $\lambda=0.9$ |
| Ours+CF | 辅助验证方法 | 在 A+S 基础上加入反事实扰动，主要用于可信度验证和 Top-5 召回增强 |

表2说明：本文最终主方法是 Ours, A+S，而不是 Ours+CF。

## 表3 SARD 数据集主结果对比

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

表3说明：Random 的各项指标最低，说明随机排序无法有效定位漏洞行。API-Heuristic 优于 Random，但不如本文方法，说明本文方法不是简单危险 API 匹配。Gradient Saliency、Integrated Gradients 和 GNNExplainer 均比 Attention-only 有提升，但缺少显式漏洞语义约束。Ours, A+S 在 Top-1、Top-3、MRR 和 IFA 上最优；Ours+CF 在 Top-5 上最高，但 Top-1 和 MRR 略低于 Ours, A+S，因此 CF 作为可信度验证和 Top-5 召回增强模块。

## 表4 DiverseVul 子集主结果对比

**表4 DiverseVul 子集主结果对比**

| 方法类型 | 方法 | Top-1 / % | Top-3 / % | Top-5 / % | MRR / % | IFA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Random | Random | 6.13 | 16.84 | 26.72 | 14.18 | 9.67 |
| Rule | API-Heuristic | 22.76 | 46.93 | 61.37 | 35.12 | 5.36 |
| Line-level Detector | LineVul | 28.64 | 55.23 | 69.76 | 44.07 | 4.19 |
| Model Explanation | Attention-only | 21.43 | 43.68 | 58.87 | 34.24 | 5.79 |
| Gradient Explanation | Gradient Saliency | 23.47 | 47.83 | 62.26 | 36.72 | 5.16 |
| Gradient Explanation | Integrated Gradients | 25.24 | 50.57 | 65.36 | 39.27 | 4.74 |
| GNN Explanation | GNNExplainer | 26.37 | 52.14 | 67.23 | 40.83 | 4.53 |
| Ours | Ours, A+S | **31.83** | **59.37** | 72.64 | **47.82** | **3.71** |
| Ours | Ours+CF | 30.94 | 58.24 | **74.07** | 47.06 | 3.76 |

表4说明：DiverseVul-subset 比 SARD 更难，因此整体结果低于 SARD。Attention-only 在真实场景下降明显，说明模型内部证据受上下文噪声影响。LineVul 作为行级检测基线，整体优于 Attention-only 和通用解释方法。Ours, A+S 在 Top-1、Top-3、MRR 和 IFA 上表现最好。Ours+CF 在 Top-5 上最高，说明反事实扰动有助于候选召回。

## 表5 SARD 数据集消融实验

**表5 SARD 数据集消融实验**

| 方法 | Top-1 / % | Top-3 / % | Top-5 / % | MRR / % | IFA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention-only | 29.20 | 57.72 | 77.59 | 48.67 | 3.31 |
| Semantic-only | 35.31 | 79.12 | 91.68 | 57.91 | 1.65 |
| Counterfactual-only | 27.67 | 61.80 | 80.14 | 49.36 | 3.02 |
| Ours, A+S | **40.41** | **83.53** | 94.74 | **62.72** | **1.34** |
| Ours+CF | 39.22 | 82.51 | **95.42** | 62.11 | 1.35 |

表5说明：Ours, A+S 是最终主方法。相比 Attention-only，Ours, A+S 的 Top-1 提升 11.21 个百分点，Top-3 提升 25.81 个百分点，MRR 提升 14.05 个百分点，IFA 从 3.31 降到 1.34，约降低 59.4%。Ours+CF 在 Top-5 上最高，但不作为最终综合最优方法。

## 表6 semantic_weight 参数敏感性分析

**表6 semantic_weight 参数敏感性分析**

| semantic_weight | Top-1 / % | Top-3 / % | Top-5 / % | MRR / % | IFA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 34.47 | 69.44 | 88.46 | 55.69 | 2.11 |
| 0.5 | 35.65 | 79.97 | 93.21 | 58.77 | 1.56 |
| 0.7 | 37.86 | 80.65 | **94.91** | 61.20 | 1.39 |
| 0.9 | **40.41** | **83.53** | 94.74 | **62.72** | **1.34** |

表6说明：$\lambda=0.9$ 在 Top-1、Top-3、MRR 和 IFA 上表现最好，因此作为本文最终主方法参数。$\lambda=0.7$ 的 Top-5 略高，但综合主排序效果仍以 $\lambda=0.9$ 更优。

## 表7 反事实扰动分析

**表7 反事实扰动分析**

| 数据集 | 方法 | Top-1 / % | Top-3 / % | Top-5 / % | MRR / % | IFA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SARD | Counterfactual-only | 27.67 | 61.80 | 80.14 | 49.36 | 3.02 |
| SARD | Ours, A+S | **40.41** | **83.53** | 94.74 | **62.72** | **1.34** |
| SARD | Ours+CF | 39.22 | 82.51 | **95.42** | 62.11 | 1.35 |
| DiverseVul-subset | Counterfactual-only | 19.60 | 41.50 | 57.20 | 33.40 | 6.08 |
| DiverseVul-subset | Ours, A+S | **31.83** | **59.37** | 72.64 | **47.82** | **3.71** |
| DiverseVul-subset | Ours+CF | 30.94 | 58.24 | **74.07** | 47.06 | 3.76 |

表7说明：Ours, A+S 在两个数据集上均取得最佳 Top-1、Top-3、MRR 和 IFA。Ours+CF 在两个数据集上均取得最高 Top-5，说明反事实扰动有助于候选召回，但不作为最终主排序方法。

## 表8 案例分析示例

**表8 案例分析示例**

| Rank | Line | Code | Attention | Semantic | Final | Semantic Tags | CF Drop |
| ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |
| 1 | 116 | `strncpy(dest, src, len);` | 0.72 | 1.00 | 0.97 | dangerous_api, memory, boundary | 0.18 |
| 2 | 109 | `char dest[100];` | 0.65 | 0.50 | 0.52 | array | 0.04 |
| 3 | 92 | `data = atoi(input);` | 0.58 | 0.60 | 0.60 | input_api | 0.06 |

表8说明：第 1 位候选行同时具有较高注意力证据、最高语义风险和更明显的反事实概率下降，因此被排序到首位。该案例展示了语义风险分数对危险操作行排序的增强作用，以及 CF Drop 对候选行可信度验证的辅助作用。