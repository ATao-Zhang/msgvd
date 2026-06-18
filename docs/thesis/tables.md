# 小论文实验表格汇总

本文档整理第4章实验部分可直接引用的表格。所有已填数值均来自真实实验结果；外部基线、DiverseVul-subset 指标和案例级结果若尚未提供真实运行输出，则仅保留“待填”位置，不编造结果。

## 表1 数据集与行级标签统计

**表1 实验数据集与行级标签构建统计**

| 数据集 | 总样本数 | 正样本数 | 负样本数 | 有行级标签的正样本数 | 定位评估样本数 | 行级标签来源 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SARD test | 2648 | 589 | 2059 | 589 | 589 | xfg_stem |
| DiverseVul-subset | 1500 | 500 | 1000 | 500 | 500 | patch diff 近似行级标签 |

表1说明：SARD 采用 `xfg_stem` 策略从 XFG 文件名中提取漏洞行号，例如 `116.xfg.pkl -> true_vul_lines=[116]`。负样本不生成漏洞行标签，只有正样本参与 Top-k、MRR 和 IFA 定位评估。DiverseVul-subset 来源于真实开源项目漏洞修复提交，行级标签由补丁 diff 中被删除或被修改的代码行映射回漏洞函数内部得到，并过滤补丁过大、修改范围过宽、无法稳定映射和 tangled patch 严重的样本。

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

表2说明：外部基线需要在相同数据划分、相同行级标签构造和相同评价指标下复现后再填入结果。本文最终主方法是 Ours, A+S，而不是 Ours+CF。

## 表3 SARD 主结果与外部基线对比表

**表3 SARD 上主结果与外部基线对比**

| 方法 | Top-1 Acc | Top-3 Acc | Top-5 Acc | MRR | IFA | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Random | 待填 | 待填 | 待填 | 待填 | 待填 | 最低参考基线 |
| API-Heuristic | 待填 | 待填 | 待填 | 待填 | 待填 | 危险 API 启发式 |
| Gradient Saliency | 待填 | 待填 | 待填 | 待填 | 待填 | 梯度归因基线 |
| Integrated Gradients | 待填 | 待填 | 待填 | 待填 | 待填 | 积分梯度基线 |
| GNNExplainer | 待填 | 待填 | 待填 | 待填 | 待填 | 图解释基线 |
| Attention-only | 0.2920 | 0.5772 | 0.7759 | 0.4867 | 3.3141 | 内部基础对照 |
| Ours, A+S ($\lambda=0.9$) | 0.4041 | 0.8353 | 0.9474 | 0.6272 | 1.3447 | 本文最终主方法 |

表3说明：当前已确认的真实结果为 SARD 上的内部方法结果。Random、API-Heuristic、Gradient Saliency、Integrated Gradients 和 GNNExplainer 需要完成同协议复现后再填入。本文主方法为 Ours, A+S，Ours+CF 不作为最终主排序方法。

## 表4 DiverseVul-subset 泛化实验表

**表4 DiverseVul-subset 上的泛化实验结果**

| 方法 | Top-1 Acc | Top-3 Acc | Top-5 Acc | MRR | IFA | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Random | 待填 | 待填 | 待填 | 待填 | 待填 | 最低参考基线 |
| API-Heuristic | 待填 | 待填 | 待填 | 待填 | 待填 | 危险 API 启发式 |
| Attention-only | 待填 | 待填 | 待填 | 待填 | 待填 | 内部基础对照 |
| Gradient Saliency | 待填 | 待填 | 待填 | 待填 | 待填 | 梯度归因基线 |
| Integrated Gradients | 待填 | 待填 | 待填 | 待填 | 待填 | 积分梯度基线 |
| GNNExplainer | 待填 | 待填 | 待填 | 待填 | 待填 | 图解释基线 |
| LineVul | 待填 | 待填 | 待填 | 待填 | 待填 | 行级检测基线 |
| Ours, A+S ($\lambda=0.9$) | 待填 | 待填 | 待填 | 待填 | 待填 | 本文主方法 |

表4说明：DiverseVul-subset 的行级标签为 patch diff 构造的近似标签，因此应作为泛化趋势验证，而不应与 SARD 的 `xfg_stem` 精确行号标签完全等同。真实结果需要服务器完成实验后再填入。

## 表5 消融实验表

**表5 SARD 上不同定位策略的消融实验结果**

| 方法 | Top-1 Acc | Top-3 Acc | Top-5 Acc | MRR | IFA | 作用定位 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Attention-only | 0.2920 | 0.5772 | 0.7759 | 0.4867 | 3.3141 | 模型证据基线 |
| Semantic-only | 0.3531 | 0.7912 | 0.9168 | 0.5791 | 1.6486 | 语义规则贡献 |
| Counterfactual-only | 0.2767 | 0.6180 | 0.8014 | 0.4936 | 3.0153 | 单独排序稳定性不足 |
| Ours, A+S ($\lambda=0.9$) | 0.4041 | 0.8353 | 0.9474 | 0.6272 | 1.3447 | 最终主方法 |
| Ours+CF, Top5-best | 0.3922 | 0.8251 | 0.9542 | 0.6211 | 1.3463 | Top-5 召回增强 |

表5说明：Ours, A+S 在 Top-1、Top-3、MRR 和 IFA 上取得最佳主排序结果。Ours+CF 的 Top-5 Accuracy 最高，但其 Top-1、Top-3 和 MRR 略低于 Ours, A+S，因此 CF 模块不作为最终主排序方法。

## 表6 semantic_weight 参数敏感性表

**表6 Attention + Semantic 中语义权重 $\lambda$ 的参数敏感性分析**

| semantic_weight $\lambda$ | Top-1 Acc | Top-3 Acc | Top-5 Acc | MRR | IFA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 0.3447 | 0.6944 | 0.8846 | 0.5569 | 2.1121 |
| 0.5 | 0.3565 | 0.7997 | 0.9321 | 0.5877 | 1.5637 |
| 0.7 | 0.3786 | 0.8065 | 0.9491 | 0.6120 | 1.3922 |
| 0.9 | 0.4041 | 0.8353 | 0.9474 | 0.6272 | 1.3447 |

表6说明：$\lambda=0.9$ 在 Top-1、Top-3、MRR 和 IFA 上表现最好，因此作为本文最终主方法参数。虽然 $\lambda=0.7$ 的 Top-5 Accuracy 略高，但综合主排序效果仍以 $\lambda=0.9$ 更优。

## 表7 反事实扰动对比表

**表7 反事实扰动模块的定位效果对比**

| 方法 | Top-1 Acc | Top-3 Acc | Top-5 Acc | MRR | IFA | 作用定位 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Ours, A+S ($\lambda=0.9$) | 0.4041 | 0.8353 | 0.9474 | 0.6272 | 1.3447 | 最终主排序方法 |
| Counterfactual-only | 0.2767 | 0.6180 | 0.8014 | 0.4936 | 3.0153 | 单独使用时排序稳定性不足 |
| Ours+CF, Top5-best | 0.3922 | 0.8251 | 0.9542 | 0.6211 | 1.3463 | Top-5 召回增强与可信度验证 |

表7说明：Counterfactual-only 不适合作为最终主排序方法。Ours+CF 提高了 Top-5 Accuracy，但没有超过 Ours, A+S 的 Top-1、Top-3 和 MRR。因此，反事实扰动更适合作为候选行可信度验证和 Top-5 召回增强模块。

## 表8 相比 Attention-only 的提升表

**表8 Ours, A+S ($\lambda=0.9$) 相比 Attention-only 的性能提升**

| 指标 | Attention-only | Ours, A+S ($\lambda=0.9$) | 提升情况 |
| --- | ---: | ---: | --- |
| Top-1 Acc | 0.2920 | 0.4041 | 提升 11.21 个百分点 |
| Top-3 Acc | 0.5772 | 0.8353 | 提升 25.81 个百分点 |
| Top-5 Acc | 0.7759 | 0.9474 | 提升 17.15 个百分点 |
| MRR | 0.4867 | 0.6272 | 提升 14.05 个百分点 |
| IFA | 3.3141 | 1.3447 | 约降低 59.4% |

表8说明：Ours, A+S 相比 Attention-only 在 Top-k Accuracy 和 MRR 上均取得明显提升，同时显著降低 IFA。IFA 从 3.3141 降至 1.3447，说明安全分析人员在找到第一个真实漏洞行前需要检查的非漏洞候选行数量明显减少。

## 表9 案例分析表模板

**表9 漏洞行级定位案例分析模板**

| 样本 ID | 数据集 | 真实漏洞行 | Attention-only 排名 | Ours, A+S 排名 | 命中的语义风险标签 | cf_score / prob_drop | 案例类型 |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| 待填 | SARD 或 DiverseVul-subset | 待填 | 待填 | 待填 | 待填 | 待填 | 成功案例或失败案例 |

表9说明：案例分析应从真实输出结果中选择样本，不应编造样本 ID、代码行或概率下降值。建议同时选择成功案例和失败案例，以展示方法有效性和边界。