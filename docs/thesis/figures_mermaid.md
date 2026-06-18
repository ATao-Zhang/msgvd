# 论文流程图 Mermaid 草稿

本文档整理第4章和第5章可使用的 Mermaid 图示。图中仅描述已实现和已验证的方法流程，不扩展未完成实验，也不夸大反事实模块作用。

## 图1 方法总体框架

**图1 基于语义增强执行路径证据的漏洞行级定位与可解释方法总体框架**

```mermaid
flowchart TD
    A["输入 XFG 样本"] --> B["加载已训练 MSAVD / DeepWuKong 模型"]
    B --> C["模型前向传播"]
    C --> D["输出漏洞预测概率 P_vul"]
    C --> E["提取节点级执行路径证据"]
    E --> F["节点证据映射到源代码行"]
    F --> G["计算 AttentionScore"]

    A --> H["读取源代码行文本"]
    H --> I["计算 SemanticScore"]

    G --> J["分数归一化"]
    I --> J
    J --> K["A+S 融合排序"]
    K --> L["输出 Top-k 候选漏洞行"]

    L --> M["Top-k 候选行反事实扰动"]
    M --> N["计算概率下降 cf_score"]
    N --> O["候选行可信度验证"]
```

图注：输入样本首先经过已训练漏洞检测模型获得样本级漏洞概率，并提取节点级执行路径证据。随后，节点证据映射为行级 Attention Score，代码文本被用于计算 Semantic Score。本文主方法采用 A+S 融合分数生成 Top-k 候选漏洞行；反事实扰动仅用于候选行可信度验证和 Top-5 召回增强，不作为最终主排序模块。

## 图2 语义风险分数计算流程

**图2 语义风险分数计算流程**

```mermaid
flowchart TD
    A["输入单行源代码文本"] --> B["危险拷贝 / 字符串 API 检测"]
    A --> C["内存分配与释放 API 检测"]
    A --> D["输入源与外部输入函数检测"]
    A --> E["数组访问与指针访问检测"]
    A --> F["长度 / 边界相关函数检测"]
    A --> G["整数与算术操作检测"]

    B --> B1["dangerous APIs: +1.0"]
    C --> C1["memory APIs: +0.7"]
    D --> D1["input APIs: +0.6"]
    E --> E1["array / pointer: +0.5"]
    F --> F1["length / boundary: +0.4"]
    G --> G1["arithmetic: +0.2"]

    B1 --> H["规则得分求和"]
    C1 --> H
    D1 --> H
    E1 --> H
    F1 --> H
    G1 --> H

    H --> I["clip 到 [0, 1]"]
    I --> J["输出 SemanticScore 与 semantic_tags"]
```

图注：Semantic Score 基于代码行文本中的安全风险语义规则计算。不同风险类型对应不同权重，最终得分裁剪到 \([0,1]\) 区间，以避免多规则命中导致分数无界增长。该分数用于补充模型内部注意力证据，使候选漏洞行排序更符合安全审计中的危险语句判断。

## 图3 反事实验证流程

**图3 轻量反事实扰动验证流程**

```mermaid
flowchart TD
    A["A+S 主排序结果"] --> B["选取 Top-k 候选行"]
    B --> C["定位候选行对应的 XFG 节点"]
    C --> D{"Data.x 是否为二维 token 序列?"}

    D -- "是" --> E["执行 length-preserving token mask"]
    E --> F["非 PAD token 替换为 UNK / MASK / 合法非 PAD token"]
    F --> G["PAD 位置保持不变"]
    G --> H["可选: stmt_features 置零"]
    H --> I["重新前向传播"]
    I --> J["得到 P_masked"]
    J --> K["cf_score = max(0, P_original - P_masked)"]
    K --> L["输出 masked_prob / prob_drop / cf_score"]

    D -- "否" --> M["记录 warning"]
    M --> N["cf_score=0, masked_prob=null, prob_drop=0"]

    I --> O{"单行扰动是否失败?"}
    O -- "失败" --> M
    O -- "成功" --> J
```

图注：反事实验证仅对主排序得到的 Top-k 候选行执行轻量扰动。为避免 RNN 编码器接收到长度为 0 的输入，本文采用 token-level length-preserving mask：原始 PAD 位置保持不变，非 PAD token 替换为 UNK、MASK 或合法非 PAD token。若候选行无法安全扰动或单行扰动失败，则记录 warning，并将该行反事实分数置为 0。该模块用于验证候选行对模型预测概率的影响，而非替代 A+S 主排序方法。