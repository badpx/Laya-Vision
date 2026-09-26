# Laya 实现原理与视觉模态可行性研究

研究日期：2026-09-26。源码基线：`4066d5d5fbf08b66c6757ddeedbd797bd7655bc0`。下述源码链接固定到该提交，避免上游后续修改造成行号漂移。本文解释技术依据与实现边界；本次本地实施及实测结果见同目录配套报告。

**结论：视觉接入可行，但现有 Laya 权重是文本决策模型，没有直接理解像素的能力。** 本次实施采用本地 BLIP 图像描述前端，再将视觉描述与用户文本一起送入 Laya。它让系统能够接收图片、输出类型化决策，属于串联的视觉桥接方案；不代表已经训练出原生多模态 Laya。原生视觉特征融合在结构上可行，但需要新增模块及图文训练，无法靠添加图片参数或做一次维度投影获得可靠能力。

## 1. 先区分网站、当前模型与历史论文

用户提供的网站是第三方演示、评测和集成项目；模型来自 Convai Innovations / Nandakishor M 的上游仓库。网页中的游戏先由程序计算场景，再把场景写成自然语言；Laya 读取这些句子并分类。Tetris 的落点枚举、空洞计数也由代码完成。这些演示没有证明模型能够直接读取游戏截图。网页的 21 ms、200 倍速度和 322M 参数都依赖具体检查点、设备与比较基准，不能直接套用到新增视觉处理后的整个系统。[网站说明](https://brainfunctioncollapse.com/laya)

网站链接的两篇历史论文与当前源码有明显区别：

| 材料 | 实际研究内容 | 与当前发布实现的关系 |
|---|---|---|
| 2025-03，SalesRLAgent | 使用 Azure 3072 维文本嵌入、销售会话特征及策略/价值网络，预测销售转化概率；描述合成数据及不确定性估计 | 提供“输出决策概率而非生成回答”的历史方向，不是当前 ModernBERT + 选项标记头的完整架构说明 |
| 2025-09，Confidence-Aware Routing | 以 SmolLM2-360M 和 MiniLM 参考嵌入为基础，组合语义对齐、层间稳定性、学习置信分数，按阈值选择本地、RAG、大模型或人工 | 是生成前路由研究；当前 SDK 的 `Router` 主要按语言、显式参数及可选工作流匹配选择检查点，并不等同于这套三信号模型 |
| 当前 Laya 源码与模型卡 | 文本双向编码器、问题类型嵌入、额外 Transformer、动态选项评分，以及 RLCD/微调工具 | 解释本地实际运行行为时应优先依据这些材料 |

两篇论文分别见 [SalesRLAgent 全文](https://arxiv.org/html/2503.23303v1) 和 [Confidence-Aware Routing 全文](https://arxiv.org/html/2510.01237v1)。后者实验部分写明置信估计训练集为 72 条、训练 30 轮；其报告的指标不能作为当前 Laya 在任意业务上的保证。以上是材料之间的实现核对，不是对论文结论的独立复现实验。

## 2. 当前 Laya 如何完成一次决策

### 2.1 API 输入本质上是文本

`state` 可以是字符串、字典或列表。字典和列表经 `json.dumps(..., ensure_ascii=False)` 变成文本，并没有结构化传感器或图像分支。每个问题提供类型、说明及选项描述。模型不会执行 JSON 中的计算表达式，也不会因为文本里出现文件路径、URL 或 Base64 就打开图片。[`serialize_state`，common.py:33](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/laya/common.py#L33)

三种类型的区别主要是选项构造和结果解码：

| 类型 | 模型内部选项 | API 输出 |
|---|---|---|
| `choice` | 请求提供的类别键及描述 | 最大概率类别、全部选项概率 |
| `score` | `level 0`、`level 1`……对应的等级描述 | 各等级概率以及期望分数 `Σ i·p(i)`，不是无限精度数值回归 |
| `noul` | 假、真两个选项；默认有对应语义描述 | `p(true)`；可通过 `labels` 修改显示文本但保留假/真顺序 |

因此动态问题并非为每个业务新建一个固定类别线性层。选项文字本身进入编码器，评分器对每个选项使用相同参数，运行时可改变选项集合。[`render_options`，common.py:65](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/laya/common.py#L65)

### 2.2 输入序列和 token 预算

每个问题形成一条序列：

```text
[CLS] <问题类型> question: <问题说明> [SEP]
[MASK] <选项 0> [MASK] <选项 1> ... [SEP]
<state 的文本> [SEP]
```

`[MASK]` 在这里是定位选项的标记。模型并不使用语言模型词表逐个补全这些位置，也不生成选项名称。它直接取这些位置的隐藏向量作为选项表示。

`head_max_len` 控制问题及选项预算，`max_len` 控制整条序列。源码首先限制每个选项的文本长度，预算不够再缩短选项，剩余空间分配给状态。选项多且描述长时，关键区分词会被截断；文档过长时，证据也可能消失。列表形式的对话会优先保留末尾的新消息，一般字符串则从尾部截断。[`build_sequence`，common.py:94](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/laya/common.py#L94)、[`_encode_state`，agent.py:650](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/laya/agent.py#L650)

视觉桥接产生的描述同样占用这个预算。长 OCR 文本、多个图像描述以及用户指令不能无限追加，应明确长度限制及截断行为。

### 2.3 编码器和决策头

核心计算为：

```text
token IDs / attention mask
        ↓
文本双向 Transformer 编码器
        ↓
每个 token 加入问题类型 embedding
        ↓
额外的 TransformerEncoder 层
        ↓
提取各个 [MASK] 位置
        ↓
LayerNorm → Linear → GELU → Linear(1)
        ↓
每个选项一个 logit → 温度缩放 → softmax
```

当前配置通常使用两层额外 Transformer；类型 embedding 有三种，对应 `choice`、`score`、`noul`。这可以看作“问题、选项、状态共同编码的动态分类器”。双向注意力让每个选项标记同时看到问题、其他选项和状态，因而不是对文档向量做一次固定标签匹配。[`DecisionModel`，common.py:149](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/laya/common.py#L149)、[`forward`，common.py:181](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/laya/common.py#L181)

模型另有一个 `act_head`：拼接首 token 表示以及最大概率、前两名差值、归一化熵、选项数等特征，再输出 act/escalate logits。该分支存在不代表已经具备可靠的拒答或转人工能力。

### 2.4 “一次前向”指批量执行，不是所有问题只编码一次状态

一个请求中的每个问题都有自己的完整序列，状态文本在这些序列中重复。SDK 可以复用状态的分词结果，再把问题序列合并成 batch 执行前向。计算量仍随问题数量和实际序列长度增加；状态的上下文表示没有被所有问题共享为一次编码的缓存。这与自回归模型逐 token 解码的代价不同，但不能理解为问题数量增加没有成本。[agent.py:650](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/laya/agent.py#L650)

### 2.5 检查点与路由

模型卡列出的英文/typed-decisions 检查点采用 ModernBERT-large，总参数约 421M；多语言检查点采用 mmBERT-base，总参数约 322M。常用默认输入上限分别是英文 512、其他两种 1024 token；多语言编码器可设置更长上下文，但速度和有效证据利用都需要重新评测。[官方模型卡](https://huggingface.co/convaiinnovations/laya)

当前 `Router` 是外围检查点选择器：处理显式 `model`、`task`、`lang` 等覆盖项，再利用语言检测决定英文或多语言路径；特定工作流的自动选择需要开启对应选项。它不是在每层 Transformer 内学习的 MoE 路由器。[`Router.route`，router.py:431](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/laya/router.py#L431)

## 3. RLCD 与概率：训练目标有依据，可靠性仍需实测

### 3.1 概率分布就是训练时的动作

可把每个问题的 logits 记为 `z`。公开微调 notebook 对其加高斯噪声，并投影为有效选项上的零均值扰动，得到同一问题的一组候选分布：

```text
εg ~ projected Gaussian
qg = softmax(z + εg)
rg = reward(qg, target)
Ag = rg - mean_g(rg)
```

奖励由三项组成：对数评分、球面评分，以及仅用于有序等级题的负 ranked probability score。后者比较预测和目标的累积分布，从而区分“错一个等级”和“跨多个等级”的距离。[`proper_reward`，common.py:278](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/laya/common.py#L278)

理想条件下，proper scoring rule 的期望最优解是真实条件分布，因而有鼓励诚实报告不确定性的统计依据。但以下条件不能省略：目标标签须可信、分布须匹配、模型有足够表达能力、优化须达到足够好的解。运行代码还对 log score 做了数值截断，因此不应把未截断评分规则的严格理论直接当作整个发布模型的无条件保证。

### 3.2 公开微调不是纯强化学习

notebook 的优化项为 group baseline 的策略梯度损失，加权重为 1.0 的 soft cross entropy。噪声组大小为 4，按训练进度降低探索噪声。它是 RL 项与监督式分布拟合的组合，不能把全部能力提升只归因于 RL。[训练代码，notebook:269](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb#L269)、[损失，notebook:335](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb#L335)

这份 notebook 对 `act` 的损失系数是零。虽然该项使训练图包含此分支，它不会为 act/escalate 提供有效监督。多轮会话的 `td_lambda_targets` 则是一个独立工具：`λ=1` 时向前传播终局标签，不意味着推理时在后台保存并学习会话状态。[`td_lambda_targets`，common.py:307](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/laya/common.py#L307)

### 3.3 三个容易混淆的字段

| 字段 | 当前代码含义 | 解释边界 |
|---|---|---|
| `probabilities` / `noul` | 对选项 logits 温度缩放后的分布 | 需要目标领域验证，不能自动视为真实世界频率 |
| `answer_confidence` | `max(p)`，各题型定义一致 | 与仓库校准评测使用的量一致；域内校准后才适合按正确率解释 |
| `confidence` | `choice`/`score` 为 `1-H(p)/log K`；`noul` 为 `max(p)` | 前者只表示分布集中度，不是答案正确的概率，不能混用同一门限 |
| `action.act_probability` | 独立动作头的输出 | 不能替代经过验证的拒答策略 |

具体解码见 [agent.py:760](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/laya/agent.py#L760)，置信定义见 [common.py:342](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/laya/common.py#L342)。温度缩放通常改变概率尖锐程度，保持同一问题内的 argmax；它不能纠正读错图像、选错类别、漏掉证据等错误。

官方模型卡也明确列出实际限制：typed-decisions 的两个基础模型准确率约 0.362/0.342，低于其多数类基线 0.461；0.766 来自对应领域微调模型。动作头几乎总输出 1，报告 AUROC 约 0.30；原始概率过度自信，需要领域内温度校准。网站“零 token 生成”能避免输出格式解析错误，不能消除误判。[模型卡的 Honest Limits](https://huggingface.co/convaiinnovations/laya#honest-limits)

## 4. 为什么当前权重不能直接看图，为什么原理上可以扩展

源码前向接口仅接受 `input_ids`、文本 attention mask、选项位置、选项 mask 和问题类型。没有图像 processor、`pixel_values`、视觉编码器或经过训练的跨模态投影。任意图像 embedding 的数值与维度即使恰好匹配文本隐藏维度，其坐标系、分布和语义也不同；随机线性层只能对齐形状，不能对齐意义。

另一方面，输出离散选项概率并不要求输入必须是文本。分类器可以对像素特征、音频特征或融合特征输出相同形式的 softmax。RLCD 的奖励也只依赖预测分布和标签，并不检查输入模态。因此限制来自现有模型的输入模块与训练数据，而不是“非自回归”或“System 1”这个形式。已有 ViLT 使用联合图文 Transformer 完成视觉语言理解，说明双向编码加任务头可以处理视觉问题。[ViLT 论文](https://arxiv.org/abs/2102.03334)

## 5. 可选技术路线与本次实施选择

| 路线 | 需要训练新 Laya 权重 | 保留什么 | 主要代价与限制 |
|---|---|---|---|
| OCR → Laya | 通常不需要 | 现有文本 API、检查点、决策头 | 适合截图和文档中的文字；视觉布局、对象关系可能丢失，OCR 错误继续传递 |
| 本地 caption/VQA → Laya | 通常不需要 | 现有 Laya 决策及业务问题接口 | 图像被压缩成描述；描述未提到的细节无法恢复；caption 阶段增加生成延迟 |
| CLIP/SigLIP 图文相似度 → 类型化输出 | 可做零样本原型；生产仍需校准 | 快速、非自回归视觉选项匹配 | 输出主要来自另一个视觉模型；直接包装结果不能称为 Laya 已学会视觉 |
| 视觉与 Laya 分数后融合 | 需要融合/校准数据 | 两侧独立模型 | 相似度与文本概率不是同一统计量，不能随意相乘；错误相关性需要验证 |
| 视觉 token + Laya 决策头联合训练 | 需要 | 原有动态选项机制、类型化输出，可维持非自回归 | 新增图文训练、显存和序列成本；原始校准参数不再适用 |

CLIP 使用对齐的图像编码器与文本编码器计算图文相似度，可按请求中的候选文本做零样本匹配；其 softmax 是给定候选集合上的相对分数，候选集合变化会改变数值。它对有限的视觉选项问题很有价值，但与完整业务文本理解仍有差距。[CLIP 论文](https://arxiv.org/abs/2103.00020)、[官方实现](https://github.com/openai/CLIP)

本次选择 BLIP caption 桥接，因为现成权重支持本地图片描述，且无需把未经训练的模块插入 Laya。BLIP 本身是图文预训练模型，可执行理解与生成任务；所选 caption 检查点用于将图片转为短文本。[BLIP 论文](https://arxiv.org/abs/2201.12086)、[Salesforce 模型卡](https://huggingface.co/Salesforce/blip-image-captioning-base)

```text
本地图片 → BLIP 视觉编码及描述生成 → 带来源的视觉描述 ┐
                                                       ├→ Laya 文本决策 → choice / score / noul
用户文本、上下文、问题与选项 ──────────────────────────┘
```

这条路径的端到端时间为图像预处理、视觉推理、文本决策及少量组装时间之和，冷启动还包括权重加载。新增系统有生成过程，不能沿用“整个系统零 token 生成”的说法。Laya 返回的置信度是基于描述文本计算的，并没有自动包含 BLIP 漏看、误描或编造细节的风险。即使 Laya 对描述判断正确，也可能对原始图片判断错误。

## 6. 如要训练原生多模态 Laya，建议如何做

以下是可实施的后续设计，不是本次已经训练或验证的能力。

1. **新增视觉编码器与投影。** 使用已有图文预训练视觉编码器，取 patch tokens，经可训练投影映射到 Laya head 的隐藏维度。先冻结视觉主干，控制训练内存和遗忘。
2. **优先在现有 head 前融合。** 文本仍由原编码器处理；将文本隐藏序列与视觉 tokens 拼接后送入新增/扩展的融合层。保留每个选项的 `[MASK]` 位置以及共享 scorer。添加模态 embedding、有效图像 token mask，并保留视觉位置线索。这比直接把随机视觉向量塞进未适配的文本词嵌入层更容易做受控实验。
3. **建立图像、状态文本、动态问题、选项和监督分布的数据集。** 覆盖物体/属性、是否存在、空间关系、等级判断和图文矛盾；根据实际领域加入 OCR、计数与图表任务。训练时改变选项顺序、标签措辞和数量，防止记住选项位置或文字捷径。
4. **分阶段训练。** 先训练投影及融合头，用监督损失学会图文对应；再按验证结果选择解冻部分主干或参数高效微调。稳定后才比较加入 RLCD 是否改进 Brier、NLL、选择性准确率，而不是预设 RL 必然更好。动作头如需拒答，必须单独提供可监督的正确性与成本目标。
5. **划分独立校准集。** 按图像来源/场景划分训练、校准、测试数据，避免同一图片裁剪版本跨集合。对新模型重拟合温度并检查不同题型、选项数、语言和图像领域的校准；旧文本温度不能直接继承为已验证参数。
6. **做视觉依赖性消融。** 同一问题换不同图片应改变答案；遮蔽关键区域、图像与描述冲突、去掉视觉输入时应出现可解释变化。仅输出看似合理答案、训练损失下降或随机模型前向成功，都不足以证明真的利用了图像。

视觉 tokens 会使融合层的序列变长。普通全局注意力的核心计算随总长度近似平方增长，多个问题批量执行还会重复处理融合序列。需要在 patch 数量、分辨率、问题批量、细节识别和时延之间取舍。因此不能提前承诺保留原文本模型的 21/33 ms 延迟。

## 7. 如何解释本次和后续的可用性测试

最低限度应分别记录三层结果：视觉前端描述是否符合图片，Laya 基于这些描述是否完成要求，端到端对原图的答案是否正确。三者分开才能知道错误来自感知、文本判断还是输入截断。

适合桥接方案的测试包括普通物体/场景识别，以及描述明确包含证据的文本与图像联合决策。应主动加入失败探针：小字、精确数量、细微空间关系、描述遗漏的属性、多张图比较和图文冲突。对于图片中的文字，单独的 OCR 前端通常比要求通用 caption 模型逐字识别更符合任务目标。

还应比较真实图片、空白图片和交换图片的结果；这些检查可以防止测试样例的文件名、用户状态或候选标签泄露答案。报告须注明样本量、模型/源码版本、设备、冷/热时延及失败样例。少量 smoke test 能证明本地链路可运行并展示能力边界，不能代替代表性业务评测或宣称通用视觉准确率。

最终，Laya 的长处是将限定答案空间中的语言判断变成快速的可组合接口。增加视觉的关键工作是提供经过训练且可验证的视觉证据路径，并对整个链路重新评估；类型化输出与非自回归决策机制本身不会阻止多模态扩展。
