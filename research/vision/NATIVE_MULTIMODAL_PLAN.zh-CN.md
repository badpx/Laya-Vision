# 原生多模态 Laya：架构、训练与验收方案

调研日期：2026-09-26。基于当前本地 Laya 源码 `4066d5d5fbf08b66c6757ddeedbd797bd7655bc0`、现有多语言权重与第一轮视觉桥接实测。

**建议路线：保留 Laya 的多语言文本编码器和动态选项决策头，增加 SigLIP 2 的密集视觉 patch 特征，以及问题驱动的门控交叉注意力。先用监督数据建立可靠基线，再决定是否加入 RLCD。** 同时短跑一个 ModernVBERT 已对齐主干的对照，避免把所有预算花在从头学习融合上。

本文件是设计与研究结果。这里没有声称已经训练、测速或验证新的原生模型；学习率、数据规模、显存档位和验收门槛均为建议起点，须由试验修正。此前 BLIP 实测不能用来推断新模型的准确率或延迟。

## 1. 真正要解决的问题

当前图片先被压成一句 caption，Laya 只能对这句话判断。此前细节探针的 3/5 与人工细节描述的 5/5，提示感知信息缺失确实构成瓶颈，但不足以证明某个原生结构必然更好。

新模型应该实现：

- 在推理时直接读取视觉特征，问题和各候选答案可以访问不同图像区域；不依赖生成描述才能决策。
- 继续返回 `choice`、`noul`、`score`，不需要自回归生成自由文本。
- 同一张图换问题，能够提取不同细节；同一问题换相关图像细节，答案应相应改变。
- 保留现有纯文本路径和多语言能力，并对新增视觉域重新评估概率。

“原生”不等于无损保存原始像素：缩放、patch 化、特征编码、token 压缩仍可能丢信息。必须一起设计**分辨率、视觉 token 保留量、空间编码、监督数据、评测**，不能只添加一个投影层。

## 2. 相关先例与路线选择

| 路线 | 已有依据 | 对 Laya 的适配成本 | 本次建议 |
|---|---|---|---|
| 密集视觉特征 + 门控交叉注意力 | Flamingo 展示了冻结主干、通过可训练交叉注意力连接视觉与语言的办法 | 保留 Laya tokenizer、mmBERT、选项头；需学习融合 | 主方案，优先保护中文与既有 API |
| 文本编码后，把 patches 拼入两层决策 head | 最直接的末端融合对照 | 保留 mmBERT；把视觉放在文本后可保留选项位置，但 head 的序列变长 | 小规模结构对照，与主方案同数据比较 |
| 视觉 token 直接进入 mmBERT | 双向图文融合在 ModernVBERT 中可行；本地 ModernBert 支持 `inputs_embeds` | 改 attention/位置/marker，视觉 token 经过整个文本主干；需图文对齐 | 更大预算时的结构对照 |
| ModernVBERT 基础主干 + Laya 风格动态头 | 现成约 250M 图文编码器，已有对齐权重 | 不能原样复用 Laya tokenizer/全部权重；任务头仍需训练 | 必做的小规模对照，文档场景尤有价值 |
| Q-Former / 少量 query token 压缩 | BLIP-2 用可学习查询连接冻结视觉和语言模型 | 可减 token 成本，但查询数又形成容量瓶颈 | 首版先保留 patch；确认瓶颈后再做压缩消融 |

依据：[Flamingo](https://arxiv.org/abs/2204.14198)、[BLIP-2](https://arxiv.org/abs/2301.12597)。这里借鉴连接方式，没有照搬它们的生成模型、数据规模或性能结论。

ModernVBERT 的基础模型在 Ettin/ModernBERT 类文本主干与 SigLIP 2 间做过图文对齐，另有文档检索专门版本。应从 `ModernVBERT/modernvbert` 基础版比较，而不是把检索向量分数直接包装成 Laya 概率。其检索成绩不证明动态 VQA 或中文判断能力，论文的部分比较只使用英文；需要另测中文与自然图像。当前官方模型卡列出 Transformers 原生支持，不必假设一定需要执行自定义远程模型代码。[论文](https://arxiv.org/abs/2510.01149)、[官方模型卡](https://huggingface.co/ModernVBERT/modernvbert)、[训练代码](https://github.com/illuin-tech/modernvbert)

直接替换主干也会改变隐藏表示的分布。即使某层维度同为 768，也不代表现有 Laya scorer、temperature 可以原样继承且保持能力。对照须分别尝试重训任务头与迁移头初始化，并重新校准。

ModernVBERT 也有视觉压缩：论文的每个 512×512 tile 先得到 32×32 patches，再把相邻 4×4 patches 重排并投影，得到 64 个视觉 token。空间重排本身不丢信息，后续降维有压缩；全局缩略图和多个 tiles 则增加总 token 数。因此它是已有对齐能力的候选，不是“天然保留全部细节”的保证。处理图片占位符后，应重新定位动态选项 markers，不能沿用 mmBERT tokenizer 的 offset。[架构说明](https://arxiv.org/html/2510.01149v2)

## 3. 推荐的首版网络

### 3.1 起点与张量形状

本地多语言 Laya 的真实配置为 `jhu-clsp/mmBERT-base`、隐藏维度 768、22 层文本编码器、2 层额外决策 Transformer；默认 `max_len=1024`、`head_max_len=256`。见本地 checkpoint 配置及 [`DecisionModel.forward`](../../laya/common.py#L181)。

视觉侧建议从 `google/siglip2-base-patch16-384` 的**视觉塔**起步，使用 patch 的 `last_hidden_state`，不使用全局 pooled embedding、不加载其文本塔来替代 Laya。384×384、16×16 patch 对应 24×24=576 个视觉 token，隐藏维度 768。该 FixRes 检查点配置属于 `siglip` 架构，使用 `SiglipVisionModel`；NaFlex 属于 `siglip2`，不能仅凭名称强行使用同一个 processor/类。显式关闭不使用的 MAP pooling head；本地 Transformers 5.17 支持 `vision_use_head=False`，实际模块嵌套与旧版本示例不同，加载时按当前配置和 state dict 核对。[固定分辨率配置](https://huggingface.co/google/siglip2-base-patch16-384/raw/main/config.json)、[官方 Transformers 说明](https://huggingface.co/docs/transformers/model_doc/siglip2)

设 `B` 为图片数，每图 `Q` 个问题，`L` 为每个问题的完整文本序列长度，`N=576`：

```text
图片像素 ── SigLIP 2视觉塔 ── V[B,N,768] ── 投影层 ─────────┐
                                                         │ K、V
问题+选项+业务文本 ── mmBERT ── H[B*Q,L,768]                 │
                           ↓                             │
                    原决策Transformer第1层                 │
                           ↓                             │
                    门控 CrossAttention  ←───────────────┤
                           ↓                             │
                    原决策Transformer第2层                 │
                           ↓                             │
                    门控 CrossAttention  ←───────────────┘
                           ↓
                 各选项[MASK]隐藏向量 → 共享scorer
                           ↓
                   choice / noul / score
```

推荐连接器：`LayerNorm → Linear(768,768) → GELU → Linear(768,768)`；交叉注意力使用 12 heads，Query 来自问题/候选/上下文的文本 hidden states，Key/Value 来自同一图的视觉 patches。首版不再添加额外 FFN，以控制参数和消融变量。

每层大致计算：

```text
H = original_head_layer(H, text_padding_mask)
H = H + tanh(gate) * cross_attention(LayerNorm(H), V, V, vision_padding_mask)
logit_i = scorer(H[option_marker_i])
```

这是可训练的视觉条件化决策网络，图像信息在 scorer 之前参与计算。caption 不是必经路径。上式是设计伪代码，不是当前仓库已实现的函数。

### 3.2 为什么把融合先放在决策头

- 现有文本序列与 `[MASK]` 的位置不变，避免第一版同时修改 tokenizer 和图片占位 token 机制。
- 576 个 patch 不必经过 22 层 mmBERT，保留原文本主干的上下文与滑窗注意力设置。
- 一图多问只运行一次视觉塔。问题相关的文本编码和融合仍要逐问题/批量执行，不能声称全部计算共用一次。
- 没有图片时直接跳过新增分支；初始冻结旧模块时可以对纯文本输出做严格一致性检查。

代价是视觉只在末端进入，深层关系推理可能弱于主干内部交互。若细节识别改善但组合推理停滞，再比较提前融合，不能预设末端融合足够解决所有任务。

交叉注意力的新增注意力矩阵规模约为 `B*Q*L*N`，文本与视觉全拼接后每层全注意力约为 `B*Q*(L+N)^2`。这些只是注意力项的量级比较，不含视觉塔、线性投影、FFN 和内存访问，不能当成端到端加速比。

### 3.3 初始化、mask 与缓存

1. 旧文本主干、旧 head 和 scorer 从现有 checkpoint 读取；视觉塔从独立预训练 checkpoint 读取；新增连接器/交叉注意力正常随机初始化，gate 可从小值如 `1e-3` 起步。严格零 gate 可保持初始化等价，但首步会阻断该分支内部权重梯度，要检查随后 gate 是否打开。
2. 只对文本部分添加原 `type_emb`。首版视觉侧复用视觉塔已有空间表征，图文由独立支路区分，不新增可训练的模态/位置 embedding。将来切图时加入原图中的 tile 坐标、缩放比例及 patch 坐标，避免两块图里的“左上”混淆；那一版的额外参数需要重新计数。
3. 对变长 patches 和 tiles 显式传有效 mask。`MultiheadAttention.key_padding_mask=True` 表示屏蔽，SDPA 的布尔 attention mask 中 True 表示允许，应在接口处转换并测试；图像为空时跳过分支，避免全部 key 被 mask 后产生 NaN。MHA 设置 `need_weights=False`，不为未使用的注意力图增加开销。
4. 同图多问通过 `image_index` 映射到共享特征。推理时可复用该图的视觉 features 与每个融合层的 K/V 投影；不要把上一张图的缓存留给下一次调用。
5. 冻结视觉塔时可以预计算训练特征，缓存键必须包括图片内容哈希、checkpoint revision、processor/缩放参数与 tile 方案。解冻视觉塔、改变裁剪或位置编码后缓存失效；训练中的可训练投影结果不可长期缓存。
6. 精确空间、计数与颜色任务不能照搬通用图片增强：水平翻转须同步修改左右关系，随机裁剪会改变数量，颜色抖动会破坏颜色标签。无法同步修标签的增强不用于对应任务。

### 3.4 如何保住细节

首轮先在自然图像上用完整 576 patches，避免一开始又压成 32/64 个 query token。第二轮比较 384、512 分辨率和原图全局视图加局部 tiles；每次都重测同一细节集、显存、延迟。512 方案需使用支持该尺寸的 checkpoint，或显式启用并验证位置编码插值，不能只把 384 processor 的 size 改成 512 就默认有效。

对于长截图、票据、小字，单纯把整图缩到 384 不足。可选择 SigLIP 2 NaFlex 保持宽高比并提高 patch 预算，或保留全局图加高分辨率局部块。NaFlex 支持可变分辨率和宽高比，但仍有最大 patch 数，不能把它描述成无损方案。[Google 官方说明](https://google-research.github.io/big_vision/big_vision/configs/proj/image_text/README_siglip2.html)、[NaFlex 权重](https://huggingface.co/google/siglip2-base-patch16-naflex)

训练时出现过的局部/全局坐标与视图规则必须与推理相同。未经训练随意增加 tiles，不一定改善准确率。多视图还可能让计数模型重复数同一物体，应单独做“单图 vs 多视图”的计数测试。

## 4. 数据应围绕细节决策构建

### 4.1 第一阶段数据组合

| 数据 | 用途 | 转换与注意事项 |
|---|---|---|
| GQA balanced | 物体属性、关系、组合判断 | 利用场景图核对干扰选项，避免只学语言先验 |
| TallyQA | 自然图像简单/复杂计数 | 保留 simple/complex 分项；不把答案强行塞进错误的等级范围 |
| CLEVR 及可验证合成场景 | 属性、空间关系、数量、反事实 | 从场景图/程序重新计算标签，检验视觉依赖；不能替代自然图泛化 |
| VQAv2 小比例 | 自然提问、多标注者分歧 | 利用多标注者标签；明确这是转换后的类型化任务 |
| TextVQA/DocVQA，第二阶段 | OCR 与文档证据 | 要求高分辨率与额外文字训练，不把选择题改造分数称为官方开放问答成绩 |

一手入口：[GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html)、[TallyQA](https://github.com/manoja328/TallyQA_dataset)、[CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/)、[VQAv2](https://visualqa.org/download.html)、[TextVQA](https://arxiv.org/abs/1904.08920)、[DocVQA](https://www.docvqa.org/datasets/docvqa)。下载前记录各数据版本和许可；模型许可不覆盖图片来源许可。CLEVR 官方标明 CC BY 4.0；VQA 注释与来源图片分别核对；TallyQA 仓库的 Apache-2.0 不能自动覆盖来源图片。本次没有确认 GQA、TextVQA、DocVQA 的完整数据许可，实施下载/分发前补齐。

建议 pilot 使用约 **2万–5万张去重图片、10万–20万条图像问题**，只覆盖存在、颜色/属性、空间关系、有限范围计数四类；并非保证训练成功的数据下限。解冻旧模块后，训练流可先设 80% 图像问题、20% 原文本任务重放，中文与英文各有明确配额。先用更小的 5k/20k/100k QA 学习曲线判断是否值得扩量。

一个具体的 10 万原始 QA 配额是 GQA 5 万、TallyQA 2.5 万、CLEVR 1.5 万、VQAv2 1 万；按图片去重后检查独立图片数，不为凑比例破坏数据分区。翻译和候选置换不计作新增视觉事实。先人工抽查 500–1,000 条，再确定扩量；这套配额是本方案建议，不是引用论文的最佳配比。

中文变体可以来自本地翻译模型或人工翻译，但位置词、否定、量词和选项含义需核验。中英版本以及同图所有问答必须进入同一数据分区。

### 4.2 一个可落地的样本格式

以下是拟议数据格式，不是当前 SDK 新增接口：

```json
{
  "image_id": "source:image:123",
  "image": "images/123.jpg",
  "group_id": "original-image-and-near-duplicates",
  "language": "zh",
  "state": "只根据图片中可见内容判断。",
  "questions": {
    "cup_color": {
      "type": "choice",
      "instructions": "杯子外侧是什么颜色？",
      "criteria": {"red": "红色", "white": "白色", "blue": "蓝色", "other": "其他颜色"}
    }
  },
  "targets": {"cup_color": {"red": 1.0, "white": 0.0, "blue": 0.0, "other": 0.0}},
  "provenance": {"source": "dataset-name", "split": "train", "annotation_id": "..."}
}
```

标签、文件名、场景图和 provenance 只供采样、监督与审计，绝不能混入 `state` 让模型抄答案。视图坐标可以输入，真值框和对象列表不能作为未声明的额外视觉证据。

### 4.3 类型化转换的四个陷阱

- **候选集合泄漏。** 每次用真值生成“正确答案+几个明显错误答案”，再宣称开放视觉问答已解决是不成立的。候选协议须固定、独立于测试真值，或清楚标记为标准多选任务；实际部署需报告候选召回率。设置“以上皆非”并与“图像证据不足”区分。
- **有序等级不等于任意数值。** 当前 `score` 返回 `Σ index*p(index)`。只有类别为完整连续的 `0…C` 时，索引才等于数量。稀疏数量、`C+` 溢出、未知类别使用 `choice` 加显式数值解释，不能把未知塞在有序量表最后并计算伪精确期望。
- **VQA 标签与评估分数。** 可以将规范化后的标注者频次变成候选上的软目标；VQA 官方共识评分不是天然和为 1 的概率。保留候选外标注质量，不能悄悄删除后重新归一化制造确定性。
- **图中没有与看不清。** 二元 noul 的否类用于命题不成立；遮挡或分辨率不足并不必然为否。需要单独 answerability 任务/拒答逻辑，且部署时重新验证；现有 `act_probability` 不可直接充当可靠拒答。

动态选项训练应打乱 `choice` 的顺序与无语义 ID，并测试未见过的措辞、组合及合理候选数量；`score` 的等级顺序则必须保留。可靠整数计数用 one-hot，不机械给相邻数字加高斯平滑；相邻误差代价交给有序损失。VQA 的答案规范化复用[官方规则](https://visualqa.org/evaluation.html)。场景图遗漏不等于对象不存在，不能仅因没有标注就构造负例。

### 4.4 先分区，再派生数据

先按原始图片、近重复图片和文档/视频来源建立 group，再分 train/dev/calibration/test。遵守官方分区；多库混合时检查 Visual Genome/COCO 等重叠来源。所有问题、翻译、选项置换、裁剪和反事实关联样本随 group 分区。合成集还需按场景组合或生成模板留出，避免只是记住同一场景。

前一轮五张 smoke 图片继续作为已知回归案例，不加入“新模型独立泛化”统计。公开预训练语料中的污染无法完全排除，应另外留出用户目标域的新采集样本。

## 5. 分阶段训练，而不是直接上 RL

### 阶段 0：结构和梯度验证

用 128–256 条经验证的合成样本做故意过拟合检查。确认模型能学会、换图能改答案、投影/交叉注意力/gate 有梯度；无图时可走原路径，padding 和混合问题数不会串图。该阶段只调试程序，不报告泛化能力。

### 阶段 1：学习视觉融合

冻结 SigLIP 2 和 mmBERT，先冻结原 decision head/scorer，只训练新增连接器、交叉注意力与 gate。输入像素/patch，监督为类型化答案，不必先生成 caption。此时无图分支完全绕过新增模块，纯文本 loss 对它们没有梯度；文本数据只用于回归监测，正式的文本重放/蒸馏从阶段 2 解冻旧模块后开始。

冻结主干保持 `eval()`，并在调用父模型的 `train()` 后恢复这个状态；冻结权重本身不会关闭 dropout。原 action head 首版保持冻结、不得将其输出宣传为视觉拒答能力；只有定义独立行动/拒答标签和损失后才训练它。

建议 AdamW：新模块学习率 `1e-4` 起步、warmup 5%、梯度裁剪 1.0、有效 batch 128 个问题；微批先从 2 张图×每图 2 问开始，按显存增加。先跑小规模 pilot，看验证集和图像依赖性是否改善，再扩大。

若冻结原 head 后任务拟合停滞，进入阶段 2，而不是只继续加长同一训练。若需要一般图文对齐，可加入图像条件下的 masked-text 或局部属性预测辅助任务；caption 可作为**训练监督**，但不能把正确描述直接作为推理必需输入，也不能把完整答案给模型再预测同一答案。现有 Laya `AutoModel` 不含现成 MLM 输出头，该辅助分支要明确新增。

### 阶段 2：适配决策任务并保护文本能力

解冻原 decision head 和 scorer，主干仍冻结。新增模块可继续 `1e-4`，迁移来的 head/scorer 从 `1e-5` 开始；验证不够再对 mmBERT 最后 4 层做 rank 8/16 LoRA，必要时适配视觉塔末层。LoRA 目标模块按实际名称配置，本地 ModernBert 使用融合 `Wqkv`，不是通用的 `q_proj/v_proj`。

混入有标签的文本重放；可用原 Laya 在纯文本样本上产生教师分布，加入小权重 KL 来减少遗忘。教师蒸馏只保护旧能力，不能保证旧模型本来错误的预测被修正；有标签监督和独立文本回归更重要。

**推荐起始损失：**

```text
L = L_soft_CE(typed answers)
  + λ_ord * RPS(仅有序等级题)
  + λ_text * KL(仅纯文本重放的旧模型分布 || 新模型分布)
```

这里的 `L_soft_CE` 同时覆盖有标签的视觉样本与文本重放。对有效候选计算 `-Σ y_k log p_k`；有序 RPS 可用 `Σ_{k=0}^{K-2}(CDF_p(k)-CDF_y(k))²/(K-1)`，只对 K≥2 的真实等级计算，按语义等级顺序累计，padding 不参与损失。KL 只用于文本重放，不能让原文本教师的视觉错误约束新模型。监督真值使用数据标注，不以 BLIP caption 作为唯一真值。

`λ_ord=0.2`、`λ_text=0.1` 可作为首轮搜索中心，比较包含 0 的少量取值。任务按类型或来源平衡采样/加权，避免多数计数等级或二元标签主导梯度。

### 阶段 3：把 RLCD 当作受控消融

当前 `proper_reward` 对概率可微；有固定监督标签时，可以直接优化 CE、RPS 或负 proper reward，不必引入高方差策略梯度。公开 notebook 也实际使用 RL loss + CE，并非纯 RL。见 [`proper_reward`](../../laya/common.py#L278)、[微调 notebook](../../notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)。

从同一个阶段 2 checkpoint 出发，比较三个分支：基础监督损失（CE 及已固定的 ordinal RPS）；基础监督 + 同一 proper reward 的直接可微优化；基础监督 + 现有 noisy-logit RLCD。保持相同数据、可训练参数和更新数，再补充相同 GPU 时间预算的比较，建议 3 个随机种子。查看准确率、NLL、Brier、RPS和校准后的风险，不仅看训练奖励。不要一边改变分辨率/数据，一边把收益归因于 RLCD。新视觉域的旧温度先重置为 1，最终再独立拟合。

### 阶段 4：校准与部署

在没有参与梯度训练或超参选择的 calibration split 上，用完整 logits 与目标分布、以 NLL 等目标拟合温度；必要时按题型/选项数分桶，但每桶要有足够样本。在 untouched test 上验证。计算答案置信度、ECE 和 risk-coverage 时使用 `max(p)`/统一 `answer_confidence`，不要混用旧 `confidence` 的熵集中度定义。视觉噪声、未知对象和中文需分别报告。

## 6. 必须做的对照与验收

至少保留下列实验；相同图像分区、候选协议、问题翻译与采样预算：

| 实验 | 回答的问题 |
|---|---|
| 原 BLIP→原 Laya | 已有基线 |
| BLIP→用相同 QA 微调过的 Laya | 收益是否只是新增监督，而非访问视觉细节 |
| native，视觉全局池化成一个 token | 是否真的需要密集局部信息 |
| native，576 patches 拼入原两层 head | 新交叉注意力是否比简单拼接更值得 |
| native，576 patches + gated cross-attention | 主方案 |
| native，换图/无图/局部证据移除 | 模型是否使用图像、是否只凭语言猜测 |
| ModernVBERT 基础主干 + 动态选项头 | 已对齐主干是否比新增融合更划算 |
| 人工正确描述→Laya | 错误归因诊断，不称为严格性能上界 |

对 ModernVBERT 和高分辨率方案，额外记录实际像素预算、tile 数、融合 token 数、训练更新时间与推理成本。先比较实际配置的质量/成本，再做匹配视觉预算的结构消融；不能只控制 QA 数量就把差异全部归因于融合结构。

主测试集建议至少 2,000 张图，分别统计属性、关系、计数、存在性与中文/英文，不用五张 demo 的总正确率替代。统计区间按图片 group bootstrap，多问题/翻译不作为独立随机样本；候选顺序置换还原语义后比较答案。

关键指标：choice accuracy/macro-F1；noul precision/recall、false-positive；计数 exact accuracy、MAE 与 ordinal RPS；概率 NLL/Brier/ECE、risk-coverage；纯文本回归；每图1/4/16问、各分辨率的 p50/p95、显存和冷启动。p95 用足够多重复样本并同步 GPU，不能靠五次调用估计。

更强的视觉因果检查应使用成对图片：同一问题、只改一个目标属性/位置/数量，两张都答对。单纯把图片换成随机噪声后准确率下降，只能说明输入扰动有影响，不能证明理解了目标细节。也不能只打乱 patch 的数组顺序：如果特征携带位置、交叉注意力对 K/V 对的共同置换不变，这种操作可能根本没有破坏空间信息；应修改像素布局或明确打乱“特征—坐标”对应。

成对样本保持同一候选集合，并验证修改后的真值；CLEVR 可采用“编辑场景→重渲染→执行问题程序”。VQAv2 的互补图像对也可用于同题不同图测试。另设不读取图片的 question/options-only 基线，记录真值不在候选、证据不足、简单/复杂计数等失败切片。黑图是诊断扰动，不能替代真正的无图分支。

**建议的 go/no-go 门槛（不是已经达到的结果）：** 主目标细节集相对“微调后的 caption 基线”提高至少 5 个百分点，图片级置信区间下界大于 0；计数 MAE 相对降低约 20%；可控反事实成对正确率 ≥90%；文本任务绝对退化 ≤2 个百分点。门槛需按业务错误代价确定，若基本指标不改善，不进入昂贵的全主干解冻。

## 7. 算力、时间与工程任务

按本地多语言 checkpoint 的 safetensors 文件头及 PyTorch meta 模型进行了**参数计数审计**，没有下载新的视觉权重、执行原生融合前向或训练：

| 部件 | 参数量 |
|---|---:|
| 现有 Laya（不含 3 个温度 buffer） | 321,908,995 |
| SigLIP 2 视觉塔，去掉 MAP pooling | 86,088,960 |
| 投影器，含 LayerNorm | 1,182,720 |
| 两处完整 768 维 MHA + LayerNorm + gate | 4,727,810 |
| **新模型合计** | **413,908,485** |
| **阶段 1 可训练参数** | **5,910,530** |
| **阶段 2 加训原两层 head 和 scorer** | **20,679,171** |

该口径不含后续 tiles 的新位置参数、LoRA 或辅助头；阶段 2 保持 type embedding 和 action head 冻结。复现命令为 `.venv/bin/python research/vision/native_parameter_audit.py`，结果见 [native-parameter-audit.json](../../artifacts/vision/native-parameter-audit.json)。meta 只建立形状，不验证模型能学到视觉能力。

冻结主干的原型和全模型训练是不同成本等级。建议先做 200 个稳定 training steps，记录每步耗时、峰值 allocated/reserved memory、数据加载时间与验证开销，再估算整个实验。

- 32 GiB M1 Pro：适合接口/梯度验证、小型冻结特征原型；统一内存不等同独立 32 GB CUDA 显存，PyTorch 算子与混合精度支持要实测。上一轮推理成功不能证明训练吞吐合适。
- 24 GB CUDA：可作为单卡冻结双主干 + 小微批融合训练的试验目标，不是已验证的显存保证；先开 SDPA、按实际长度分桶。
- 48 GB CUDA：增加分辨率、批量或部分 LoRA 适配的更宽裕试验档位。
- 80 GB/多卡：用于更大的多视图训练或广泛解冻，首轮无需从这里开始。

训练参数的 AdamW 状态、权重及梯度可按每参数约 12–16 字节做粗估，冻结 BF16/FP16 权重约 2 字节/参数，另加激活、注意力、临时张量和框架开销。该估算不是峰值显存；是否持有 FP32 master weights 等细节会改变数值。

冻结视觉特征缓存同样有成本：每张 `576*768*2` 字节约 0.844 MiB，5 万张约 41.2 GiB（只计一种视图的 FP16/BF16 特征）；应放磁盘分片而非全部留在内存。每次增加 tiles 或保存更多层，容量相应增加。

例如总训练流 20 万条问题、有效 batch 128、3 epochs，约 4,689 次更新。若实测每步 1 s，则仅训练更新约 1.3 h；若 5 s，则约 6.5 h。两者都只是算术示例，未包含特征提取、评测、数据准备、失败重跑和多随机种子，不能据此报价或承诺完成日期。

工程建议分四个可审查阶段：

1. 数据分区与固定评测；保持旧 caption 方案作为对照。
2. 独立 `multimodal_model.py`/processor 与训练器；不要在已训练文本模型中静默插入未训练参数。
3. 显式 checkpoint 类型/版本与加载器、原生图片运行时；迁移允许缺失的新增参数需列白名单，旧参数仍严格校验。新 checkpoint 缺视觉权重时必须失败，不用全面 `strict=False` 掩盖问题。
4. 通过核心门槛后，再扩 HTTP/MCP/ONNX/TileLang 等后端。现有文本 fast path 不认识视觉参数，应明确拒绝或走已验证的普通路径，不可静默忽略图片。

未来代码落点包括 `common.py` 的模型构建/评分复用、`agent.py` 的加载与批量图像映射、独立视觉 processor/训练脚本、校准工具、`tests/test_hooks_api.py` 的接口契约与新多模态测试。所有推理与训练组件在用户进程或自有硬件上运行，无需托管服务依赖。

训练实现特别注意：冻结参数不等于把整个含适配器的网络包在 `no_grad()` 中。冻结且位于可训练模块之前的两侧主干可用 `no_grad()`；融合后的旧 head 即使冻结，也要保留对其输入的梯度，否则上游融合学不到。`inference_mode()` 产生的张量直接送入需要保存输入以反传的 Linear 可能报错，训练特征提取宜用 `no_grad()` 或转换为普通张量。原 `Agent.predict` 是推理入口，训练器应调用模型 forward 并处理损失，不复用带推理装饰器的预测 API。

现有训练 notebook 按参数名是否含 `encoder.` 分组，新增 `vision.encoder` 后会发生误归类，应改成明确的模块参数集合。迁移先严格加载旧模型，再正常初始化新增层；当前 `build_model(pretrained=False)` 跳过初始化，不能把新增缺失参数留在这条路径上。至少留下加载往返、无图 logits 一致、图像索引隔离、适配层有效梯度、冻结参数不变、候选置换和视觉 ablation 检查。

## 8. 首个实际交付目标

优先完成一个自然图像细节版：**mmBERT + SigLIP 2 密集 patches + 两处交叉注意力，冻结主干起步，覆盖中英文属性/存在/空间关系/有限计数，输出仍是类型化分布。** 与微调 caption 基线和 ModernVBERT 对照后，再决定继续原生融合、换主干，还是为 OCR/文档另设高分辨率版本。

真正的成功标准是：独立细节任务改善，反事实图片带来正确的答案变化，文本能力可控、概率经过验证、资源成本可接受。仅有图片参数、前向成功或训练集 loss 下降都不满足这个标准。
