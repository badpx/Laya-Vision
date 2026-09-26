# Laya 本地视觉扩展：实施、复现与测试

本目录对应 2026-09-26 的研究与本地实施。完整技术分析见 [PRINCIPLES.zh-CN.md](PRINCIPLES.zh-CN.md)。上游源码基线为 `4066d5d5fbf08b66c6757ddeedbd797bd7655bc0`，包版本为 0.3.20。

历史评测 JSON 中本机仓库的绝对路径前缀已统一为 `./`，预测值与耗时保持原测量记录。

## 实施范围

已新增 `laya.VisionAgent` 和 `laya.BlipCaptioner`。流程为：本地图片像素 → BLIP 图像描述 → 描述加可选业务文本 → 原有 Laya → `choice` / `noul` / `score`。

这是**系统层面的视觉接入**。Laya 本身的文本权重与决策头未重新训练；该扩展不是原生图文特征融合模型，也不具备通过任意截图完成复杂操作的已验证能力。BLIP 生成描述，因此整个流水线不是“零 token 生成”或“单次前向”。

支持本地文件路径与 PIL 图片，自动处理 EXIF 方向和 RGB。输入限定为一张不超过 2000 万像素的静态图像；URL、动画、多页图像及损坏文件会明确报错。模型依赖懒加载，普通 `import laya` 不加载视觉栈。图片不会发往外部推理服务；首次权重下载与本地图片推理是两件独立的事。

`vision` 返回字段保留完整描述、送入 Laya 之前的文本、模型路径、预处理/描述/Laya/总耗时，并明确：

- `mode="caption_bridge"`：视觉文本桥接。
- `probability_scope="text_surrogate_only"`：概率只针对代理描述，没有包含视觉识别错误。
- `end_to_end_single_pass=false`：整个流水线含描述生成。
- `usage_scope="laya_text_stage_only"`：原 `usage` 不包括 BLIP 生成量。

原始 `answers`、`usage`、`routing` 等返回值及 hooks/其他预测参数继续透传。`vision.state` 是 hooks 和分词之前的输入；hooks 仍可能改写内容，超长文本仍受 Laya token 预算限制。描述放在上下文前面以减少被长上下文挤掉的风险。

## 本地运行

在项目根目录执行：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[vision]' \
  -r research/vision/requirements-tested.txt
.venv/bin/python research/vision/prepare.py
.venv/bin/python examples/vision.py artifacts/vision/images/chelsea.png --device mps
```

没有 Apple GPU 时将 `mps` 换成 `cpu`。NVIDIA 可用 `cuda`，但本次未验证 CUDA。macOS 的 GPU 在受限沙箱中可能不可见，本次 MPS 测试在获得系统设备访问权限的本机进程运行。

下载阶段使用固定模型提交。标准 Hugging Face 下载过慢时可用同目录的分段下载工具，它会检查 HTTP 分段响应并验证所有文件的 SHA-256 或 Git blob 哈希：

```bash
.venv/bin/python research/vision/prepare_ranges.py
```

本机直连 Hugging Face 超时，实测下载使用 `HF_ENDPOINT=https://hf-mirror.com` 的公开镜像；模型版本固定，记录的哈希校验通过。该哈希与镜像提供的固定版本元数据一致，不表示另行完成了密码学上的发布者签名验证。PyPI 下载使用清华镜像解决连接速度问题；推理本身不依赖这些镜像。

`prepare.py` 的标准缓存位于 `downloads/huggingface/`，备用工具的模型目录位于 `downloads/vision-models/`。这两个路径均被 Git 忽略，也不会与 Zensical 的 `.cache/` 构建清理冲突。图片缓存在 `artifacts/vision/images/`。模型清单 `artifacts/vision/models.json` 记录本机实际路径，示例程序从该清单加载模型；也可以显式指定 `--laya-model` 与 `--vision-model` 两个本地路径。

完成下载后，示例程序设置 `HF_HUB_OFFLINE=1`，BLIP 同时使用 `local_files_only=True`。已缓存的示例和评测可以离线运行，无需 API key。

Python 调用示例：

```python
from laya import BlipCaptioner, VisionAgent, load

agent = load("downloads/vision-models/laya/multilingual", device="mps")
captioner = BlipCaptioner("downloads/vision-models/blip", device="mps", local_files_only=True)
vision = VisionAgent(agent, captioner)
result = vision.predict(
    "artifacts/vision/images/chelsea.png",
    {"animal": {"type": "noul", "instructions": "图像中有动物吗？"}},
    state="可选业务上下文",
)
print(result["vision"]["caption"])
print(result["answers"])
```

上述本地模型路径对应备用下载工具。使用标准下载工具时请从 `models.json` 获取路径，或使用 `examples/vision.py` 自动读取它。

## 测试方法与复现

`evaluate.py` 使用五张 scikit-image 官方公开示例：宇航员、摄影者、猫、咖啡、马的剪影。图片固定在 `v0.25.2` 并校验上游注册的 SHA-256，来源保存在 `artifacts/vision/fixtures.json`。标签经人工检查；文件名和标签不传入模型。

每图询问主类别、是否有人、是否有动物、人数等级四项。`score` 的可用性按等级分布 argmax 与标签比较；API 本身仍返回期望分数。总计每种语言 20 项标签，覆盖三种 API 类型。这些公开图片可能出现在模型预训练中，因此只是小规模集成 smoke test，不能作为独立泛化基准或部署准确率。

四组输入使用相同问题及文本封装：真实 BLIP 描述、人工正确描述、交换到另一张图的描述、没有视觉描述。后两组是**文本瓶颈消融**，没有假称重新训练模型或遮蔽图像 patch。它们用于检查答案是否受到视觉描述影响。

```bash
HF_HUB_OFFLINE=1 .venv/bin/python research/vision/evaluate.py \
  --device mps --language en --output artifacts/vision/evaluation-mps-en.json
HF_HUB_OFFLINE=1 .venv/bin/python research/vision/evaluate.py \
  --device mps --language zh --output artifacts/vision/evaluation-mps-zh.json
HF_HUB_OFFLINE=1 .venv/bin/python research/vision/evaluate.py \
  --device cpu --language en --output artifacts/vision/evaluation-cpu-en.json
HF_HUB_OFFLINE=1 .venv/bin/python research/vision/probes.py --device mps
```

`probes.py` 复用已记录的描述，进一步测试服装颜色、支撑物、眼睛颜色、杯子外侧颜色、朝向等细节，并与包含正确细节的人工描述对照。它另用两个相反的文字分流规则测试同一张猫图的上下文敏感性。每次问法和标签均保存在输出中。

模型配置与版本：

| 项目 | 固定版本 |
|---|---|
| Laya multilingual | `convaiinnovations/laya` 的 `multilingual/`，提交 `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` |
| BLIP caption base | `Salesforce/blip-image-captioning-base`，提交 `82a37760796d32b1411fe092ab5d4e227313294b` |
| 运行依赖 | Python 3.12.11；精确依赖见 `requirements-tested.txt` |
| 硬件 | Apple M1 Pro，32 GiB 内存，macOS arm64 |

## 真实模型结果

以下结果使用固定问题完成，未按测试结果微调权重、拟合温度或反复挑选提示词。性能计时来自串行运行，排除了下载、权重加载和首个 warmup；只有 5 张图片，不足以估计稳定的 p95。

| 设备/问题语言 | 真实视觉描述 | 人工正确描述 | 错配描述 | 无视觉描述 | 每图 4 问总耗时中位数 |
|---|---:|---:|---:|---:|---:|
| MPS / 英文 | 16/20 | 16/20 | 8/20 | 8/20 | 296 ms |
| MPS / 中文 | 17/20 | 14/20 | 7/20 | 10/20 | 249 ms |
| CPU / 英文，4 线程 | 16/20 | 16/20 | 8/20 | 8/20 | 448 ms |

分项：主类别 `choice` 两种语言均为 5/5，是否有动物均为 5/5；是否有人英文 4/5、中文 5/5；人数等级 `score` 两种语言均为 2/5。这里的 80%/85% **仅是这 20 项 smoke 标签上的比例，不是通用视觉准确率**。同一张图片的多个判断相互关联，不能把它们当成 20 个独立随机样本。

人工描述对照用于区分错误来源，并不是“应当更好的模型”。中文下人工描述下降到 14/20，说明模型明显受文本表达影响；本次没有为了提高数字重新改写问题。它也说明修正视觉前端并不能自动修复 Laya 的文本判断。

MPS 英文的描述生成中位数约 198 ms、Laya 的四问批处理约 61 ms；CPU 对应约 359 ms、79 ms。总耗时中位数不是各分段中位数的和。首个 MPS 英文进程的 Laya 加载约 4.18 s，首次视觉调用约 5.64 s（含 BLIP 延迟加载与首次推理）；不包括下载和 Python 导入。后续进程可能受操作系统文件缓存影响，不能当作全新机器的冷启动承诺。

典型成功与失败：

| 图片 | BLIP 实际描述 | 观察 |
|---|---|---|
| 宇航员 | `a young man in an orange space suit holding a helmet` | 主类别正确，但原图为女性，描述中的人物属性有错误；基础 20 项标签没有覆盖该属性 |
| 摄影者 | `a man in a suit taking a picture of a building` | 漏掉相机三脚架；英文“是否有人”还被 Laya 判为否，`P(true)=0.2996` |
| 猫 | `a cat with green eyes` | 动物与眼睛颜色有正确证据；人数等级仍错误地偏向 1 |
| 咖啡 | `a cup of coffee on a wooden table` | 正确识别饮品，但没有杯子外侧红色的信息 |
| 马剪影 | `a horse silhouetted on a white background` | 主类别正确，朝向信息未写入描述 |

五个细节探针中，真实描述答对 **3/5**，人工加入正确细节后 **5/5**。摄影者问题选成“仅用手支撑”（置信度 0.8843），杯子外侧选成白色；马的朝向虽然猜中，描述并没有提供朝向，不能据此证明具备可靠空间理解。把同一张猫图配上两条相反的文字分流政策时，队列答案按政策改变，两个案例均正确，证明可选文字上下文能参与判断，但样本量不足以证明任意图文规则推理。

原始结果包含每个问题、完整概率、预期标签、模型版本、设备信息与各阶段耗时：

- [MPS 英文](../../artifacts/vision/evaluation-mps-en.json)
- [MPS 中文](../../artifacts/vision/evaluation-mps-zh.json)
- [CPU 英文](../../artifacts/vision/evaluation-cpu-en.json)
- [细节与上下文探针](../../artifacts/vision/probes.json)
- [可直接运行的猫图示例输出](../../artifacts/vision/example-cat.json)

## 工程检查

新增视觉边界/错误/EXIF/懒加载/参数透传测试 35 项通过，API 契约测试 154 项通过；原有路由 546 项、criteria 139 项、hooks 162 项、训练辅助函数 61 项通过。hooks 测试的故意异常路径产生既有未 await coroutine 警告，测试断言全部通过。

仓库要求的 Ruff 检查、`compileall`、`git diff --check` 通过；Zensical 严格文档构建通过且没有 `griffe:` 警告。没有修改或测试 CUDA TileLang/ONNX 视觉快路径，因为新增模块不向这些后端注入视觉 token。

```bash
.venv/bin/python -m pytest tests/test_vision.py -q
.venv/bin/python tests/test_hooks_api.py
.venv/bin/ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
.venv/bin/python -m compileall -q laya/ tests/
uv pip install --python .venv/bin/python -r requirements-docs.txt
.venv/bin/zensical build --strict --clean
```

## 能力边界

视觉描述会丢失细节，Laya 无法从丢失的信息恢复像素。OCR、精确计数、空间关系、图表、细小对象、多图比较和长上下文需要专门验证，当前模块没有提供 OCR 或多图功能。Laya 对错误或不完整描述仍可能输出很高的概率；本次没有给这个小样本集拟合温度，也没有宣称端到端概率已校准。

普通物体和场景的初步分流可从这个可运行的基线继续评估。若产品要求原生像素融合、复杂图文推理或保留文本模型的毫秒级速度，需要另外建设图文监督集，训练视觉投影与融合层，并重新测量校准、延迟和泛化。具体方案见原理报告第 6 节。
