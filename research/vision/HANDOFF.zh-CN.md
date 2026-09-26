# 换设备接续：原生多模态 Laya

## 当前状态

本仓库保留两部分工作：

1. **已实现并验证：** 本地 BLIP Caption → Laya 桥接、API 契约测试、五张公开图片的 smoke 评测和细节探针。结果不能视作独立泛化成绩。
2. **已设计、尚未实现训练：** [原生多模态方案](NATIVE_MULTIMODAL_PLAN.zh-CN.md)，即 mmBERT + SigLIP 2 密集 patches + 两处门控交叉注意力。当前只有可运行的参数审计，尚无原生模型 forward、训练器、训练数据分区或训练后的 checkpoint。

上游源码起点：`4066d5d5fbf08b66c6757ddeedbd797bd7655bc0`。原理分析见 [PRINCIPLES.zh-CN.md](PRINCIPLES.zh-CN.md)，Caption 基线的复现及测量见 [README.zh-CN.md](README.zh-CN.md)。

## 在新设备恢复环境

```bash
git clone git@github.com:badpx/Laya-Vision.git
cd Laya-Vision
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

NVIDIA 设备先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装与驱动兼容、相互匹配的 PyTorch/torchvision CUDA 构建，再安装项目和测试工具：

```bash
python -m pip install -e '.[vision]' pytest ruff
python -m pip check
python -c 'import torch, transformers; print("torch:", torch.__version__, "transformers:", transformers.__version__); print("CUDA:", torch.cuda.is_available(), "runtime:", torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable")'
```

[requirements-tested.txt](requirements-tested.txt) 是原 Mac/Python 3.12 的精确实测版本记录，不是已经验证过的 Linux/CUDA 训练锁文件。参数审计在 Transformers 5.17.0 上验证；不同版本可能有模块布局差异。新机器确认环境可用后，将自己的 `pip freeze`、GPU/驱动、CUDA 和精度配置保存到单独实验目录，再固定用于对照实验。不要覆盖已有 Mac 实测结果。

先运行无需下载新权重的检查：

```bash
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/
python -m pytest tests/test_vision.py -q
python tests/test_hooks_api.py
python research/vision/native_parameter_audit.py --output /tmp/laya-native-parameter-audit.json
```

参数审计使用 meta 张量，仅验证形状和参数量；没有本地 Laya checkpoint 时会记录 `local_checkpoint_header_audit.status=not_present`，其余计数仍可运行。期望新增可训练参数 5,910,530，总参数 413,908,485。它不是训练可用性或性能测试。

## 恢复 Caption 基线（用于公平对照）

权重、测试图片、虚拟环境和包含本机绝对路径的 `models.json` 不随 Git 传输。在新机器重新生成：

```bash
python research/vision/prepare.py
python examples/vision.py artifacts/vision/images/chelsea.png --device cuda
HF_HUB_OFFLINE=1 python research/vision/evaluate.py --device cuda --language en --output /tmp/laya-vision-cuda-en.json
HF_HUB_OFFLINE=1 python research/vision/evaluate.py --device cuda --language zh --output /tmp/laya-vision-cuda-zh.json
```

下载脚本固定 Laya/BLIP 的 revision，并校验权重 SHA-256；备用分段下载见基线 README。CUDA 命令供新设备验证，原设备只验证过 CPU/MPS。Git 中的历史 JSON 保留测量值，模型路径的本机仓库前缀已统一改为 `./`；运行时应使用新生成的 `artifacts/vision/models.json`。

## 建议从这里继续实现

1. 阅读根目录 `AGENTS.md` 与原生方案。先固定按图片去重分区的数据清单、候选协议、许可和独立评测；不要先训练再挑有利测试集。
2. 实现独立 native model/processor/collator：每批唯一图片编码一次，问题通过 image index 读取视觉特征。保留无图旁路，正常初始化新层，明确 checkpoint 架构版本并严格加载。先禁用尚未适配的 FastLaya/ONNX 路径。
3. 用 128–256 条可靠合成样本验证梯度、过拟合、图像索引和 checkpoint 往返。冻结的旧 head 在视觉分支之后仍须传递输入梯度。
4. 实测约 200 个稳定训练步骤的吞吐和峰值内存，再选择微批和累积步数。阶段 1 只训约 591 万新增参数；阶段 2 加训旧 head/scorer，保留文本重放。
5. 先做 10 万原始 QA pilot，再扩量。同时保留原 Caption、同数据微调 Caption、原生融合以及 ModernVBERT 对照。细节、计数、成对反事实、文本回归和校准共同决定是否继续。

第一项训练交付应是自然图像属性/关系/有限计数模型；高分辨率 OCR、多图、后端加速另行验证。具体损失、数据配额、参数分组、反事实要求与验收门槛以[完整方案](NATIVE_MULTIMODAL_PLAN.zh-CN.md)为准。
